//! Owns the Python sidecar's lifetime.
//!
//! The sidecar picks its own port and generates its own token, so neither is
//! knowable in advance. It writes exactly one JSON line to stdout when it is
//! listening; this module reads that line, stores it, and hands it to the
//! frontend through the `sidecar_info` command.
//!
//! The other half of the job is making sure the sidecar dies. An orphaned
//! sidecar holds a speech model in memory and a port open, so it is killed on
//! window close, on app exit, and on drop.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::{RunEvent, State, WindowEvent};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Set to `1` to spawn `python -m meeting_notes.sidecar` instead of the
/// bundled binary. Lets the shell run before PyInstaller is wired up.
const DEV_ENV: &str = "MEETING_NOTES_SIDECAR_DEV";

/// Parsed from the sidecar's stdout handshake.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarInfo {
    pub host: String,
    pub port: u16,
    pub token: String,
    pub pid: u32,
}

#[derive(Default)]
pub struct SidecarState {
    info: Mutex<Option<SidecarInfo>>,
    child: Mutex<Option<CommandChild>>,
}

impl SidecarState {
    fn set_info(&self, info: SidecarInfo) {
        *self.info.lock().unwrap() = Some(info);
    }

    fn set_child(&self, child: CommandChild) {
        *self.child.lock().unwrap() = Some(child);
    }

    /// Ask the sidecar to exit. Idempotent — safe to call from every hook.
    fn terminate(&self) {
        if let Some(child) = self.child.lock().unwrap().take() {
            // The HTTP /shutdown route is the graceful path and the frontend
            // uses it. This is the backstop for when the window is destroyed
            // without the frontend getting a chance to run.
            let _ = child.kill();
        }
    }
}

impl Drop for SidecarState {
    fn drop(&mut self) {
        self.terminate();
    }
}

/// Connection details for the running sidecar.
///
/// Errors while the handshake has not arrived yet; the frontend's health-poll
/// loop retries, so this doubles as the readiness signal.
#[tauri::command]
fn sidecar_info(state: State<'_, Arc<SidecarState>>) -> Result<SidecarInfo, String> {
    state
        .info
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "Sidecar is still starting".to_string())
}

fn spawn_sidecar(app: &tauri::AppHandle, state: Arc<SidecarState>) -> Result<(), String> {
    let dev_mode = std::env::var(DEV_ENV).map(|v| v == "1").unwrap_or(false);

    let command = if dev_mode {
        app.shell()
            .command("python")
            .args(["-m", "meeting_notes.sidecar"])
    } else {
        app.shell()
            .sidecar("meeting-notes-server")
            .map_err(|e| format!("sidecar binary not found: {e}"))?
    };

    // Tell the sidecar which process to die with. It cannot work this out
    // itself: under PyInstaller its parent is a bootloader, and a hard kill of
    // this process (Task Manager, a crash) orphans that bootloader rather than
    // ending it — leaving both alive with a port bound and a model resident.
    let command = command.env("MEETING_NOTES_PARENT_PID", std::process::id().to_string());

    let (mut rx, child) = command
        .spawn()
        .map_err(|e| format!("failed to spawn sidecar: {e}"))?;

    state.set_child(child);

    let handshake_state = state.clone();
    tauri::async_runtime::spawn(async move {
        let mut announced = false;

        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    if announced {
                        continue;
                    }
                    let line = String::from_utf8_lossy(&bytes);
                    match serde_json::from_str::<SidecarInfo>(line.trim()) {
                        Ok(info) => {
                            eprintln!("sidecar ready on port {}", info.port);
                            handshake_state.set_info(info);
                            announced = true;
                        }
                        Err(e) => {
                            // stdout is reserved for the handshake, so
                            // anything unparseable is a bug worth seeing.
                            eprintln!("unparseable sidecar stdout {line:?}: {e}");
                        }
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    eprint!("[sidecar] {}", String::from_utf8_lossy(&bytes));
                }
                CommandEvent::Terminated(payload) => {
                    eprintln!("sidecar exited: {payload:?}");
                    break;
                }
                _ => {}
            }
        }
    });

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = Arc::new(SidecarState::default());

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(state.clone())
        .invoke_handler(tauri::generate_handler![sidecar_info])
        .setup({
            let state = state.clone();
            move |app| {
                spawn_sidecar(app.handle(), state.clone())?;
                Ok(())
            }
        })
        .on_window_event({
            let state = state.clone();
            move |_window, event| {
                if matches!(event, WindowEvent::Destroyed) {
                    state.terminate();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error building Meeting Notes AI")
        .run(move |_app, event| {
            if matches!(event, RunEvent::Exit) {
                state.terminate();
            }
        });
}
