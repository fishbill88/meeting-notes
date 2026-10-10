# Meeting Notes AI — Desktop

Tauri 2 shell over the Python pipeline. The UI is TypeScript/React;
transcription and note generation run in a Python **sidecar** the shell spawns
and owns.

## Prerequisites

| Tool | Why |
|---|---|
| Node 18+ | Frontend build |
| **Rust** ([rustup.rs](https://rustup.rs)) | Tauri needs it. Not optional. |
| **MSVC C++ build tools** | Rust's `x86_64-pc-windows-msvc` target links through them. Visual Studio with the "Desktop development with C++" workload, or standalone Build Tools. |
| WebView2 runtime | Preinstalled on Windows 11. |
| Python 3.9+ with the project installed | The sidecar |
| PyInstaller | Only for producing a shippable bundle |

⚠️ `winget install Rustlang.Rustup` installs rustup but its silent flags make
`rustup-init` fail. Run `rustup-init.exe -y` directly, then
`rustup default stable`.

## Running it

Three ways, fastest first. All three are verified working on Windows 11.

### 1. Just run the built app — no toolchain needed

```powershell
& "d:\GitHub\meeting-notes\desktop\src-tauri\target\release\meeting-notes-desktop.exe"
```

Or install from `src-tauri/target/release/bundle/` — either the `.msi` or the
NSIS `-setup.exe`. Both are unsigned, so SmartScreen will warn: **More info →
Run anyway**.

### 2. Dev mode — hot reload for UI work

The shell spawns `python -m meeting_notes.sidecar` instead of the bundled
binary, so there is no PyInstaller step in the loop.

```powershell
cd d:\GitHub\meeting-notes
pip install -e ".[desktop,dev]"      # editable, so edits take effect

cd desktop
npm install
$env:MEETING_NOTES_SIDECAR_DEV = '1'
npm run tauri dev
```

First run compiles the Rust debug profile — allow 3–5 minutes. After that,
frontend edits hot-reload; Rust edits trigger a rebuild.

### 3. Frontend only — no Rust, no Python

```powershell
cd desktop
npm run dev          # Vite on :1420, opens in a browser
npm run typecheck    # must pass
npm test             # vitest
```

The sidecar is unreachable this way, so the app will sit on "Starting the
transcription engine…". Useful for pure layout work, nothing else.

### Rebuilding the shippable bundle

```powershell
pip install pyinstaller
python scripts/build_sidecar.py    # writes desktop/src-tauri/binaries/
cd desktop && npm run tauri build
```

⚠️ **Nothing may be running while `tauri build` runs.** An open app locks
`meeting-notes-desktop.exe` and the bundling step fails with "Access is
denied" or "used by another process".

## Before a demo

| Check | Why |
|---|---|
| **Ollama is running** | If it is not, notes silently fall back to a much weaker extractive summarizer. No error is shown. |
| **Model is pre-warmed** | `large-v3-turbo` is ~1.55 GB and downloads on first transcription. Run one throwaway transcription first. |
| **Note format** | Defaults to Meeting (summary, topics, decisions, action items). Consultation is still selectable on the record screen but is parked until the clinical phase. |
| Mic permission | Prompts on first recording. |

## How the sidecar handshake works

The sidecar's port and token cannot be known in advance, so:

1. Rust spawns the sidecar ([`src-tauri/src/lib.rs`](src-tauri/src/lib.rs)).
2. The sidecar binds a **random loopback port**, generates a **bearer token**,
   and writes exactly one JSON line to stdout:
   `{"event":"ready","host":"127.0.0.1","port":51823,"token":"…","pid":4242}`
3. Rust parses that line and stores it. Everything else the sidecar emits goes
   to stderr — stdout carries the handshake and nothing else.
4. The frontend calls the `sidecar_info` command, then polls `/health` until
   the engine has loaded.

`sidecar_info` errors while the handshake has not arrived. That is deliberate:
the frontend's health-poll loop retries, so the error doubles as the readiness
signal.

## Security posture

This process handles recorded meeting audio, so:

- **Loopback only.** `127.0.0.1`, not configurable. Verified not reachable on
  the machine's LAN address.
- **Bearer token on every route except `/health`.** Missing token → 401, wrong
  token → 403, compared in constant time.
- **CORS is an allow-list**, not `*` — see `ALLOWED_ORIGINS` in
  [`server.py`](../src/meeting_notes/server.py).
- **The sidecar cannot outlive the app.** It is killed on window destroy, on
  app exit and on drop, *and* it watches its own parent and exits when that
  parent goes away.

⚠️ The parent watchdog is not redundancy — it is the only thing that actually
works. PyInstaller's one-file bundle runs the real program as a **child of a
bootloader process**, so killing "the sidecar" kills the bootloader and leaves
the Python process alive, holding its port and a loaded speech model. Measured,
not theorised: before the watchdog, killing the bootloader reliably orphaned
the child. Force-quitting the shell had the same effect. `--no-parent-watch`
disables it for debugging; never ship with that set.

⚠️ The Vite dev server port (1420) is named in `ALLOWED_ORIGINS`. Changing it
in `vite.config.ts` means changing it there too, or the browser will silently
block requests in dev.

## Layout

```
desktop/
  public/pcm-worklet.js       — PCM capture worklet (served from 'self' for CSP)
  src/
    App.tsx                   — record → transcribe → correct → notes
    components/RecordPanel    — capture controls, level meter, source picker
    components/TranscriptEditor — segment-level editing (the priority feature)
    components/NotesView      — structured notes + engine provenance
    components/HighlightedText— term highlighting (consultation template only)
    components/Icons.tsx      — inline SVG (no emoji, no external icon package)
    lib/api.ts                — sidecar client (handshake, health, process, summarize)
    lib/recorder.ts           — Web Audio capture → 16 kHz mono WAV
    lib/medicalTerms.ts       — lexicon, mirrored from mobile (parked: consultation only)
    lib/types.ts              — wire contract, mirrored from mobile
  src-tauri/
    src/lib.rs                — sidecar lifecycle, handshake parsing, teardown
    tauri.conf.json           — window, CSP, bundle targets, externalBin
    capabilities/             — permissions (minimal on purpose)
```

`lib/types.ts` and `lib/medicalTerms.ts` are duplicated from `mobile/` rather
than shared through a package: the two apps ship on different cadences, and a
breaking change should surface as a type error here rather than silently at
runtime there.

## Recording

Capture happens in the webview, not in Rust — no native audio module to
maintain per platform. `AudioContext` is created at 16 kHz so the worklet emits
frames at the rate the pipeline wants, with no resampling. Output is 16-bit PCM
WAV rather than webm/opus: the pipeline can decode opus, but lossy compression
before ASR is a quality risk that buys nothing on local disk.

System-audio capture (for remote meetings on Zoom, Teams or Meet) uses
`getDisplayMedia` with the video track discarded. The user must tick "share
system audio" in the picker.

## Design

Generated with the `ui-ux-pro-max` skill: "Accessible & Ethical" style, calm
cyan + health green, WCAG-oriented. Two deliberate deviations, both documented
at the top of `src/styles.css`: **system fonts** instead of Figtree/Noto Sans
(the CSP blocks Google Fonts, and system UI fonts render natively), and cyan as
an accent rather than a full-window background wash.

## Testing

```bash
npm run typecheck   # must pass
npm test            # vitest — WAV encoder + lexicon
```

The WAV encoder is tested by reading the bytes back, because a wrong header
field produces a file that looks fine and transcribes to silence.

## Storage

Meetings are saved automatically once processed — losing one because nobody
pressed Save is not an acceptable failure mode, and the store is local and
encrypted so there is nothing to weigh against it.

Storage lives on the **Python** side (`storage.py`, `crypto.py`), not in Tauri,
so one process owns the recorded content end to end. Record content is
encrypted as a single payload; only creation time, duration, language and
engine stay in the clear, so the history list can sort and paginate without
decrypting everything. None of that identifies a participant.

**Audio is never persisted.** A recording is held in memory, uploaded, and the
server deletes the temp file. There is no audio-retention setting because there
is no audio to retain.

Read the threat model at the top of `src/meeting_notes/crypto.py` before
relying on the encryption. Key-file mode (default) protects against the
database being read somewhere other than this account — a stolen backup, a
synced folder. It does **not** protect against malware running as this user.
Passphrase mode (`MEETING_NOTES_PASSPHRASE`) does, at the cost of a forgotten
passphrase meaning the records are gone.

## Export

Three formats, rendered to bytes by the sidecar so a meeting never lands in a
temp directory on the way out:

- **Copy notes** — plain text, Markdown and checkbox syntax stripped. The
  fastest route into an email, ticket or wiki page, and the primary action.
- **Word (.docx)** and **Markdown (.md)** — file downloads.

⚠️ Downloads use an object URL rather than a native save dialog. A real save
dialog needs the Tauri fs plugin plus a capability grant — worth upgrading, but
this works in the webview today.

## Status

Phase 6 — **builds and runs.** Verified on Windows 11 x64: the release binary
launches, the sidecar handshake completes, the startup gate resolves, engine
availability is reported, and closing the window reaps both sidecar processes.

Works: startup gate, engine warning, mic and system-audio recording with level
meter and pause/resume, language and note-template selection, file import,
editable transcript with low-confidence flags, notes regenerated from
corrections, engine provenance, encrypted local storage, history with search
and delete, export and copy-to-clipboard, speaker diarization ("Person 1/2"
turn labels, click a name to rename, names persist with the meeting;
sherpa-onnx models download on first use, ~34 MB).

Not built yet: live captions (gated on RTF measurements that do not exist yet),
native save dialog, PDF export.

**Not yet exercised against real audio in the built app.** The transcription
path is verified through the HTTP layer, but nobody has recorded a live
meeting through the packaged application.

⚠️ **Installers are unsigned.** Windows SmartScreen will warn, and hospital IT
will likely block them. Code signing is the next packaging step.
