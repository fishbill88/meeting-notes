"""Desktop sidecar: the Python process the Tauri shell spawns and owns.

Differs from ``meeting-notes serve`` in three ways that matter for a desktop
app handling patient audio:

* **Loopback, random port.** No fixed port to collide with, nothing reachable
  off the machine.
* **Bearer token generated per launch.** Any other local process would need to
  read this process's stdout to talk to it.
* **Handshake on stdout.** The port and token are unknowable in advance, so
  the first line of stdout is a JSON object the shell parses::

      {"event": "ready", "port": 51823, "token": "...", "pid": 4242}

  Nothing else is written to stdout — logs go to stderr — so the shell can
  read exactly one line and stop.

Run it directly for debugging::

    python -m meeting_notes.sidecar --log-level debug
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import socket
import sys
import threading
import time
from typing import Optional

from meeting_notes.server import TOKEN_ENV, app, generate_token, set_shutdown_hook

logger = logging.getLogger("meeting_notes.sidecar")

#: Loopback only. Not configurable on purpose — see the module docstring.
HOST = "127.0.0.1"

#: The desktop shell puts its own pid here so the watchdog can watch the
#: process that actually matters. See :func:`watch_target_pid`.
PARENT_PID_ENV = "MEETING_NOTES_PARENT_PID"


def find_free_port(host: str = HOST) -> int:
    """Ask the OS for an unused port.

    Binding to port 0 and reading back the assignment races with any other
    process that grabs it in between, but the alternative — a fixed port — is
    worse: it collides with whatever else the clinic runs, and it makes the
    endpoint guessable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def watch_target_pid() -> int:
    """Which process this sidecar should die with.

    ``getppid()`` is not good enough. PyInstaller's one-file bundle runs the
    real program as a *child of a bootloader*, so the immediate parent is that
    bootloader, not the desktop shell. Force-killing the shell (Task Manager,
    a crash) orphans the bootloader — which then keeps running, so the child
    never observes a parent death and both survive holding a port and a loaded
    speech model.

    The shell therefore passes its own pid in :data:`PARENT_PID_ENV`, and that
    wins when present. ``getppid()`` remains the fallback for running the
    sidecar directly.
    """
    declared = os.environ.get(PARENT_PID_ENV, "").strip()
    if declared.isdigit():
        return int(declared)
    return os.getppid()


def watch_parent(on_parent_exit: "callable") -> threading.Thread:
    """Exit when the process named by :func:`watch_target_pid` does.

    Not belt-and-braces — this is the only thing that reliably prevents
    orphaned sidecars, and the shell's own cleanup does not cover a hard kill.

    On Windows this blocks on a handle to that process, so the reaction is
    immediate rather than polled.

    Returns the daemon thread, already started.
    """
    def _wait() -> None:
        try:
            # Resolved inside the guard: a watchdog that cannot identify its
            # parent must degrade to doing nothing, not take the server down.
            parent_pid = watch_target_pid()

            if os.name == "nt":
                import ctypes  # noqa: PLC0415

                SYNCHRONIZE = 0x0010_0000
                INFINITE = 0xFFFF_FFFF

                handle = ctypes.windll.kernel32.OpenProcess(
                    SYNCHRONIZE, False, parent_pid
                )
                if not handle:
                    # Parent already gone, or not visible to us.
                    on_parent_exit()
                    return
                try:
                    ctypes.windll.kernel32.WaitForSingleObject(handle, INFINITE)
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
            else:
                # POSIX re-parents orphans, so a changed ppid means the
                # original parent has exited.
                while os.getppid() == parent_pid:
                    time.sleep(1.0)
        except Exception:  # noqa: BLE001 - a watchdog must never take us down
            logger.exception("parent watchdog failed; continuing without it")
            return

        logger.warning("parent process %s exited; shutting down", parent_pid)
        on_parent_exit()

    thread = threading.Thread(target=_wait, name="parent-watchdog", daemon=True)
    thread.start()
    return thread


def emit_handshake(port: int, token: str) -> None:
    """Write the one line of stdout the desktop shell reads."""
    payload = {
        "event": "ready",
        "host": HOST,
        "port": port,
        "token": token,
        "pid": os.getpid(),
    }
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meeting Notes desktop sidecar")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to bind. 0 (default) asks the OS for a free one.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token. Generated per launch when omitted.",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=("critical", "error", "warning", "info", "debug"),
        help="Uvicorn log level. Logs go to stderr, never stdout.",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable the bearer token. Debugging only — never ship this.",
    )
    parser.add_argument(
        "--no-parent-watch",
        action="store_true",
        help="Keep running after the parent exits. Debugging only — this is "
        "what stops orphaned sidecars.",
    )
    return parser.parse_args(argv)


def build_server(port: int, log_level: str):
    """Construct the uvicorn Server without starting it."""
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise RuntimeError(
            "uvicorn is required for the sidecar. "
            "Install it with: pip install 'meeting-notes[server]'"
        ) from exc

    config = uvicorn.Config(
        app,
        host=HOST,
        port=port,
        log_level=log_level,
        access_log=False,
    )
    return uvicorn.Server(config)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    port = args.port or find_free_port()
    token = "" if args.no_auth else (args.token or generate_token())

    # The server reads the token from the environment, so a child process
    # inherits nothing a parent did not deliberately grant.
    if token:
        os.environ[TOKEN_ENV] = token
    else:
        os.environ.pop(TOKEN_ENV, None)
        logger.warning("Running without authentication — debugging only.")

    server = build_server(port, args.log_level)

    def _stop() -> None:
        # Uvicorn watches this flag in its own loop; setting it from the
        # request handler lets the response finish before the process exits.
        server.should_exit = True

    set_shutdown_hook(_stop)

    if not args.no_parent_watch:
        watch_parent(_stop)

    # The handshake has to reach the shell before serving starts, otherwise a
    # fast shell polls a port that is not listening yet and reports a hang.
    # Uvicorn binds during startup, so emit once startup has completed.
    def _announce() -> None:
        started = threading.Event()

        def _watch() -> None:
            while not server.started and not server.should_exit:
                started.wait(0.02)
            if server.started:
                emit_handshake(port, token)

        threading.Thread(target=_watch, daemon=True).start()

    _announce()

    try:
        server.run()
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        pass
    finally:
        set_shutdown_hook(None)
        # The language-model server (if one was started) holds ~3 GB; stop it with the engine. A hard kill
        # is covered separately, by the job object it was started under.
        from meeting_notes.llm import shutdown_runtime  # noqa: PLC0415

        shutdown_runtime()

    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        raise SystemExit(main())
