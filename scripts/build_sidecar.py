"""Bundle the Python sidecar into a single executable for Tauri.

Tauri's ``externalBin`` mechanism requires the binary to be named with the
Rust target triple appended, e.g. ``meeting-notes-server-x86_64-pc-windows-msvc.exe``.
This script builds it and puts it where ``tauri.conf.json`` expects.

    pip install pyinstaller
    python scripts/build_sidecar.py

Then::

    cd desktop && npm run tauri build

The bundle deliberately excludes torch: the runtime path is CTranslate2
(faster-whisper) plus onnxruntime (sherpa-onnx). If PyInstaller starts pulling
torch in, something has re-imported the NeMo export path at runtime and the
installer will grow by about a gigabyte.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY = ROOT / "src" / "meeting_notes" / "sidecar.py"
OUT_DIR = ROOT / "desktop" / "src-tauri" / "binaries"
BINARY_STEM = "meeting-notes-server"

#: Pulled in dynamically, so PyInstaller's static analysis misses them.
HIDDEN_IMPORTS = [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "meeting_notes.backends.whisper",
    "meeting_notes.backends.nemo",
]

#: Packages whose native libraries and data files must ship whole.
#: sherpa-onnx carries its ONNX runtime in .pyd/.dll files inside the
#: package — without --collect-all the frozen sidecar imports it fine on the
#: build machine and dies on a clean one. Speaker diarization needs it.
COLLECT_ALL = [
    "sherpa_onnx",
]

#: Kept out of the bundle. torch is the big one — see the module docstring.
EXCLUDES = [
    "torch",
    "torchaudio",
    "nemo",
    "nemo_toolkit",
    "matplotlib",
    "tkinter",
    "pytest",
    "datasets",
]


def target_triple() -> str:
    """The Rust target triple Tauri expects in the binary's filename."""
    try:
        out = subprocess.check_output(["rustc", "-vV"], text=True)
    except (OSError, subprocess.CalledProcessError):
        sys.exit(
            "Could not run `rustc -vV`. Install Rust (https://rustup.rs) — "
            "Tauri needs it, and the sidecar filename must carry its target "
            "triple."
        )
    for line in out.splitlines():
        if line.startswith("host:"):
            return line.split(":", 1)[1].strip()
    sys.exit("Could not parse the host triple from `rustc -vV`.")


def build(debug: bool = False) -> Path:
    try:
        import PyInstaller  # noqa: F401, PLC0415
    except ImportError:
        sys.exit("PyInstaller is not installed. Run: pip install pyinstaller")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work = ROOT / "build" / "sidecar"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(ENTRY),
        "--name",
        BINARY_STEM,
        "--onefile",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(OUT_DIR),
        "--workpath",
        str(work),
        "--specpath",
        str(work),
        "--paths",
        str(ROOT / "src"),
    ]
    if not debug:
        # The sidecar has no console UI; its stdout is a machine-readable
        # handshake and its logs go to stderr.
        cmd.append("--console")

    for name in HIDDEN_IMPORTS:
        cmd += ["--hidden-import", name]
    for name in COLLECT_ALL:
        cmd += ["--collect-all", name]
    for name in EXCLUDES:
        cmd += ["--exclude-module", name]

    print("Running:", " ".join(cmd), "\n")
    subprocess.check_call(cmd)

    suffix = ".exe" if sys.platform == "win32" else ""
    built = OUT_DIR / f"{BINARY_STEM}{suffix}"
    if not built.exists():
        sys.exit(f"Expected {built} but PyInstaller did not produce it.")
    return built


def rename_for_tauri(built: Path) -> Path:
    """Append the target triple, as Tauri's externalBin resolution requires."""
    triple = target_triple()
    suffix = ".exe" if sys.platform == "win32" else ""
    target = built.with_name(f"{BINARY_STEM}-{triple}{suffix}")

    if target.exists():
        target.unlink()
    shutil.copy2(built, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="Unstripped build.")
    parser.add_argument(
        "--skip-rename",
        action="store_true",
        help="Do not append the Rust target triple (useful without Rust).",
    )
    args = parser.parse_args()

    built = build(debug=args.debug)
    size_mb = built.stat().st_size / 1048576
    print(f"\nBuilt {built.name} ({size_mb:.1f} MB)")

    if args.skip_rename:
        print("Skipped the target-triple rename; Tauri will not find this yet.")
        return 0

    target = rename_for_tauri(built)
    print(f"Placed {target.name} for Tauri externalBin")
    print("\nNext: cd desktop && npm run tauri build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
