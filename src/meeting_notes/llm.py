"""A language model that ships with the engine: llama.cpp's server plus a GGUF, fetched on demand.

**Why this exists.** Notes written by a language model are what make Make notes worth pressing, and the
only model the engine could use was Ollama — a separate program every user would have to install and run.
The extractive fallback it drops to when Ollama is absent is, honestly, a sentence-picker. So the engine
can now bring its own: the upstream llama.cpp ``llama-server`` (a ~19 MB CPU build) and one pinned,
checksummed GGUF, downloaded when the person asks and kept under the models directory.

**Pinned, not "latest".** Every download names an exact file with its SHA-256, a size, and — for the model
— an exact Hugging Face *revision*. A changed upstream file fails the check instead of silently changing
what writes people's notes. Moving to a new model or a new llama.cpp is a deliberate edit to the catalog
below, after re-running the benchmark that chose it (see ``NOTES-PLAN.md``).

**Not shipped in the engine zip**, for the reason the engine is not shipped in Weave: the model is ~2.7 GB.

**Layout** under ``config.models_dir() / "llm"``::

    runtime/<tag>/llama-server.exe …    the unpacked llama.cpp build
    models/<file>.gguf                  the model
    installed.json                      written LAST: its presence means the whole install verified

**The server process** is started on first use and stopped after it has been idle (it holds the model,
~3 GB of RAM), and — via a Windows job object — dies with this process however this process dies, so a
killed engine cannot leave a model resident.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("meeting_notes.llm")


# ---------------------------------------------------------------------------
# The catalog: exactly what may be downloaded
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Asset:
    url: str
    sha256: str
    size: int
    filename: str


#: Upstream llama.cpp build, pinned. CPU only: it runs on every machine, where a GPU build depends on a
#: driver the company does not control. The zip carries a CPU DLL per micro-architecture and picks at run
#: time, so one asset serves old and new laptops. Digest published by GitHub for the release asset.
RUNTIME_TAG = "b11429"
RUNTIME = Asset(
    url=f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_TAG}/llama-{RUNTIME_TAG}-bin-win-cpu-x64.zip",
    sha256="1283323272b04cd07905816a597a0da810918102de958f4ff6f7bbaa70ed2efe",
    size=19_398_918,
    filename=f"llama-{RUNTIME_TAG}-bin-win-cpu-x64.zip",
)
RUNTIME_EXE = "llama-server.exe"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    asset: Asset
    license: str


#: The models the engine may install. Hashes and revisions are from the Hugging Face API at the time of
#: writing; the revision is in the URL so a repo that is later edited cannot change this download.
MODELS: dict[str, ModelSpec] = {
    "qwen3.5-4b": ModelSpec(
        key="qwen3.5-4b",
        label="Qwen3.5 4B",
        license="Apache-2.0",
        asset=Asset(
            url=(
                "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/"
                "e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf"
            ),
            sha256="00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
            size=2_740_937_888,
            filename="Qwen3.5-4B-Q4_K_M.gguf",
        ),
    ),
}

#: Chosen by the benchmark in ``NOTES-PLAN.md``; change it only by re-running that.
DEFAULT_MODEL = "qwen3.5-4b"

#: Context window the server is started with. Matches ``summarizer.OLLAMA_NUM_CTX``, which the prompt
#: sizes (``SINGLE_PASS_CHARS`` and friends) are written against.
CONTEXT_TOKENS = 8192

#: How long the server may sit unused before it is stopped to give the memory back.
IDLE_STOP_SECONDS = 300


def supported() -> bool:
    """Whether this platform has a pinned runtime. Windows x64 only for now."""
    return sys.platform == "win32" and platform.machine().lower() in ("amd64", "x86_64")


def llm_dir() -> Path:
    from meeting_notes.config import models_dir  # noqa: PLC0415

    return models_dir() / "llm"


def _marker() -> Path:
    return llm_dir() / "installed.json"


def installed_info() -> Optional[dict]:
    """The install marker if the install is intact (exe and model both on disk), else ``None``."""
    try:
        info = json.loads(_marker().read_text(encoding="utf-8"))
        spec = MODELS[info["model"]]
        exe = llm_dir() / "runtime" / info["runtime"] / RUNTIME_EXE
        model = llm_dir() / "models" / spec.asset.filename
        return info if exe.is_file() and model.is_file() else None
    except (OSError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Download + verify + unpack
# ---------------------------------------------------------------------------


class Cancelled(Exception):
    pass


class LlmUnavailable(RuntimeError):
    """The model could not be reached or started — as opposed to answering badly.

    The summarizer treats the two differently: an unreachable model on the FIRST part of a long transcript
    means give up and use the next fallback at once; a bad answer for one part only degrades that part.
    """


class InstallError(Exception):
    """A failure with a sentence a person can read."""


def _download(asset: Asset, dest: Path, on_bytes: Callable[[int], None], cancel: threading.Event) -> None:
    """Fetch ``asset`` to ``dest``, resuming a partial file, then verify size and SHA-256.

    The partial file is named by the asset's hash, so bytes from a different build are never resumed into
    this one. A file that fails verification is deleted rather than kept: it has bad bytes in it somewhere.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / (asset.sha256[:16] + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > asset.size:
        part.unlink()
        have = 0

    if have < asset.size:
        request = urllib.request.Request(asset.url, headers={"User-Agent": "meeting-notes-engine/1"})
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            # timeout is per blocking read, i.e. a stall detector rather than an overall deadline: a slow
            # link may be slow, a link that has stopped may not hold the install forever.
            response = urllib.request.urlopen(request, timeout=60)  # noqa: S310 — https, pinned URLs
        except urllib.error.HTTPError as exc:
            if exc.code == 416:   # the part is not a prefix of what the server has
                part.unlink(missing_ok=True)
                raise InstallError("The partial download was stale. Try again.") from exc
            raise InstallError(f"The download server answered {exc.code}.") from exc
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            raise InstallError("Couldn't reach the download server. Check your connection and try again.") from exc

        with response:
            if have and getattr(response, "status", 200) != 206:
                have = 0   # the server ignored Range and is sending everything: start over
            done = have
            on_bytes(done)
            try:
                with open(part, "ab" if have else "wb") as out:
                    while True:
                        if cancel.is_set():
                            raise Cancelled()
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if done > asset.size:
                            raise InstallError("The server sent more than it said it would. Try again.")
                        on_bytes(done)
            except (socket.timeout, OSError) as exc:
                raise InstallError("The download was interrupted. Try again — it will resume.") from exc

    if part.stat().st_size != asset.size:
        raise InstallError("The download was incomplete. Try again.")

    digest = hashlib.sha256()
    with open(part, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            if cancel.is_set():
                raise Cancelled()
            digest.update(block)
    if digest.hexdigest() != asset.sha256:
        part.unlink(missing_ok=True)
        logger.warning("hash mismatch for %s: expected %s, got %s", asset.filename, asset.sha256, digest.hexdigest())
        raise InstallError("The download didn't match what was published, so it was thrown away. Try again.")

    dest.unlink(missing_ok=True)
    part.replace(dest)


def _unpack_runtime(zip_path: Path, dest: Path) -> None:
    """Unpack llama.cpp with zip-slip protection and a size ceiling taken from the (hash-verified) zip."""
    staging = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    root = staging.resolve()
    budget = 400 * 1024 * 1024   # the CPU build unpacks to ~65 MB; anything past this is not it
    written = 0
    with zipfile.ZipFile(zip_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            target = (staging / entry.filename).resolve()
            if root != target and root not in target.parents:
                raise InstallError("The download contained an unsafe file path, so it was rejected.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as src, open(target, "wb") as dst:
                while True:
                    block = src.read(1 << 20)
                    if not block:
                        break
                    written += len(block)
                    if written > budget:
                        raise InstallError("The download unpacked to far more than expected, so it was rejected.")
                    dst.write(block)
    if not (staging / RUNTIME_EXE).is_file():
        shutil.rmtree(staging, ignore_errors=True)
        raise InstallError("The download didn't contain the language-model runtime.")
    shutil.rmtree(dest, ignore_errors=True)
    staging.replace(dest)


class Installer:
    """Runs the install on a background thread and reports progress. One at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._state = "idle"          # idle | running | done | failed | cancelled
        self._stage = ""
        self._done = 0
        self._total = 0
        self._error: Optional[str] = None

    def status(self) -> dict:
        model = MODELS[DEFAULT_MODEL]
        info = installed_info()
        with self._lock:
            return {
                "supported": supported(),
                "installed": info is not None,
                "model": info["model"] if info else model.key,
                "model_label": MODELS[info["model"]].label if info else model.label,
                "license": model.license,
                "download_bytes": RUNTIME.size + model.asset.size,
                "state": self._state,
                "stage": self._stage,
                "bytes_done": self._done,
                "bytes_total": self._total,
                "error": self._error,
            }

    def start(self, model_key: Optional[str] = None) -> dict:
        key = model_key or DEFAULT_MODEL
        if key not in MODELS:
            raise InstallError(f"Unknown model '{key}'.")
        if not supported():
            raise InstallError("The bundled language model isn't available on this platform yet.")
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._cancel.clear()
                self._state, self._stage, self._done, self._total, self._error = "running", "starting", 0, 0, None
                self._thread = threading.Thread(target=self._run, args=(key,), name="llm-install", daemon=True)
                self._thread.start()
        return self.status()   # an install already running is simply reported, never started twice

    def cancel(self) -> None:
        self._cancel.set()

    # ------------------------------------------------------------------ internals

    def _set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, "_" + k, v)

    def _run(self, key: str) -> None:
        spec = MODELS[key]
        base = llm_dir()
        try:
            shutil.rmtree(base / "runtime" / (RUNTIME_TAG + ".tmp"), ignore_errors=True)

            runtime_dir = base / "runtime" / RUNTIME_TAG
            if not (runtime_dir / RUNTIME_EXE).is_file():
                self._set(stage="Downloading the runtime", done=0, total=RUNTIME.size)
                zip_path = base / "downloads" / RUNTIME.filename
                _download(RUNTIME, zip_path, lambda n: self._set(done=n), self._cancel)
                self._set(stage="Unpacking the runtime")
                _unpack_runtime(zip_path, runtime_dir)
                zip_path.unlink(missing_ok=True)

            model_path = base / "models" / spec.asset.filename
            if not (model_path.is_file() and model_path.stat().st_size == spec.asset.size):
                self._set(stage=f"Downloading {spec.label}", done=0, total=spec.asset.size)
                _download(spec.asset, model_path, lambda n: self._set(done=n), self._cancel)

            # Start it once and ask it a question before calling it installed: a blocked exe or a model
            # this build cannot load is found here, not when somebody presses Make notes.
            self._set(stage="Checking it works", done=0, total=0)
            check = Runtime(model_path=model_path, exe=runtime_dir / RUNTIME_EXE)
            try:
                reply = check.complete('Reply with the JSON {"ok": true} and nothing else.', max_tokens=20)
                if "ok" not in reply:
                    raise InstallError("The language model started but gave an unusable answer.")
            finally:
                check.stop()

            tmp = _marker().with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "runtime": RUNTIME_TAG, "model": key, "installed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }), encoding="utf-8")
            tmp.replace(_marker())
            self._set(state="done", stage="")
            logger.info("language model %s installed", key)
        except Cancelled:
            self._set(state="cancelled", stage="")
        except InstallError as exc:
            self._set(state="failed", error=str(exc))
        except Exception as exc:  # noqa: BLE001 — never let the thread die silently
            logger.exception("language model install failed")
            self._set(state="failed", error=f"Setting up the language model failed: {exc}")


_installer = Installer()


def get_installer() -> Installer:
    return _installer


def uninstall() -> None:
    """Remove the model and runtime (frees ~2.8 GB). Stops the server first."""
    shutdown_runtime()
    shutil.rmtree(llm_dir(), ignore_errors=True)


# ---------------------------------------------------------------------------
# The running server
# ---------------------------------------------------------------------------

_job_handle = None


def _bind_to_kill_on_close_job(pid: int) -> None:
    """Make ``pid`` die when THIS process does, however this process dies (Windows job object).

    The engine is stopped by being told to, and by Weave's pid watch, but also by Task Manager and by
    crashes; a child that outlived any of those would sit on ~3 GB for nothing. Best effort: a failure
    here costs only that guarantee.
    """
    global _job_handle
    if sys.platform != "win32":
        return
    try:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("a", "b", "c", "d", "e", "f")]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.OpenProcess.restype = wintypes.HANDLE
        if _job_handle is None:
            _job_handle = kernel32.CreateJobObjectW(None, None)
            info = EXTENDED()
            info.Basic.LimitFlags = 0x2000   # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            kernel32.SetInformationJobObject(_job_handle, 9, ctypes.byref(info), ctypes.sizeof(info))
        process = kernel32.OpenProcess(0x0100 | 0x0001, False, pid)   # SET_QUOTA | TERMINATE
        if process:
            kernel32.AssignProcessToJobObject(_job_handle, process)
            kernel32.CloseHandle(process)
    except Exception:  # noqa: BLE001
        logger.warning("could not bind the language-model server to this process", exc_info=True)


def default_threads() -> int:
    """Half the logical CPUs, 2–8. Generation is memory-bound; more threads than that stop helping and
    start fighting the person's other programs."""
    return max(2, min(8, (os.cpu_count() or 4) // 2))


class Runtime:
    """One ``llama-server`` process holding one model."""

    def __init__(self, model_path: Path, exe: Path, threads: Optional[int] = None,
                 command: Optional[list] = None) -> None:
        self.model_path = Path(model_path)
        self.exe = Path(exe)
        #: Replaces ``[exe]`` as the program to run (tests point it at a stand-in server).
        self._command = command
        self.threads = threads or default_threads()
        self._proc: Optional[subprocess.Popen] = None
        self._port = 0
        self._lock = threading.Lock()
        self._idle: Optional[threading.Timer] = None

    # -------------------------------------------------------------------- lifecycle

    def _start_locked(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self._port = s.getsockname()[1]
        flags = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW
        cmd = [*(self._command or [str(self.exe)]), "-m", str(self.model_path), "-c", str(CONTEXT_TOKENS), "-t", str(self.threads),
               "--host", "127.0.0.1", "--port", str(self._port), "--jinja", "--no-warmup"]
        self._proc = subprocess.Popen(
            cmd, cwd=str(self.exe.parent), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        _bind_to_kill_on_close_job(self._proc.pid)

        deadline = time.time() + 180   # a cold start loads ~3 GB from disk, antivirus included
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("the language-model server stopped as it started")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self._port}/health", timeout=2).close()
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        self._stop_locked()
        raise RuntimeError("the language-model server did not start in time")

    def _stop_locked(self) -> None:
        if self._idle is not None:
            self._idle.cancel()
            self._idle = None
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _arm_idle_locked(self) -> None:
        if self._idle is not None:
            self._idle.cancel()
        self._idle = threading.Timer(IDLE_STOP_SECONDS, self.stop)
        self._idle.daemon = True
        self._idle.start()

    # ------------------------------------------------------------------- generation

    def complete(self, prompt: str, max_tokens: int = 1400) -> str:
        """One chat completion, JSON-constrained: the prompt asks for JSON and the server's grammar makes
        sure that is what comes back, so a small model cannot wrap it in prose or fences."""
        with self._lock:
            try:
                self._start_locked()
            except Exception as exc:  # noqa: BLE001
                self._stop_locked()
                raise LlmUnavailable(str(exc)) from exc
            body = json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                # Reasoning models spend the budget on a hidden monologue; notes do not need it.
                "chat_template_kwargs": {"enable_thinking": False},
            }).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{self._port}/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json"},
            )
            try:
                # Generous: a few thousand tokens in and ~1000 out on a laptop CPU is minutes.
                with urllib.request.urlopen(request, timeout=900) as response:  # noqa: S310 — loopback
                    data = json.load(response)
            except Exception as exc:  # noqa: BLE001
                self._stop_locked()   # a wedged server is cheaper to restart than to diagnose
                raise LlmUnavailable(f"the language model did not answer: {exc}") from exc
            self._arm_idle_locked()
        return data["choices"][0]["message"].get("content") or ""


_runtime: Optional[Runtime] = None
_runtime_lock = threading.Lock()


def get_runtime() -> Optional[Runtime]:
    """The runtime for the installed model, or ``None`` when nothing is installed."""
    global _runtime
    info = installed_info()
    if info is None:
        return None
    with _runtime_lock:
        spec = MODELS[info["model"]]
        model = llm_dir() / "models" / spec.asset.filename
        exe = llm_dir() / "runtime" / info["runtime"] / RUNTIME_EXE
        if _runtime is None or _runtime.model_path != model or _runtime.exe != exe:
            if _runtime is not None:
                _runtime.stop()
            _runtime = Runtime(model_path=model, exe=exe)
        return _runtime


def shutdown_runtime() -> None:
    """Stop the server if one is running. Called when the engine shuts down."""
    global _runtime
    with _runtime_lock:
        runtime, _runtime = _runtime, None
    if runtime is not None:
        runtime.stop()


class BundledProvider:
    """What the summarizer calls: ``generate(prompt) -> str`` backed by the installed model."""

    def __init__(self, runtime: Runtime, model_key: str) -> None:
        self._runtime = runtime
        self.name = f"llamacpp:{model_key}"

    def generate(self, prompt: str) -> str:
        return self._runtime.complete(prompt)


def default_provider() -> Optional[BundledProvider]:
    """A provider for the installed model, or ``None`` (the caller falls back to Ollama, then to the
    extractive summary)."""
    info = installed_info()
    runtime = get_runtime()
    return BundledProvider(runtime, info["model"]) if info and runtime else None
