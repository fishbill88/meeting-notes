"""Tests for the bundled language model: catalog, verified download, unpacking, the server process,
the summarizer's use of it, and the routes.

No real model or llama.cpp is touched. Downloads come from a local HTTP server that supports ``Range``;
the "llama-server" is ``fake_llama.py``. What these guard: nothing unverified is ever installed, a
partial download resumes, an unreachable model falls through to the next fallback instead of grinding
on, the server dies with its handle, and nothing downloads unless a host asks.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import re
import socketserver
import sys
import threading
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meeting_notes import llm
from meeting_notes.llm import (
    Asset,
    Cancelled,
    InstallError,
    Installer,
    LlmUnavailable,
    Runtime,
    _download,
    _unpack_runtime,
)

FAKE = Path(__file__).with_name("fake_llama.py")


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


def test_every_pinned_asset_is_fully_specified():
    assets = [llm.RUNTIME] + [m.asset for m in llm.MODELS.values()]
    for asset in assets:
        assert asset.url.startswith("https://"), asset.url
        assert re.fullmatch(r"[0-9a-f]{64}", asset.sha256), asset.filename
        assert asset.size > 0
        assert asset.filename.endswith((".zip", ".gguf"))


def test_a_model_download_is_pinned_to_a_revision_not_main():
    """A repo that is edited after we chose the model must not change what people download."""
    for spec in llm.MODELS.values():
        assert re.search(r"/resolve/[0-9a-f]{40}/", spec.asset.url), spec.key
        assert "/main/" not in spec.asset.url


def test_the_runtime_is_a_pinned_build_not_latest():
    assert llm.RUNTIME_TAG in llm.RUNTIME.url
    assert "latest" not in llm.RUNTIME.url


def test_the_default_model_exists_and_is_openly_licensed():
    assert llm.DEFAULT_MODEL in llm.MODELS
    assert llm.MODELS[llm.DEFAULT_MODEL].license in ("Apache-2.0", "MIT")


# ---------------------------------------------------------------------------
# Install state on disk
# ---------------------------------------------------------------------------


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MEETING_NOTES_MODELS_DIR", str(tmp_path))
    return tmp_path


def _fake_install(models_dir: Path, key: str = None):
    key = key or llm.DEFAULT_MODEL
    spec = llm.MODELS[key]
    base = models_dir / "llm"
    (base / "runtime" / llm.RUNTIME_TAG).mkdir(parents=True)
    (base / "runtime" / llm.RUNTIME_TAG / llm.RUNTIME_EXE).write_bytes(b"x")
    (base / "models").mkdir(parents=True)
    (base / "models" / spec.asset.filename).write_bytes(b"y")
    (base / "installed.json").write_text(json.dumps({"runtime": llm.RUNTIME_TAG, "model": key}))


def test_nothing_installed_is_none(models_dir):
    assert llm.installed_info() is None
    assert llm.default_provider() is None


def test_a_marker_without_its_files_is_not_installed(models_dir):
    _fake_install(models_dir)
    (models_dir / "llm" / "models" / llm.MODELS[llm.DEFAULT_MODEL].asset.filename).unlink()
    assert llm.installed_info() is None


def test_an_intact_install_is_found(models_dir):
    _fake_install(models_dir)
    info = llm.installed_info()
    assert info and info["model"] == llm.DEFAULT_MODEL
    provider = llm.default_provider()
    assert provider is not None and provider.name == f"llamacpp:{llm.DEFAULT_MODEL}"


def test_a_corrupt_marker_is_not_installed(models_dir):
    (models_dir / "llm").mkdir()
    (models_dir / "llm" / "installed.json").write_text("{not json")
    assert llm.installed_info() is None


# ---------------------------------------------------------------------------
# A download server with Range support
# ---------------------------------------------------------------------------


class _Server:
    def __init__(self, payloads: dict[str, bytes], ignore_range: bool = False):
        outer = self
        self.requests: list[tuple[str, str]] = []
        self.ignore_range = ignore_range

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                data = payloads.get(self.path.lstrip("/"))
                outer.requests.append((self.path, self.headers.get("Range", "")))
                if data is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                rng = self.headers.get("Range", "")
                start = 0
                if rng.startswith("bytes=") and not outer.ignore_range:
                    start = int(rng[6:].split("-")[0])
                if start >= len(data):
                    self.send_response(416)
                    self.end_headers()
                    return
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(data) - start))
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.end_headers()
                self.wfile.write(data[start:])

        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.port}/{name}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _asset(server: _Server, name: str, data: bytes, **over) -> Asset:
    kw = dict(url=server.url(name), sha256=hashlib.sha256(data).hexdigest(), size=len(data), filename=name)
    kw.update(over)
    return Asset(**kw)


@pytest.fixture
def served():
    data = bytes(range(256)) * 4000   # ~1 MB, not all one value
    server = _Server({"model.bin": data})
    yield server, data
    server.close()


def test_a_download_is_verified_and_lands_where_asked(served, tmp_path):
    server, data = served
    dest = tmp_path / "out" / "model.bin"
    seen = []
    _download(_asset(server, "model.bin", data), dest, seen.append, threading.Event())
    assert dest.read_bytes() == data
    assert seen and seen[-1] == len(data)
    assert not list(dest.parent.glob("*.part"))


def test_a_download_that_does_not_match_its_hash_is_thrown_away(served, tmp_path):
    server, data = served
    wrong = _asset(server, "model.bin", data, sha256="0" * 64)
    dest = tmp_path / "model.bin"
    with pytest.raises(InstallError, match="didn't match"):
        _download(wrong, dest, lambda n: None, threading.Event())
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))   # bad bytes are not kept for resuming


def test_a_server_that_sends_more_than_promised_is_refused(served, tmp_path):
    server, data = served
    short = _asset(server, "model.bin", data, size=len(data) - 100)
    with pytest.raises(InstallError, match="more than"):
        _download(short, tmp_path / "m.bin", lambda n: None, threading.Event())


def test_a_partial_download_resumes_with_a_range_request(served, tmp_path):
    server, data = served
    asset = _asset(server, "model.bin", data)
    dest = tmp_path / "model.bin"
    (tmp_path / (asset.sha256[:16] + ".part")).write_bytes(data[:300_000])

    _download(asset, dest, lambda n: None, threading.Event())
    assert dest.read_bytes() == data
    assert server.requests[-1][1] == "bytes=300000-"   # asked for only the rest


def test_a_server_that_ignores_range_restarts_cleanly(tmp_path):
    data = bytes(range(256)) * 2000
    server = _Server({"model.bin": data}, ignore_range=True)
    try:
        asset = _asset(server, "model.bin", data)
        (tmp_path / (asset.sha256[:16] + ".part")).write_bytes(data[:1000])
        _download(asset, tmp_path / "model.bin", lambda n: None, threading.Event())
        assert (tmp_path / "model.bin").read_bytes() == data   # not 1000 bytes of the start twice
    finally:
        server.close()


def test_cancelling_stops_the_download_and_keeps_the_partial(served, tmp_path):
    server, data = served
    asset = _asset(server, "model.bin", data)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        _download(asset, tmp_path / "model.bin", lambda n: None, cancel)
    assert not (tmp_path / "model.bin").exists()


def test_a_missing_file_is_a_sentence_not_a_traceback(served, tmp_path):
    server, data = served
    with pytest.raises(InstallError, match="404"):
        _download(_asset(server, "nope.bin", data), tmp_path / "x", lambda n: None, threading.Event())


def test_an_unreachable_server_is_a_sentence(tmp_path):
    asset = Asset(url="http://127.0.0.1:9/x", sha256="0" * 64, size=1, filename="x")
    with pytest.raises(InstallError, match="reach"):
        _download(asset, tmp_path / "x", lambda n: None, threading.Event())


# ---------------------------------------------------------------------------
# Unpacking the runtime
# ---------------------------------------------------------------------------


def _zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return path


def test_unpacking_a_good_runtime(tmp_path):
    z = _zip(tmp_path / "rt.zip", {llm.RUNTIME_EXE: b"MZ", "ggml.dll": b"d"})
    _unpack_runtime(z, tmp_path / "runtime" / "b1")
    assert (tmp_path / "runtime" / "b1" / llm.RUNTIME_EXE).is_file()
    assert not (tmp_path / "runtime" / "b1.tmp").exists()


def test_a_zip_that_climbs_out_of_its_folder_is_rejected(tmp_path):
    z = _zip(tmp_path / "rt.zip", {llm.RUNTIME_EXE: b"MZ", "../evil.txt": b"pwned"})
    with pytest.raises(InstallError, match="unsafe"):
        _unpack_runtime(z, tmp_path / "runtime" / "b1")
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "runtime" / "evil.txt").exists()


def test_a_zip_without_the_server_is_rejected(tmp_path):
    z = _zip(tmp_path / "rt.zip", {"readme.txt": b"hi"})
    with pytest.raises(InstallError, match="runtime"):
        _unpack_runtime(z, tmp_path / "runtime" / "b1")
    assert not (tmp_path / "runtime" / "b1").exists()


# ---------------------------------------------------------------------------
# The server process
# ---------------------------------------------------------------------------


def _runtime(tmp_path: Path, **env) -> Runtime:
    return Runtime(model_path=tmp_path / "m.gguf", exe=tmp_path / "x.exe", threads=2,
                   command=[sys.executable, str(FAKE)])


def test_complete_starts_the_server_and_asks_for_json_without_thinking(tmp_path):
    rt = _runtime(tmp_path)
    try:
        assert json.loads(rt.complete("hello")) == {"ok": True}
        assert rt._proc is not None and rt._proc.poll() is None
    finally:
        rt.stop()


def test_the_request_asks_for_json_mode_and_no_reasoning(tmp_path, monkeypatch):
    seen = {}
    real = llm.urllib.request.urlopen

    def spy(request, *a, **kw):
        if isinstance(request, llm.urllib.request.Request) and request.data:
            seen.update(json.loads(request.data))
        return real(request, *a, **kw)

    monkeypatch.setattr(llm.urllib.request, "urlopen", spy)
    rt = _runtime(tmp_path)
    try:
        rt.complete("hello")
    finally:
        rt.stop()
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["messages"][0]["content"] == "hello"


def test_the_server_is_reused_between_calls(tmp_path):
    rt = _runtime(tmp_path)
    try:
        rt.complete("one")
        first = rt._proc.pid
        rt.complete("two")
        assert rt._proc.pid == first
    finally:
        rt.stop()


def test_stop_ends_the_process(tmp_path):
    rt = _runtime(tmp_path)
    rt.complete("go")
    proc = rt._proc
    rt.stop()
    assert proc.poll() is not None
    assert rt._proc is None


def test_an_idle_server_stops_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "IDLE_STOP_SECONDS", 0.3)
    rt = _runtime(tmp_path)
    try:
        rt.complete("go")
        proc = rt._proc
        deadline = time.time() + 5
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        assert proc.poll() is not None
    finally:
        rt.stop()


def test_a_server_that_will_not_start_is_unavailable_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "exit")
    rt = _runtime(tmp_path)
    with pytest.raises(LlmUnavailable):
        rt.complete("go")
    assert rt._proc is None


def test_a_server_that_stops_answering_is_unavailable_and_restarted_next_time(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "hang")
    monkeypatch.setattr(llm.urllib.request, "urlopen", _timeout_on_completions(llm.urllib.request.urlopen))
    rt = _runtime(tmp_path)
    try:
        with pytest.raises(LlmUnavailable):
            rt.complete("go")
        assert rt._proc is None   # the wedged server was stopped rather than kept
    finally:
        rt.stop()


def _timeout_on_completions(real):
    """The engine waits up to 15 minutes for a completion; a test cannot. Shorten that one call."""
    def wrapper(request, *a, **kw):
        if isinstance(request, llm.urllib.request.Request) and request.data:
            kw["timeout"] = 1
        return real(request, *a, **kw)
    return wrapper


def test_default_threads_are_bounded():
    assert 2 <= llm.default_threads() <= 8


# ---------------------------------------------------------------------------
# The installer, end to end (fake downloads, fake server)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_catalog(models_dir, monkeypatch):
    """Point the catalog at a local server and the runtime at the fake llama-server."""
    runtime_zip = io.BytesIO()
    with zipfile.ZipFile(runtime_zip, "w") as z:
        z.writestr(llm.RUNTIME_EXE, b"MZ-not-really")
    runtime_bytes = runtime_zip.getvalue()
    model_bytes = b"GGUF" + bytes(range(256)) * 500

    server = _Server({"rt.zip": runtime_bytes, "m.gguf": model_bytes})
    monkeypatch.setattr(llm, "RUNTIME", Asset(server.url("rt.zip"), hashlib.sha256(runtime_bytes).hexdigest(),
                                              len(runtime_bytes), "rt.zip"))
    spec = llm.ModelSpec("test-model", "Test model", Asset(server.url("m.gguf"), hashlib.sha256(model_bytes).hexdigest(),
                                                           len(model_bytes), "m.gguf"), "Apache-2.0")
    monkeypatch.setattr(llm, "MODELS", {"test-model": spec})
    monkeypatch.setattr(llm, "DEFAULT_MODEL", "test-model")
    monkeypatch.setattr(llm, "supported", lambda: True)

    real_runtime = llm.Runtime
    monkeypatch.setattr(llm, "Runtime", lambda model_path, exe, threads=None: real_runtime(
        model_path, exe, threads, command=[sys.executable, str(FAKE)]))
    yield server
    server.close()


def _wait(installer: Installer, states=("done", "failed", "cancelled"), timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = installer.status()
        if s["state"] in states:
            return s
        time.sleep(0.05)
    raise AssertionError(f"install stayed {installer.status()['state']}")


def test_a_full_install_downloads_verifies_checks_it_runs_and_marks_installed(fake_catalog, models_dir):
    inst = Installer()
    assert inst.status()["installed"] is False
    inst.start()
    status = _wait(inst)
    assert status["state"] == "done", status
    assert status["installed"] is True
    assert llm.installed_info()["model"] == "test-model"
    assert (models_dir / "llm" / "models" / "m.gguf").is_file()
    assert (models_dir / "llm" / "runtime" / llm.RUNTIME_TAG / llm.RUNTIME_EXE).is_file()
    assert not list((models_dir / "llm").rglob("*.part"))


def test_the_marker_is_written_only_after_the_start_up_check_passes(fake_catalog, models_dir, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "exit")   # downloads fine, but the server will not run
    inst = Installer()
    inst.start()
    status = _wait(inst)
    assert status["state"] == "failed"
    assert status["installed"] is False
    assert not (models_dir / "llm" / "installed.json").exists()


def test_a_corrupt_model_download_fails_and_installs_nothing(fake_catalog, models_dir, monkeypatch):
    bad = llm.ModelSpec("test-model", "Test model", Asset(llm.MODELS["test-model"].asset.url, "0" * 64,
                                                         llm.MODELS["test-model"].asset.size, "m.gguf"), "Apache-2.0")
    monkeypatch.setattr(llm, "MODELS", {"test-model": bad})
    inst = Installer()
    inst.start()
    status = _wait(inst)
    assert status["state"] == "failed" and "didn't match" in status["error"]
    assert llm.installed_info() is None


def test_starting_twice_does_not_start_two_installs(fake_catalog, models_dir):
    inst = Installer()
    inst.start()
    inst.start()
    _wait(inst)
    assert fake_catalog.requests.count(("/m.gguf", "")) == 1


def test_an_unsupported_platform_is_refused_up_front(models_dir, monkeypatch):
    monkeypatch.setattr(llm, "supported", lambda: False)
    with pytest.raises(InstallError, match="platform"):
        Installer().start()


def test_an_unknown_model_is_refused(models_dir):
    with pytest.raises(InstallError, match="Unknown"):
        Installer().start("gpt-17")


def test_uninstall_removes_everything(fake_catalog, models_dir):
    inst = Installer()
    inst.start()
    _wait(inst)
    llm.uninstall()
    assert not (models_dir / "llm").exists()
    assert llm.installed_info() is None


# ---------------------------------------------------------------------------
# The summarizer's use of it
# ---------------------------------------------------------------------------

from meeting_notes.summarizer import Summarizer  # noqa: E402

GOOD_JSON = json.dumps({
    "title": "Bundled notes", "summary": "Written by the bundled model.",
    "key_topics": ["a"], "decisions": ["b"], "action_items": ["c"],
})
TRANSCRIPT = "We agreed to ship on Friday. Ana will send the build. " * 5


class _Provider:
    name = "llamacpp:test-model"

    def __init__(self, reply=GOOD_JSON, exc=None):
        self.reply, self.exc, self.calls = reply, exc, 0

    def generate(self, prompt):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.reply


@patch("requests.post")
def test_the_bundled_model_is_used_first_and_ollama_is_not_asked(mock_post):
    provider = _Provider()
    notes = Summarizer(use_ollama=True, provider=provider).summarize(TRANSCRIPT)
    assert notes.title == "Bundled notes"
    assert notes.generator == "llamacpp:test-model"
    assert provider.calls == 1
    mock_post.assert_not_called()


@patch("requests.post")
def test_an_unreachable_bundled_model_falls_back_to_ollama(mock_post):
    reply = MagicMock()
    reply.json.return_value = {"response": GOOD_JSON.replace("Bundled", "Ollama")}
    reply.raise_for_status = MagicMock()
    mock_post.return_value = reply
    summarizer = Summarizer(use_ollama=True, provider=_Provider(exc=LlmUnavailable("won't start")))
    with patch.object(Summarizer, "resolve_model", lambda self: "llama3"):
        notes = summarizer.summarize(TRANSCRIPT)
    assert notes.generator == "ollama:llama3"
    assert notes.title == "Ollama notes"


@patch("requests.post")
def test_with_nothing_reachable_the_notes_are_extractive_and_say_so(mock_post):
    mock_post.side_effect = Exception("refused")
    summarizer = Summarizer(use_ollama=True, provider=_Provider(exc=LlmUnavailable("won't start")))
    with patch.object(Summarizer, "resolve_model", lambda self: "llama3"):
        notes = summarizer.summarize(TRANSCRIPT)
    assert notes.generator == "extractive"


def test_ollama_disabled_still_tries_the_bundled_model_then_extractive():
    provider = _Provider(exc=LlmUnavailable("won't start"))
    notes = Summarizer(use_ollama=False, provider=provider).summarize(TRANSCRIPT)
    assert provider.calls == 1
    assert notes.generator == "extractive"


def test_use_ollama_false_with_no_provider_never_reaches_for_any_model():
    """What the pipeline passes when the caller asked for no model at all."""
    notes = Summarizer(use_ollama=False).summarize(TRANSCRIPT)
    assert notes.generator == "extractive"


def test_a_long_transcript_gives_up_on_an_unreachable_model_at_the_first_part():
    long_text = ("Sentence number %d is about the release plan and who owns what. " * 1) * 1
    long_text = " ".join(long_text % i for i in range(400))
    provider = _Provider(exc=LlmUnavailable("won't start"))
    notes = Summarizer(use_ollama=False, provider=provider).summarize(long_text)
    assert provider.calls == 1            # did not retry a dead model for every part
    assert notes.generator == "extractive"


def test_a_long_transcript_uses_the_bundled_model_for_every_part_and_the_merge():
    long_text = " ".join("Sentence number %d is about the release plan." % i for i in range(1000))
    provider = _Provider()
    notes = Summarizer(use_ollama=False, provider=provider).summarize(long_text)
    assert provider.calls > 2             # several parts, then a merge
    assert notes.generator == "llamacpp:test-model"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from meeting_notes import server as server_module  # noqa: E402
from meeting_notes.server import TOKEN_ENV, app  # noqa: E402

TOKEN = "llm-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(monkeypatch, models_dir):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    monkeypatch.setattr(llm, "_installer", Installer())
    return TestClient(app)


@pytest.mark.parametrize("method,path", [("get", "/llm"), ("post", "/llm/install"),
                                         ("delete", "/llm/install"), ("delete", "/llm")])
def test_every_llm_route_needs_the_token(client, method, path):
    assert getattr(client, method)(path).status_code == 401


def test_status_says_what_is_and_is_not_there(client):
    body = client.get("/llm", headers=AUTH).json()
    assert body["installed"] is False
    assert body["state"] == "idle"
    assert body["download_bytes"] == llm.RUNTIME.size + llm.MODELS[llm.DEFAULT_MODEL].asset.size
    assert body["license"]


def test_nothing_downloads_unless_a_host_asks(client, monkeypatch):
    started = []
    monkeypatch.setattr(llm.Installer, "_run", lambda self, key: started.append(key))
    client.get("/llm", headers=AUTH)
    client.get("/health")
    assert started == []


def test_install_on_an_unsupported_platform_is_a_400_with_a_sentence(client, monkeypatch):
    monkeypatch.setattr(llm, "supported", lambda: False)
    response = client.post("/llm/install", headers=AUTH)
    assert response.status_code == 400
    assert "platform" in response.json()["detail"]


def test_install_takes_the_model_as_a_query_parameter_and_tolerates_an_empty_form(client, monkeypatch):
    seen = []
    monkeypatch.setattr(llm._installer, "start", lambda model=None: seen.append(model) or {"state": "running"})
    assert client.post("/llm/install", headers=AUTH).status_code == 200                       # no body at all
    assert client.post("/llm/install?model=qwen3.5-4b", headers=AUTH).status_code == 200      # named
    assert client.post("/llm/install", headers=AUTH, files={}, data={}).status_code in (200, 400, 415, 422)
    assert seen[:2] == [None, "qwen3.5-4b"]


def test_remove_is_refused_while_an_install_runs(client, monkeypatch):
    monkeypatch.setattr(llm._installer, "status", lambda: {"state": "running"})
    assert client.delete("/llm", headers=AUTH).status_code == 409
