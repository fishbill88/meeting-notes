"""Tests for the REST server's security posture and sidecar handshake.

These guard decisions that are cheap to make now and expensive to discover
were wrong later: loopback binding, token auth on every data route, and a
shutdown path that stops orphaned sidecars.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from meeting_notes.server import (  # noqa: E402
    ALLOWED_ORIGINS,
    TOKEN_ENV,
    app,
    generate_token,
    set_shutdown_hook,
)

TOKEN = "test-token-value"

#: Every route that must never answer an unauthenticated caller.
PROTECTED_ROUTES = [
    ("get", "/backends"),
    ("post", "/shutdown"),
    ("post", "/summarize"),
    ("get", "/meetings"),
    ("post", "/meetings"),
    ("post", "/export"),
]


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    """Point the process-wide store at a temp directory for one test."""
    from meeting_notes import server as server_module

    monkeypatch.setenv("MEETING_NOTES_DATA_DIR", str(tmp_path))
    server_module.reset_store()
    yield tmp_path
    server_module.reset_store()


@pytest.fixture
def authed(monkeypatch):
    """Run the app with a known token configured."""
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def open_auth(monkeypatch):
    """Run the app with auth disabled."""
    monkeypatch.delenv(TOKEN_ENV, raising=False)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_needs_no_token(client, authed):
    # The shell polls this before it can prove anything. It must stay open.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_whether_auth_is_on(client, authed):
    assert client.get("/health").json()["auth"] is True


def test_health_reports_auth_off(client, open_auth):
    assert client.get("/health").json()["auth"] is False


# ---------------------------------------------------------------------------
# Token enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_protected_routes_reject_missing_token(client, authed, method, path):
    response = getattr(client, method)(path)
    assert response.status_code == 401


@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_protected_routes_reject_wrong_token(client, authed, method, path):
    response = getattr(client, method)(
        path, headers={"Authorization": "Bearer wrong"}
    )
    assert response.status_code == 403


def test_protected_route_rejects_non_bearer_scheme(client, authed):
    response = client.get("/backends", headers={"Authorization": f"Basic {TOKEN}"})
    assert response.status_code == 401


def test_protected_route_rejects_bare_token_without_scheme(client, authed):
    response = client.get("/backends", headers={"Authorization": TOKEN})
    assert response.status_code == 401


def test_valid_token_is_accepted(client, authed):
    assert client.get("/backends", headers=authed).status_code == 200


def test_routes_are_open_when_no_token_configured(client, open_auth):
    # `meeting-notes serve` for local development stays frictionless.
    assert client.get("/backends").status_code == 200


def test_process_route_is_protected(client, authed):
    response = client.post("/process", files={"file": ("a.wav", b"", "audio/wav")})
    assert response.status_code == 401


def test_transcribe_route_is_protected(client, authed):
    response = client.post("/transcribe", files={"file": ("a.wav", b"", "audio/wav")})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_is_not_a_wildcard():
    # The previous config was allow_origins=["*"], which is a LAN-phone
    # setting, not something that should guard patient audio.
    assert "*" not in ALLOWED_ORIGINS


def test_cors_allows_the_tauri_origin(client, open_auth):
    response = client.get("/health", headers={"Origin": "tauri://localhost"})
    assert response.headers.get("access-control-allow-origin") == "tauri://localhost"


def test_cors_rejects_an_unknown_origin(client, open_auth):
    response = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def test_shutdown_calls_the_registered_hook(client, authed):
    called = []
    set_shutdown_hook(lambda: called.append(True))
    try:
        response = client.post("/shutdown", headers=authed)
        assert response.status_code == 200
        assert called == [True]
    finally:
        set_shutdown_hook(None)


def test_shutdown_without_a_hook_reports_not_implemented(client, authed):
    set_shutdown_hook(None)
    assert client.post("/shutdown", headers=authed).status_code == 501


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_generated_tokens_are_unique_and_long():
    a, b = generate_token(), generate_token()
    assert a != b
    assert len(a) >= 32


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------


def test_sidecar_binds_loopback_only():
    from meeting_notes import sidecar

    # Not configurable on purpose.
    assert sidecar.HOST == "127.0.0.1"


def test_find_free_port_returns_a_usable_port():
    import socket

    from meeting_notes.sidecar import find_free_port

    port = find_free_port()
    assert 1024 < port < 65536

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))  # must still be free


def test_handshake_is_one_json_line(capsys):
    from meeting_notes.sidecar import emit_handshake

    emit_handshake(51823, "sekrit")
    out = capsys.readouterr().out

    assert out.count("\n") == 1
    payload = json.loads(out)
    assert payload["event"] == "ready"
    assert payload["port"] == 51823
    assert payload["token"] == "sekrit"
    assert payload["host"] == "127.0.0.1"
    assert isinstance(payload["pid"], int)


def test_sidecar_args_default_to_generated_port_and_auth():
    from meeting_notes.sidecar import parse_args

    args = parse_args([])
    assert args.port == 0        # 0 means "ask the OS"
    assert args.token is None    # means "generate one"
    assert args.no_auth is False


def test_sidecar_logs_never_go_to_stdout():
    from meeting_notes.sidecar import parse_args

    # stdout carries exactly one machine-readable line; anything else would
    # break the shell's handshake parse.
    assert parse_args([]).log_level == "warning"


def test_parent_watchdog_is_on_by_default():
    from meeting_notes.sidecar import parse_args

    # The only thing that reliably prevents orphaned sidecars.
    assert parse_args([]).no_parent_watch is False
    assert parse_args(["--no-parent-watch"]).no_parent_watch is True


def test_watch_target_defaults_to_the_immediate_parent(monkeypatch):
    import os as _os

    from meeting_notes import sidecar as sidecar_module

    monkeypatch.delenv(sidecar_module.PARENT_PID_ENV, raising=False)
    assert sidecar_module.watch_target_pid() == _os.getppid()


def test_watch_target_prefers_the_declared_pid(monkeypatch):
    # Under PyInstaller the immediate parent is a bootloader, not the shell.
    # Watching the bootloader means a hard kill of the shell orphans both.
    from meeting_notes import sidecar as sidecar_module

    monkeypatch.setenv(sidecar_module.PARENT_PID_ENV, "4242")
    assert sidecar_module.watch_target_pid() == 4242


@pytest.mark.parametrize("junk", ["", "   ", "abc", "-1", "12.5"])
def test_watch_target_ignores_a_malformed_pid(monkeypatch, junk):
    import os as _os

    from meeting_notes import sidecar as sidecar_module

    monkeypatch.setenv(sidecar_module.PARENT_PID_ENV, junk)
    assert sidecar_module.watch_target_pid() == _os.getppid()


def test_parent_watchdog_fires_when_the_parent_is_gone(monkeypatch):
    import threading as _threading

    from meeting_notes import sidecar as sidecar_module

    # Point the watchdog at a pid that cannot exist so it resolves at once.
    monkeypatch.setenv(sidecar_module.PARENT_PID_ENV, "4294967294")

    fired = _threading.Event()
    thread = sidecar_module.watch_parent(fired.set)

    assert fired.wait(timeout=10), "watchdog did not fire for a dead parent"
    thread.join(timeout=5)


def test_parent_watchdog_never_raises(monkeypatch):
    from meeting_notes import sidecar as sidecar_module

    def boom():
        raise RuntimeError("no ppid here")

    monkeypatch.delenv(sidecar_module.PARENT_PID_ENV, raising=False)
    monkeypatch.setattr(sidecar_module.os, "getppid", boom)

    # A failing watchdog must not take the server down with it.
    thread = sidecar_module.watch_parent(lambda: None)
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Stored consultations
# ---------------------------------------------------------------------------

pytest.importorskip("cryptography")

TRANSCRIPT = {
    "text": "Nag-reklamo siya ng chest pain.",
    "language": "tl",
    "language_name": "Filipino / Tagalog",
    "timed_transcript": "[00:00 – 00:04] Nag-reklamo siya ng chest pain.",
    "backend": "nemo-tagalog",
}
NOTES = {
    "title": "Chest pain, 3 days",
    "summary": "Patient reports chest pain.",
    "key_topics": ["chest pain"],
    "decisions": [],
    "action_items": ["Amoxicillin 500 mg"],
    "language": "tl",
}


def save_one(client, headers, **overrides):
    body = {
        "transcript": TRANSCRIPT,
        "notes": NOTES,
        "duration_seconds": 42.0,
        **overrides,
    }
    response = client.post("/meetings", json=body, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_save_returns_an_id(client, open_auth, store_dir):
    record_id = save_one(client, {})
    assert isinstance(record_id, str) and record_id


def test_saved_meeting_can_be_fetched(client, open_auth, store_dir):
    record_id = save_one(client, {})
    body = client.get(f"/meetings/{record_id}").json()

    assert body["notes"]["title"] == "Chest pain, 3 days"
    assert body["transcript"]["text"] == TRANSCRIPT["text"]
    assert body["duration_seconds"] == 42.0


def test_fetching_a_missing_meeting_is_404(client, open_auth, store_dir):
    assert client.get("/meetings/nope").status_code == 404


def test_list_returns_summaries(client, open_auth, store_dir):
    save_one(client, {})
    rows = client.get("/meetings").json()

    assert len(rows) == 1
    assert rows[0]["title"] == "Chest pain, 3 days"
    assert rows[0]["language"] == "tl"
    assert rows[0]["backend"] == "nemo-tagalog"
    # The list view must not ship full transcripts to render a row.
    assert "transcript" not in rows[0]


def test_list_search_filters(client, open_auth, store_dir):
    save_one(client, {})
    assert len(client.get("/meetings", params={"q": "amoxicillin"}).json()) == 1
    assert client.get("/meetings", params={"q": "metformin"}).json() == []


def test_list_respects_limit(client, open_auth, store_dir):
    for _ in range(3):
        save_one(client, {})
    assert len(client.get("/meetings", params={"limit": 2}).json()) == 2


def test_saving_with_an_id_updates_in_place(client, open_auth, store_dir):
    record_id = save_one(client, {})
    save_one(client, {}, id=record_id, notes={**NOTES, "title": "Revised"})

    assert len(client.get("/meetings").json()) == 1
    assert client.get(f"/meetings/{record_id}").json()["notes"]["title"] == "Revised"


def test_delete_removes_the_meeting(client, open_auth, store_dir):
    record_id = save_one(client, {})

    assert client.delete(f"/meetings/{record_id}").status_code == 200
    assert client.get(f"/meetings/{record_id}").status_code == 404


def test_deleting_a_missing_meeting_is_404(client, open_auth, store_dir):
    assert client.delete("/meetings/nope").status_code == 404


def test_meetings_are_encrypted_on_disk(client, open_auth, store_dir):
    from meeting_notes.storage import DB_FILENAME

    save_one(client, {})
    raw = (store_dir / DB_FILENAME).read_bytes()
    assert b"chest pain" not in raw
    assert b"Nag-reklamo" not in raw


def test_speaker_names_round_trip(client, open_auth, store_dir):
    names = {"Person 1": "Dr. Cruz", "Person 2": "Patient"}
    record_id = save_one(client, {}, speaker_names=names)

    assert client.get(f"/meetings/{record_id}").json()["speaker_names"] == names


def test_speaker_names_are_encrypted_on_disk(client, open_auth, store_dir):
    from meeting_notes.storage import DB_FILENAME

    save_one(client, {}, speaker_names={"Person 1": "Dr. Magpantay"})
    raw = (store_dir / DB_FILENAME).read_bytes()
    assert b"Magpantay" not in raw


# ---------------------------------------------------------------------------
# Speaker diarization on /transcribe
# ---------------------------------------------------------------------------


@pytest.fixture
def two_segment_backend(monkeypatch):
    from meeting_notes.backends.base import TranscriptionResult, TranscriptSegment

    class TwoSegments:
        name = "fake"

        def transcribe_file(self, path):
            return TranscriptionResult(
                text="Hello. Hi there.",
                language="en",
                language_name="English",
                segments=[
                    TranscriptSegment(start=0.0, end=1.0, text="Hello."),
                    TranscriptSegment(start=1.2, end=2.0, text="Hi there."),
                ],
                backend=self.name,
            )

    monkeypatch.setattr(
        "meeting_notes.backends.router.resolve_backend",
        lambda config: TwoSegments(),
    )


def test_transcribe_labels_speakers(
    client, open_auth, two_segment_backend, monkeypatch
):
    from meeting_notes.diarize import SpeakerSpan

    monkeypatch.setattr(
        "meeting_notes.diarize._diarize",
        lambda path: (
            [
                SpeakerSpan(start=0.0, end=1.1, speaker=0),
                SpeakerSpan(start=1.1, end=2.0, speaker=1),
            ],
            [0.0] * 32000,
            16000,
        ),
    )
    response = client.post(
        "/transcribe", files={"file": ("a.wav", b"RIFF", "audio/wav")}
    )
    assert response.status_code == 200, response.text
    segments = response.json()["segments"]
    assert [s["speaker"] for s in segments] == ["Person 1", "Person 2"]


def test_transcribe_diarize_can_be_disabled(
    client, open_auth, two_segment_backend, monkeypatch
):
    def explode(path):
        raise AssertionError("diarize=false must not diarize")

    monkeypatch.setattr("meeting_notes.diarize._diarize", explode)
    response = client.post(
        "/transcribe",
        files={"file": ("a.wav", b"RIFF", "audio/wav")},
        data={"diarize": "false"},
    )
    assert response.status_code == 200, response.text
    assert [s["speaker"] for s in response.json()["segments"]] == [None, None]


def test_transcribe_survives_unavailable_diarization(
    client, open_auth, two_segment_backend
):
    # The autouse offline_diarization fixture makes the model download fail;
    # the transcript must come back unlabelled rather than erroring.
    response = client.post(
        "/transcribe", files={"file": ("a.wav", b"RIFF", "audio/wav")}
    )
    assert response.status_code == 200, response.text
    assert [s["speaker"] for s in response.json()["segments"]] == [None, None]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_markdown(client, open_auth):
    response = client.post("/export", json={"notes": NOTES, "output_format": "markdown"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "Chest pain, 3 days" in response.text


def test_export_docx_is_a_zip(client, open_auth):
    response = client.post("/export", json={"notes": NOTES, "output_format": "docx"})

    assert response.status_code == 200
    assert response.content[:2] == b"PK"


def test_export_suggests_a_filename(client, open_auth):
    response = client.post("/export", json={"notes": NOTES, "output_format": "docx"})

    suggested = response.headers.get("x-suggested-filename")
    assert suggested and suggested.endswith(".docx")


def test_suggested_filename_is_readable_cross_origin(client, open_auth):
    # A cross-origin fetch cannot read custom headers unless the server
    # exposes them. Without this the shell silently downloads every export
    # as "notes".
    response = client.post(
        "/export",
        json={"notes": NOTES, "output_format": "docx"},
        headers={"Origin": "tauri://localhost"},
    )
    exposed = response.headers.get("access-control-expose-headers", "")
    assert "X-Suggested-Filename" in exposed


def test_delete_is_an_allowed_cors_method(client, open_auth):
    response = client.options(
        "/meetings/abc",
        headers={
            "Origin": "tauri://localhost",
            "Access-Control-Request-Method": "DELETE",
        },
    )
    assert "DELETE" in response.headers.get("access-control-allow-methods", "")


def test_export_includes_the_transcript_when_given(client, open_auth):
    response = client.post(
        "/export",
        json={
            "notes": NOTES,
            "transcript": "[00:00] Nag-reklamo siya.",
            "output_format": "markdown",
        },
    )
    assert "Nag-reklamo" in response.text


def test_export_rejects_an_unknown_format(client, open_auth):
    response = client.post("/export", json={"notes": NOTES, "output_format": "pdf"})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def test_summarize_accepts_the_consultation_template(client, open_auth):
    response = client.post(
        "/summarize",
        json={
            "text": "Patient reports chest pain. Prescribed amoxicillin.",
            "template": "consultation",
            "use_ollama": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["title"]


def test_summarize_rejects_an_unknown_template(client, open_auth):
    response = client.post(
        "/summarize",
        json={"text": "hello", "template": "nonsense", "use_ollama": False},
    )
    assert response.status_code == 400


def test_notes_response_exposes_medical_terms(client, open_auth):
    response = client.post(
        "/summarize", json={"text": "Patient has fever.", "use_ollama": False}
    )
    assert "medical_terms" in response.json()
