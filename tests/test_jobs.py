"""Tests for the background job queue and the /jobs routes.

The queue is exercised with a fake ``execute`` so no model is loaded; the routes with a faked
pipeline. What these guard: jobs run one at a time and in order, a failing job does not kill the
worker, a cancelled job saves nothing, the audio copy is always deleted, and every route needs the
token.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from meeting_notes import jobs as jobs_module
from meeting_notes.jobs import (
    CANCELLED,
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    Job,
    JobQueue,
    sweep_stale,
)


def _job(tmp_path: Path, name: str = "a") -> Job:
    work = tmp_path / f"job-{name}"
    work.mkdir()
    audio = work / "audio.wav"
    audio.write_bytes(b"RIFF....")
    return Job(id=name, source_name=f"{name}.webm", audio_path=str(audio), work_dir=str(work), config=None)


def _wait(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached in time")


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------


def test_a_job_runs_and_records_the_meeting_id(tmp_path):
    q = JobQueue(lambda job: "meeting-1")
    job = q.submit(_job(tmp_path))
    _wait(lambda: job.finished)
    assert job.state == DONE
    assert job.meeting_id == "meeting-1"
    assert job.error is None


def test_the_audio_copy_is_deleted_whatever_happens(tmp_path):
    def boom(job):
        raise RuntimeError("model exploded")

    ok = JobQueue(lambda job: "m")
    bad = JobQueue(boom)

    j1 = ok.submit(_job(tmp_path, "ok"))
    j2 = bad.submit(_job(tmp_path, "bad"))
    _wait(lambda: j1.finished and j2.finished)

    assert not Path(j1.work_dir).exists()
    assert not Path(j2.work_dir).exists()


def test_a_failing_job_reports_its_error_and_the_worker_carries_on(tmp_path):
    calls = []

    def execute(job):
        calls.append(job.id)
        if job.id == "first":
            raise RuntimeError("could not decode this file")
        return "m2"

    q = JobQueue(execute)
    first = q.submit(_job(tmp_path, "first"))
    second = q.submit(_job(tmp_path, "second"))
    _wait(lambda: first.finished and second.finished)

    assert first.state == FAILED
    assert "could not decode" in first.error
    assert second.state == DONE            # the worker survived
    assert calls == ["first", "second"]    # and ran them in order


def test_a_long_error_is_truncated(tmp_path):
    def boom(job):
        raise RuntimeError("x" * 5000)

    q = JobQueue(boom)
    job = q.submit(_job(tmp_path))
    _wait(lambda: job.finished)
    assert len(job.error) <= jobs_module.MAX_ERROR_CHARS


def test_jobs_run_one_at_a_time(tmp_path):
    gate = threading.Event()
    running_at_once = []
    active = []

    def execute(job):
        active.append(job.id)
        running_at_once.append(len(active))
        gate.wait(2)
        active.remove(job.id)
        return "m"

    q = JobQueue(execute)
    jobs = [q.submit(_job(tmp_path, n)) for n in ("a", "b", "c")]
    _wait(lambda: jobs[0].state == RUNNING)
    gate.set()
    _wait(lambda: all(j.finished for j in jobs))
    assert max(running_at_once) == 1


def test_queue_position_counts_only_waiting_jobs(tmp_path):
    gate = threading.Event()
    q = JobQueue(lambda job: gate.wait(3) and "m")
    a = q.submit(_job(tmp_path, "a"))
    _wait(lambda: a.state == RUNNING)
    b = q.submit(_job(tmp_path, "b"))
    c = q.submit(_job(tmp_path, "c"))

    assert q.get("a")[1] is None      # running, not waiting
    assert q.get("b")[1] == 1
    assert q.get("c")[1] == 2
    gate.set()
    _wait(lambda: c.finished)


def test_cancelling_a_queued_job_means_it_never_runs(tmp_path):
    gate = threading.Event()
    ran = []

    def execute(job):
        ran.append(job.id)
        gate.wait(3)
        return "m"

    q = JobQueue(execute)
    a = q.submit(_job(tmp_path, "a"))
    _wait(lambda: a.state == RUNNING)
    b = q.submit(_job(tmp_path, "b"))

    q.cancel("b")
    assert b.state == CANCELLED
    assert not Path(b.work_dir).exists()
    gate.set()
    _wait(lambda: a.finished)
    time.sleep(0.1)
    assert ran == ["a"]


def test_cancelling_a_running_job_discards_the_result(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def execute(job):
        started.set()
        release.wait(3)
        # a real execute checks job.cancel_requested before saving, and returns None
        return None if job.cancel_requested else "saved"

    q = JobQueue(execute)
    job = q.submit(_job(tmp_path))
    started.wait(2)
    q.cancel(job.id)
    assert job.state == RUNNING and job.cancel_requested   # cannot be interrupted, only flagged
    release.set()
    _wait(lambda: job.finished)
    assert job.state == CANCELLED
    assert job.meeting_id is None


def test_cancelling_a_finished_job_forgets_it(tmp_path):
    q = JobQueue(lambda job: "m")
    job = q.submit(_job(tmp_path))
    _wait(lambda: job.finished)
    q.cancel(job.id)
    assert q.get(job.id) is None


def test_cancelling_an_unknown_job_is_none(tmp_path):
    assert JobQueue(lambda job: "m").cancel("nope") is None


def test_finished_jobs_are_forgotten_after_an_hour(tmp_path, monkeypatch):
    q = JobQueue(lambda job: "m")
    old = q.submit(_job(tmp_path, "old"))
    _wait(lambda: old.finished)
    old.finished_at = time.time() - jobs_module.KEEP_FINISHED_SECONDS - 10

    q.submit(_job(tmp_path, "new"))     # a submit purges
    assert q.get("old") is None
    assert q.get("new") is not None


def test_public_view_leaks_neither_the_audio_path_nor_the_config(tmp_path):
    job = _job(tmp_path)
    shown = job.public()
    assert "audio_path" not in shown and "work_dir" not in shown and "config" not in shown
    assert set(shown) >= {"id", "state", "stage", "message", "meeting_id", "error", "elapsed_seconds"}


def test_progress_reported_by_the_pipeline_is_visible(tmp_path):
    seen = {}
    release = threading.Event()

    def execute(job):
        job.report("diarize", "Identifying speakers…")
        seen["stage"] = job.stage
        release.wait(2)
        return "m"

    q = JobQueue(execute)
    job = q.submit(_job(tmp_path))
    _wait(lambda: job.stage == "diarize")
    assert q.get(job.id)[0].public()["message"] == "Identifying speakers…"
    release.set()
    _wait(lambda: job.finished)


# ---------------------------------------------------------------------------
# Start-up sweep
# ---------------------------------------------------------------------------


def test_sweep_removes_audio_left_by_a_previous_process(tmp_path):
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "job-old").mkdir(parents=True)
    (jobs_dir / "job-old" / "audio.webm").write_bytes(b"somebody's recording")
    (jobs_dir / "stray.tmp").write_bytes(b"x")

    sweep_stale(jobs_dir)
    assert list(jobs_dir.iterdir()) == []


def test_sweep_of_a_missing_folder_is_fine(tmp_path):
    sweep_stale(tmp_path / "nothing-here")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from meeting_notes import server as server_module  # noqa: E402
from meeting_notes.backends.base import TranscriptionResult, TranscriptSegment  # noqa: E402
from meeting_notes.pipeline import PipelineResult  # noqa: E402
from meeting_notes.server import TOKEN_ENV, app  # noqa: E402
from meeting_notes.summarizer import MeetingNotes  # noqa: E402

TOKEN = "jobs-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    monkeypatch.setenv("MEETING_NOTES_DATA_DIR", str(tmp_path))
    server_module.reset_store()
    server_module.reset_jobs()
    yield tmp_path
    server_module.reset_store()
    server_module.reset_jobs()


@pytest.fixture
def client(env):
    return TestClient(app)


def _fake_result() -> PipelineResult:
    segs = [
        TranscriptSegment(start=0.0, end=3.0, text="Good morning everyone.", lang="en", speaker="Person 1"),
        TranscriptSegment(start=3.0, end=7.5, text="We ship on Friday.", lang="en", speaker="Person 2"),
    ]
    transcript = TranscriptionResult(
        text="Good morning everyone. We ship on Friday.",
        language="en",
        language_name="English",
        segments=segs,
        backend="faster-whisper",
    )
    notes = MeetingNotes(
        title="Release sync",
        summary="Agreed to ship on Friday.",
        decisions=["Ship on Friday"],
        action_items=["Ana sends the build"],
        generator="extractive",
    )
    return PipelineResult(transcript=transcript, notes=notes)


def _post_job(client, **form):
    return client.post(
        "/jobs",
        headers=AUTH,
        files={"file": ("Sprint sync - 2026-10-12 08.30.webm", b"not really audio", "audio/webm")},
        data=form,
    )


def _wait_state(client, job_id, states, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}", headers=AUTH).json()
        if body["state"] in states:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job stayed {body['state']}")


@pytest.mark.parametrize(
    "method,path",
    [("post", "/jobs"), ("get", "/jobs"), ("get", "/jobs/x"), ("delete", "/jobs/x")],
)
def test_every_job_route_needs_the_token(client, method, path):
    kwargs = {"files": {"file": ("a.wav", b"", "audio/wav")}} if method == "post" else {}
    assert getattr(client, method)(path, **kwargs).status_code == 401


def test_a_job_runs_end_to_end_and_the_meeting_is_saved(client, monkeypatch):
    seen = {}

    def fake_pipeline(path, config, on_progress=None, **kw):
        seen["audio_existed"] = Path(path).is_file()
        seen["suffix"] = Path(path).suffix
        on_progress("transcribe", "Transcribing…")
        return _fake_result()

    monkeypatch.setattr(server_module, "run_pipeline", fake_pipeline)

    created = _post_job(client)
    assert created.status_code == 200
    job = created.json()
    assert job["state"] in ("queued", "running", "done")
    assert job["source_name"].startswith("Sprint sync")

    done = _wait_state(client, job["id"], {"done", "failed"})
    assert done["state"] == "done", done
    assert seen == {"audio_existed": True, "suffix": ".webm"}   # real file, extension kept for the decoder

    meeting = client.get(f"/meetings/{done['meeting_id']}", headers=AUTH).json()
    assert meeting["notes"]["title"] == "Release sync"
    assert meeting["notes"]["generator"] == "extractive"
    assert meeting["transcript"]["segments"][1]["speaker"] == "Person 2"
    assert meeting["duration_seconds"] == 7.5


def test_the_audio_copy_is_gone_when_the_job_ends(client, env, monkeypatch):
    monkeypatch.setattr(server_module, "run_pipeline", lambda p, c, on_progress=None, **kw: _fake_result())
    job = _post_job(client).json()
    _wait_state(client, job["id"], {"done"})
    time.sleep(0.1)
    leftovers = list((env / "jobs").glob("*")) if (env / "jobs").exists() else []
    assert leftovers == []


def test_a_pipeline_failure_becomes_a_failed_job_with_a_sentence(client, monkeypatch):
    def boom(path, config, on_progress=None, **kw):
        raise RuntimeError("the audio could not be decoded")

    monkeypatch.setattr(server_module, "run_pipeline", boom)
    job = _post_job(client).json()
    done = _wait_state(client, job["id"], {"failed"})
    assert "could not be decoded" in done["error"]
    assert done["meeting_id"] is None


def test_bad_settings_are_refused_up_front(client):
    response = _post_job(client, template="not-a-template")
    assert response.status_code == 400
    assert client.get("/jobs", headers=AUTH).json()["jobs"] == []   # nothing was queued


def test_unknown_job_is_404(client):
    assert client.get("/jobs/nope", headers=AUTH).status_code == 404
    assert client.delete("/jobs/nope", headers=AUTH).status_code == 404


def test_list_shows_newest_first(client, monkeypatch):
    monkeypatch.setattr(server_module, "run_pipeline", lambda p, c, on_progress=None, **kw: _fake_result())
    first = _post_job(client).json()
    second = _post_job(client).json()
    _wait_state(client, second["id"], {"done"})
    listed = client.get("/jobs", headers=AUTH).json()["jobs"]
    assert [j["id"] for j in listed][:2] == [second["id"], first["id"]]


def test_cancelling_a_running_job_saves_nothing(client, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow(path, config, on_progress=None, **kw):
        started.set()
        release.wait(3)
        return _fake_result()

    monkeypatch.setattr(server_module, "run_pipeline", slow)
    job = _post_job(client).json()
    started.wait(2)

    client.delete(f"/jobs/{job['id']}", headers=AUTH)
    release.set()
    done = _wait_state(client, job["id"], {"cancelled", "done"})
    assert done["state"] == "cancelled"
    assert client.get("/meetings", headers=AUTH).json() == []   # nothing was saved


def test_building_the_queue_sweeps_audio_from_a_dead_process(env):
    orphan = env / "jobs" / "job-dead"
    orphan.mkdir(parents=True)
    (orphan / "audio.webm").write_bytes(b"left behind")
    server_module.get_jobs()
    assert not orphan.exists()
