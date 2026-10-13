"""Background notes jobs: one recording in, one saved meeting out.

A long recording takes minutes to transcribe on a laptop CPU. ``POST /process`` holds the HTTP request
open for all of it, which is fine for the Tauri shell that sits and waits on a spinner and wrong for a
host (Weave) that wants to show progress, survive its window being closed, and not time out a request
after N minutes. So a host hands the audio to ``POST /jobs`` and gets an id back at once; this module
runs it, one job at a time, and the host polls ``GET /jobs/{id}``.

**One at a time, on purpose.** Transcription is CPU-bound and its models are large and resident:
two at once on a laptop is slower than one after the other, and doubles the memory. Jobs queue.

**In memory, on purpose.** A job's state does not survive an engine restart. What survives is the
*result* — the saved meeting, in the encrypted store. A host that finds its job gone after a restart
says so and offers to run it again; the recording is still on disk. Persisting a half-run job would
mean persisting a copy of the audio beside the store, which is the opposite of what the store is for.

**The audio copy is temporary.** The host's upload is written under the data directory's ``jobs/``
folder and deleted when the job ends, however it ends. Anything still there at start-up belongs to a
job that died with the previous process and is removed.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("meeting_notes.jobs")

#: How long a finished job's record is kept, so a host that was away can still learn the outcome.
KEEP_FINISHED_SECONDS = 3600

#: Longest error text handed to a host. Engine exceptions can carry whole tracebacks' worth of text.
MAX_ERROR_CHARS = 500

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
FINISHED_STATES = (DONE, FAILED, CANCELLED)


@dataclass
class Job:
    """One recording being turned into notes."""

    id: str
    source_name: str
    audio_path: str
    #: The folder holding the audio copy; removed when the job ends.
    work_dir: str
    config: Any
    state: str = QUEUED
    stage: str = ""
    message: str = "Waiting its turn…"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    meeting_id: Optional[str] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    #: Overall progress 0-1, or ``None`` when the work has no measure of it (a short recording is one
    #: pass). A long one reports it part by part.
    progress: Optional[float] = None
    #: ``"notes"`` (a recording in, a saved meeting out) or ``"enhance"`` (a saved meeting plus the
    #: person's notepad in, the enhanced notes saved back onto it).
    kind: str = "notes"
    #: What an ``"enhance"`` job needs: ``{"meeting_id", "notepad_key"}``. Never shown to a host.
    request: Optional[dict] = None

    @property
    def finished(self) -> bool:
        return self.state in FINISHED_STATES

    def report(self, stage: str, message: str) -> None:
        """Progress callback handed to the pipeline (``(stage, message)``)."""
        self.stage = stage
        self.message = message

    def set_progress(self, fraction: float) -> None:
        self.progress = max(0.0, min(1.0, float(fraction)))

    def public(self, queue_position: Optional[int] = None) -> dict:
        """What a host is told. Never includes the audio path or the config."""
        now = time.time()
        started = self.started_at
        end = self.finished_at or now
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "stage": self.stage,
            "message": self.message,
            "source_name": self.source_name,
            "meeting_id": self.meeting_id,
            "error": self.error,
            "queue_position": queue_position,
            "progress": round(self.progress, 3) if self.progress is not None else None,
            "elapsed_seconds": round(end - started, 1) if started else 0.0,
        }


class JobQueue:
    """Runs jobs one at a time on a single background thread.

    Args:
        execute: Runs one job and returns the saved meeting's id, or ``None`` when the job was
            cancelled while it ran (so nothing was saved). Raises to fail the job. It is handed the
            :class:`Job` so it can call :meth:`Job.report`.
    """

    def __init__(self, execute: Callable[[Job], Optional[str]]) -> None:
        self._execute = execute
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._pending: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ public

    def submit(self, job: Job) -> Job:
        with self._lock:
            self._purge_locked()
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._pending.put(job.id)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._loop, name="notes-jobs", daemon=True
                )
                self._worker.start()
        return job

    def get(self, job_id: str) -> Optional[tuple[Job, Optional[int]]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return (job, self._position_locked(job)) if job else None

    def list(self) -> list[tuple[Job, Optional[int]]]:
        """Newest first."""
        with self._lock:
            self._purge_locked()
            return [
                (self._jobs[i], self._position_locked(self._jobs[i]))
                for i in reversed(self._order)
                if i in self._jobs
            ]

    def cancel(self, job_id: str) -> Optional[Job]:
        """Cancel a job, or forget a finished one.

        A queued job simply never runs. A RUNNING job cannot be interrupted — the speech model is
        mid-way through a C++ call — so it is flagged, finishes, and its result is discarded rather
        than saved. A finished job's record is dropped. Returns the job, or ``None`` if unknown.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.finished:
                self._jobs.pop(job_id, None)
                if job_id in self._order:
                    self._order.remove(job_id)
                return job
            job.cancel_requested = True
            if job.state == QUEUED:
                self._finish_locked(job, CANCELLED)
                self._cleanup(job)
            else:
                job.message = "Stopping…"
            return job

    # ---------------------------------------------------------------- internals

    def _position_locked(self, job: Optional[Job]) -> Optional[int]:
        """1 for the next job to run, counting only those still waiting; ``None`` otherwise."""
        if job is None or job.state != QUEUED:
            return None
        waiting = [self._jobs[i] for i in self._order if i in self._jobs and self._jobs[i].state == QUEUED]
        return waiting.index(job) + 1 if job in waiting else None

    def _purge_locked(self) -> None:
        cutoff = time.time() - KEEP_FINISHED_SECONDS
        for job_id in list(self._order):
            job = self._jobs.get(job_id)
            if job is None or (job.finished and (job.finished_at or 0) < cutoff):
                self._jobs.pop(job_id, None)
                self._order.remove(job_id)

    def _finish_locked(self, job: Job, state: str) -> None:
        job.state = state
        job.finished_at = time.time()

    @staticmethod
    def _cleanup(job: Job) -> None:
        if job.work_dir:      # an enhance job has no audio copy, and rmtree of "" is not a path to rely on
            shutil.rmtree(job.work_dir, ignore_errors=True)

    def _loop(self) -> None:
        while True:
            job_id = self._pending.get()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.finished:
                    continue   # cancelled while it waited
                if job.cancel_requested:
                    self._finish_locked(job, CANCELLED)
                    self._cleanup(job)
                    continue
                job.state = RUNNING
                job.started_at = time.time()
                job.message = "Starting…"

            try:
                meeting_id = self._execute(job)
            except Exception as exc:  # noqa: BLE001 — a job failing must never kill the worker
                logger.exception("job %s failed", job.id)
                with self._lock:
                    job.error = _friendly(exc)
                    self._finish_locked(job, FAILED)
            else:
                with self._lock:
                    if meeting_id is None or job.cancel_requested:
                        self._finish_locked(job, CANCELLED)
                    else:
                        job.meeting_id = meeting_id
                        job.message = "Notes are ready." if job.kind == "notes" else "Your notes are enhanced."
                        self._finish_locked(job, DONE)
            finally:
                self._cleanup(job)


def _friendly(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text if len(text) <= MAX_ERROR_CHARS else text[: MAX_ERROR_CHARS - 1] + "…"


def sweep_stale(jobs_dir: Path) -> None:
    """Remove audio copies left by jobs that died with a previous process.

    Called once, before the first job: nothing in this process owns anything in the folder yet, and
    no job survives a restart, so everything there is an orphan holding somebody's recording.
    """
    try:
        if not jobs_dir.is_dir():
            return
        for child in jobs_dir.iterdir():
            shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
    except OSError:
        logger.warning("could not sweep stale job folders in %s", jobs_dir, exc_info=True)
