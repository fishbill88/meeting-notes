"""FastAPI server — exposes transcription and summarization as REST endpoints.

Start with:
    meeting-notes serve
    # or as the desktop sidecar (loopback, random port, bearer token):
    python -m meeting_notes.sidecar

**This process handles patient audio.** Three things follow, and none of them
are optional:

* It binds loopback only. There is no configuration in which a clinic
  workstation should serve transcription to its LAN.
* Every route except ``/health`` requires a bearer token when
  ``MEETING_NOTES_TOKEN`` is set. The sidecar always sets it.
* CORS is restricted to the desktop shell's own origins, not ``*``.

``/health`` stays open deliberately: the desktop shell polls it before it has
anything else to say, and it leaks nothing but liveness.
"""

from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path
from typing import Callable, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from meeting_notes.config import AppConfig, DEFAULT_WHISPER_MODEL, LanguageMode
from meeting_notes.pipeline import run_pipeline
from meeting_notes.summarizer import Summarizer

#: Environment variable holding the shared bearer token. Unset means no auth,
#: which is only appropriate for local development.
TOKEN_ENV = "MEETING_NOTES_TOKEN"

#: Optional passphrase for the consultation store. Unset means key-file mode —
#: see the threat model in ``meeting_notes.crypto``.
PASSPHRASE_ENV = "MEETING_NOTES_PASSPHRASE"

#: Origins the desktop shell can present. Tauri serves the app from a custom
#: scheme in release builds and from the Vite dev server in development.
ALLOWED_ORIGINS = (
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
)

app = FastAPI(title="Meeting Notes AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    # Custom response headers are invisible to a cross-origin fetch unless
    # named here. The shell reads the suggested filename off /export, so
    # omitting this silently degrades every download to a default name.
    expose_headers=["X-Suggested-Filename"],
)

#: Set by the sidecar so ``/shutdown`` can stop the server it is running under.
_shutdown_hook: Optional[Callable[[], None]] = None

#: Lazily-built consultation store; see :func:`get_store`.
_store = None


def reset_store() -> None:
    """Drop the cached store. Used by tests to point at a temp directory."""
    global _store
    _store = None


def set_shutdown_hook(hook: Optional[Callable[[], None]]) -> None:
    """Register what ``/shutdown`` should call. Used by the sidecar runner."""
    global _shutdown_hook
    _shutdown_hook = hook


def expected_token() -> str:
    """The configured bearer token, or ``""`` when auth is disabled."""
    return os.environ.get(TOKEN_ENV, "")


def generate_token() -> str:
    """A fresh token for one process lifetime."""
    return secrets.token_urlsafe(32)


async def require_token(request: Request) -> None:
    """Reject requests without the bearer token.

    A no-op when no token is configured, so ``meeting-notes serve`` stays
    usable for local development without ceremony. The sidecar never runs
    without one.
    """
    token = expected_token()
    if not token:
        return

    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")

    if scheme.lower() != "bearer" or not presented:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    # Constant-time compare: the token guards patient audio, and a timing
    # oracle on a loopback socket is still an oracle.
    if not secrets.compare_digest(presented, token):
        raise HTTPException(status_code=403, detail="Invalid token")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SpeakerSegmentResponse(BaseModel):
    """Mirrors SpeakerSegment in mobile/lib/types.ts."""

    text: str
    t0: float           # centiseconds
    t1: float
    lang: str = "en"
    confidence: Optional[float] = None
    #: "Person N" from diarization. None = unlabelled (diarization off,
    #: unavailable, or a monologue — the apps show a flat transcript then).
    speaker: Optional[str] = None


class TranscriptResponse(BaseModel):
    text: str
    language: str
    language_name: str
    timed_transcript: str
    backend: str = ""
    segments: list[SpeakerSegmentResponse] = []


class NotesResponse(BaseModel):
    title: str
    summary: str
    key_topics: list[str]
    decisions: list[str]
    action_items: list[str]
    language: str
    medical_terms: list[str] = []


class SummarizeRequest(BaseModel):
    text: str
    language: str = "en"
    template: str = "meeting"
    use_ollama: bool = True
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"


class SaveMeetingRequest(BaseModel):
    transcript: dict
    notes: dict
    duration_seconds: float = 0.0
    speaker_names: dict = {}
    #: Present when updating an existing record; omitted to create one.
    id: Optional[str] = None


class MeetingSummaryResponse(BaseModel):
    id: str
    created_at: str
    duration_seconds: float
    language: str
    backend: str
    title: str
    preview: str


class ExportRequest(BaseModel):
    notes: dict
    transcript: Optional[str] = None
    output_format: str = "markdown"


class ProcessResponse(BaseModel):
    transcript: TranscriptResponse
    notes: NotesResponse


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Liveness. Deliberately unauthenticated — the shell polls this first."""
    return {"status": "ok", "version": "0.1.0", "auth": bool(expected_token())}


@app.get("/backends", dependencies=[Depends(require_token)])
async def backends():
    """Report which engines are installed — the desktop startup check."""
    from meeting_notes.backends.router import available_backends

    return {"backends": available_backends()}


@app.post("/shutdown", dependencies=[Depends(require_token)])
async def shutdown():
    """Stop the server.

    The desktop shell calls this when its window closes. Without it, killing
    the Tauri process can leave an orphaned sidecar holding a model in memory
    and a port open.
    """
    if _shutdown_hook is None:
        raise HTTPException(
            status_code=501,
            detail="No shutdown hook registered; not running under the sidecar.",
        )
    _shutdown_hook()
    return {"status": "stopping"}


@app.post("/transcribe", response_model=TranscriptResponse,
          dependencies=[Depends(require_token)])
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model: str = Form(DEFAULT_WHISPER_MODEL),
    backend: Optional[str] = Form(None),
    diarize: str = Form("true"),
):
    """Transcribe an uploaded audio file, routing by language mode."""
    from meeting_notes.backends.router import resolve_backend

    config = _build_config(
        language=language,
        model=model,
        backend=backend,
        diarize=_form_bool(diarize),
    )
    tmp_path = await _save_upload(file)

    try:
        engine = resolve_backend(config)
        result = engine.transcribe_file(tmp_path)
        if config.diarize and result.segments:
            from meeting_notes.diarize import assign_speakers

            assign_speakers(result.segments, tmp_path)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    return _transcript_response(result)


@app.post("/summarize", response_model=NotesResponse,
          dependencies=[Depends(require_token)])
async def summarize(req: SummarizeRequest):
    """Generate structured meeting notes from a transcript."""
    summarizer = Summarizer(
        use_ollama=req.use_ollama,
        ollama_url=req.ollama_url,
        ollama_model=req.ollama_model,
    )
    try:
        notes = summarizer.summarize(
            req.text, language=req.language, template=req.template
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return NotesResponse(
        title=notes.title,
        summary=notes.summary,
        key_topics=notes.key_topics,
        decisions=notes.decisions,
        action_items=notes.action_items,
        language=notes.language,
        medical_terms=notes.medical_terms,
    )


@app.post("/process", response_model=ProcessResponse,
          dependencies=[Depends(require_token)])
async def process(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    whisper_model: str = Form(DEFAULT_WHISPER_MODEL),
    backend: Optional[str] = Form(None),
    template: str = Form("meeting"),
    use_ollama: str = Form("true"),
    ollama_url: str = Form("http://localhost:11434"),
    ollama_model: str = Form("llama3"),
    diarize: str = Form("true"),
):
    """Transcribe an audio file and generate meeting notes in one call."""
    config = _build_config(
        language=language,
        model=whisper_model,
        backend=backend,
        template=template,
        use_ollama=_form_bool(use_ollama),
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        diarize=_form_bool(diarize),
    )
    tmp_path = await _save_upload(file)

    try:
        result = run_pipeline(tmp_path, config)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    notes = result.notes
    return ProcessResponse(
        transcript=_transcript_response(result.transcript),
        notes=NotesResponse(
            title=notes.title,
            summary=notes.summary,
            key_topics=notes.key_topics,
            decisions=notes.decisions,
            action_items=notes.action_items,
            language=notes.language,
        ),
    )


# ---------------------------------------------------------------------------
# Stored consultations
# ---------------------------------------------------------------------------


def get_store():
    """The process-wide :class:`~meeting_notes.storage.MeetingStore`.

    Built lazily so importing this module does not create a database, and
    cached so the vault key is read once rather than per request.
    """
    global _store
    if _store is None:
        from meeting_notes.storage import MeetingStore  # noqa: PLC0415

        _store = MeetingStore(passphrase=os.environ.get(PASSPHRASE_ENV) or None)
    return _store


@app.post("/meetings", dependencies=[Depends(require_token)])
async def save_meeting(req: SaveMeetingRequest):
    """Save (or update) a consultation."""
    from meeting_notes.storage import MeetingRecord  # noqa: PLC0415

    record = MeetingRecord.new(
        transcript=req.transcript,
        notes=req.notes,
        duration_seconds=req.duration_seconds,
        speaker_names=req.speaker_names,
    )
    if req.id:
        record.id = req.id

    try:
        record_id = get_store().save(record)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"id": record_id, "created_at": record.created_at}


@app.get(
    "/meetings",
    response_model=list[MeetingSummaryResponse],
    dependencies=[Depends(require_token)],
)
async def list_meetings(
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """History list, most recent first. Pass ``q`` to search."""
    store = get_store()
    try:
        summaries = store.search(q, limit=limit) if q else store.list(limit, offset)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return [MeetingSummaryResponse(**vars(s)) for s in summaries]


@app.get("/meetings/{record_id}", dependencies=[Depends(require_token)])
async def get_meeting(record_id: str):
    """Full record, decrypted."""
    try:
        record = get_store().get(record_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if record is None:
        raise HTTPException(status_code=404, detail="No such consultation")
    return record.to_dict()


@app.delete("/meetings/{record_id}", dependencies=[Depends(require_token)])
async def delete_meeting(record_id: str):
    """Permanently delete a consultation."""
    if not get_store().delete(record_id):
        raise HTTPException(status_code=404, detail="No such consultation")
    return {"status": "deleted", "id": record_id}


@app.post("/export", dependencies=[Depends(require_token)])
async def export(req: ExportRequest):
    """Render notes to a downloadable document.

    Returns bytes rather than writing a file, so the desktop app can hand the
    result to a native save dialog without a patient consultation passing
    through a temp directory on the way.
    """
    from fastapi.responses import Response  # noqa: PLC0415

    from meeting_notes.document import (  # noqa: PLC0415
        EXPORT_FORMATS,
        render_document,
        suggested_filename,
    )
    from meeting_notes.summarizer import MeetingNotes  # noqa: PLC0415

    if req.output_format not in EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid output_format '{req.output_format}'. "
                f"Choose from: {', '.join(EXPORT_FORMATS)}"
            ),
        )

    notes = MeetingNotes(
        title=req.notes.get("title", "Notes"),
        summary=req.notes.get("summary", ""),
        key_topics=req.notes.get("key_topics", []),
        decisions=req.notes.get("decisions", []),
        action_items=req.notes.get("action_items", []),
        medical_terms=req.notes.get("medical_terms", []),
        language=req.notes.get("language", "en"),
    )

    try:
        payload = render_document(notes, req.transcript, req.output_format)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    filename = suggested_filename(notes, req.output_format)
    return Response(
        content=payload,
        media_type=EXPORT_FORMATS[req.output_format],
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            # The frontend reads the suggested name to pre-fill the save
            # dialog; Content-Disposition is awkward to parse from fetch().
            "X-Suggested-Filename": filename,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _form_bool(value: str) -> bool:
    """Parse an HTML-form boolean the way ``use_ollama`` always has."""
    return value.lower() not in ("false", "0", "no")


def _build_config(
    language: Optional[str],
    model: str,
    backend: Optional[str],
    template: str = "meeting",
    use_ollama: bool = True,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "llama3",
    diarize: bool = True,
) -> AppConfig:
    """Build and validate config, turning bad input into a 400."""
    config = AppConfig(
        whisper_model=model,
        language_mode=LanguageMode.from_value(language),
        backend=backend or None,
        template=template,
        use_ollama=use_ollama,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        diarize=diarize,
    )
    try:
        config.validate()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return config


async def _save_upload(file: UploadFile) -> str:
    """Persist an upload to a temp file and return its path."""
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        return tmp.name


def _transcript_response(result) -> TranscriptResponse:
    """Convert a TranscriptionResult into the wire format."""
    return TranscriptResponse(
        text=result.text,
        language=result.language,
        language_name=result.language_name,
        timed_transcript=result.timed_transcript or result.text,
        backend=result.backend,
        segments=[
            SpeakerSegmentResponse(
                text=seg.text,
                t0=seg.start * 100,  # seconds → centiseconds, as mobile expects
                t1=seg.end * 100,
                lang=seg.lang,
                confidence=seg.confidence,
                speaker=seg.speaker,
            )
            for seg in result.segments
        ],
    )
