"""The shared transcribe → summarize pipeline.

Both the CLI and the REST server call :func:`run_pipeline`. Keeping one
implementation here is what stops them drifting apart — before this module
they each had their own copy, and the server's had already lost the CLI's
language handling.

This function is deliberately free of terminal output and process exits so it
is safe to call from a server request handler, a test, or the desktop sidecar.
Progress is reported through an optional callback instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from meeting_notes.backends.base import TranscriptionResult
from meeting_notes.config import AppConfig
from meeting_notes.summarizer import MeetingNotes, Summarizer

#: ``(stage, message)`` — stage is one of ``"transcribe"``, ``"diarize"``,
#: or ``"summarize"``.
ProgressCallback = Callable[[str, str], None]


@dataclass
class PipelineResult:
    """Everything the pipeline produced, ready to render or persist."""

    transcript: TranscriptionResult
    notes: MeetingNotes

    @property
    def backend(self) -> str:
        """Which engine actually ran.

        Worth surfacing in the UI: the Tagalog router falls back to Whisper
        when the NeMo model has not been exported, and a clinician should be
        able to see that rather than silently getting the weaker engine.
        """
        return self.transcript.backend

    @property
    def timed_transcript(self) -> str:
        """Timestamped transcript, falling back to plain text."""
        return self.transcript.timed_transcript or self.transcript.text


def run_pipeline(
    audio_path: str,
    config: AppConfig,
    on_progress: Optional[ProgressCallback] = None,
) -> PipelineResult:
    """Transcribe ``audio_path`` and generate structured meeting notes.

    Args:
        audio_path: Path to an audio file in any format the backend accepts.
        config: Validated app configuration. Its ``language_mode`` selects
            the engine.
        on_progress: Optional ``(stage, message)`` callback for UI feedback.

    Returns:
        :class:`PipelineResult` with the transcript and the notes.

    Raises:
        NotImplementedError: If ``config.language_mode`` is Taglish.
        BackendUnavailableError: If the selected engine cannot run.
    """
    from meeting_notes.backends.router import resolve_backend  # noqa: PLC0415

    report = on_progress or (lambda stage, message: None)

    backend = resolve_backend(config)
    report("transcribe", f"Transcribing with {backend.name}…")
    transcript = backend.transcribe_file(audio_path)

    if config.diarize and transcript.segments:
        from meeting_notes.diarize import assign_speakers  # noqa: PLC0415

        report("diarize", "Identifying speakers…")
        assign_speakers(transcript.segments, audio_path)

    engine = "Ollama" if config.use_ollama else "extractive"
    report("summarize", f"Generating meeting notes ({engine})…")
    summarizer = Summarizer(
        use_ollama=config.use_ollama,
        ollama_url=config.ollama_url,
        ollama_model=config.ollama_model,
    )
    notes = summarizer.summarize(
        transcript.text,
        language=transcript.language,
        template=config.template,
    )

    return PipelineResult(transcript=transcript, notes=notes)
