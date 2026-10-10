"""Transcription facade.

The engines themselves live in :mod:`meeting_notes.backends`. This module keeps
the historical ``Transcriber`` API working and re-exports the shared result
types so existing imports do not break.

New code should prefer :func:`meeting_notes.backends.router.resolve_backend`,
which picks the right engine for the configured language mode.
"""

from __future__ import annotations

from typing import Optional

from meeting_notes.backends.base import (
    BackendUnavailableError,
    TranscriptionBackend,
    TranscriptionResult,
    TranscriptSegment,
    language_name as _language_name,
    resample as _resample,
)
from meeting_notes.config import DEFAULT_SAMPLE_RATE, DEFAULT_WHISPER_MODEL, WHISPER_MODELS

__all__ = [
    "BackendUnavailableError",
    "Transcriber",
    "TranscriptionBackend",
    "TranscriptionResult",
    "TranscriptSegment",
]


class Transcriber:
    """Offline speech-to-text, defaulting to faster-whisper.

    Retained for compatibility with existing callers. It is a thin wrapper
    around a :class:`~meeting_notes.backends.base.TranscriptionBackend`.

    Args:
        model_size: A Whisper model identifier from
            :data:`~meeting_notes.config.WHISPER_MODELS`.
        language: ISO 639-1 code (``"tl"`` for Filipino, ``"en"`` for English).
            ``None`` lets the engine auto-detect.
        device: ``"cpu"``, ``"cuda"``, or ``"auto"``.
        backend: Pre-built backend to use instead of constructing a Whisper
            one. Mainly useful for tests and for routing to the Tagalog engine.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_WHISPER_MODEL,
        language: Optional[str] = None,
        device: str = "cpu",
        backend: Optional[TranscriptionBackend] = None,
    ) -> None:
        if model_size not in WHISPER_MODELS:
            raise ValueError(
                f"Invalid model_size '{model_size}'. "
                f"Choose from: {', '.join(WHISPER_MODELS)}"
            )
        self.model_size = model_size
        self.language = language
        self.device = device
        self._backend = backend

    @property
    def backend(self) -> TranscriptionBackend:
        """The underlying backend, constructed on first access."""
        if self._backend is None:
            from meeting_notes.backends.whisper import (  # noqa: PLC0415
                FasterWhisperBackend,
            )

            self._backend = FasterWhisperBackend(
                model_size=self.model_size,
                language=self.language,
                device=self.device,
            )
        return self._backend

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file (WAV, MP3, M4A, …)."""
        return self.backend.transcribe_file(audio_path)

    def transcribe_array(
        self,
        audio_array,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> TranscriptionResult:
        """Transcribe a raw NumPy float32 audio array."""
        return self.backend.transcribe_array(audio_array, sample_rate)
