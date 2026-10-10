"""Pluggable transcription backends.

Each backend wraps one ASR engine behind :class:`TranscriptionBackend` so the
rest of the app never imports an engine directly.  Pick one via
:func:`meeting_notes.backends.router.resolve_backend`.

Backends are imported lazily by the router — importing this package does *not*
pull in faster-whisper or sherpa-onnx.
"""

from __future__ import annotations

from meeting_notes.backends.base import (
    BackendUnavailableError,
    TranscriptionBackend,
    TranscriptionResult,
    TranscriptSegment,
    language_name,
)

__all__ = [
    "BackendUnavailableError",
    "TranscriptionBackend",
    "TranscriptionResult",
    "TranscriptSegment",
    "language_name",
]
