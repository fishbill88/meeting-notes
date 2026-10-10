"""Shared types and the backend interface every ASR engine implements."""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from meeting_notes.config import DEFAULT_SAMPLE_RATE


class BackendUnavailableError(RuntimeError):
    """Raised when a backend's engine or model files are not installed.

    Carries a human-readable remedy so the CLI and the desktop sidecar can
    show the same actionable message.
    """

    def __init__(self, backend: str, reason: str, remedy: str = "") -> None:
        self.backend = backend
        self.reason = reason
        self.remedy = remedy
        message = f"Backend '{backend}' is unavailable: {reason}"
        if remedy:
            message = f"{message}\n\n{remedy}"
        super().__init__(message)


@dataclass
class TranscriptSegment:
    """A single timed segment of the transcript."""

    start: float                        # seconds
    end: float                          # seconds
    text: str
    lang: str = "en"                    # per-segment language tag
    confidence: Optional[float] = None  # 0.0–1.0, None when the engine has no estimate
    speaker: Optional[str] = None       # "Person N" from diarization; None = unlabelled

    def __str__(self) -> str:
        minutes_start, secs_start = divmod(int(self.start), 60)
        minutes_end, secs_end = divmod(int(self.end), 60)
        who = f"{self.speaker}: " if self.speaker else ""
        return (
            f"[{minutes_start:02d}:{secs_start:02d} – "
            f"{minutes_end:02d}:{secs_end:02d}] {who}{self.text.strip()}"
        )


@dataclass
class TranscriptionResult:
    """Full transcription result including detected language and segments."""

    text: str
    language: str
    language_name: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    backend: str = ""  # which backend produced this, for provenance in the UI

    @property
    def timed_transcript(self) -> str:
        """Return transcript with timestamps for each segment."""
        return "\n".join(str(s) for s in self.segments)


class TranscriptionBackend(ABC):
    """Interface every ASR engine implements.

    Subclasses must provide :meth:`transcribe_array`; :meth:`transcribe_file`
    has a working default that loads the file and delegates.  Engines that can
    read files more efficiently themselves (faster-whisper does) should
    override it.
    """

    #: Short stable identifier, e.g. ``"faster-whisper"``.
    name: str = "base"

    @abstractmethod
    def transcribe_array(
        self,
        audio_array,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> TranscriptionResult:
        """Transcribe a 1-D float32 mono array normalised to [-1, 1]."""

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file (WAV, MP3, M4A, …)."""
        audio_path = _require_file(audio_path)
        from meeting_notes.audio import load_audio_file  # noqa: PLC0415

        audio, sample_rate = load_audio_file(audio_path)
        return self.transcribe_array(audio, sample_rate)

    def is_available(self) -> bool:
        """Return True when this backend can run right now.

        Checks imports and model files without loading a model.  Backends that
        need no such check inherit this permissive default.
        """
        return True

    def warmup(self) -> None:
        """Optionally pre-load the model so the first transcription is fast.

        The desktop sidecar calls this during its startup splash.  The default
        is a no-op for backends with nothing to pre-load.
        """


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def language_name(code: str) -> str:
    """Return a human-readable language name for an ISO code."""
    from meeting_notes.config import SUPPORTED_LANGUAGES  # noqa: PLC0415

    return SUPPORTED_LANGUAGES.get(code, code)


def logprob_to_confidence(avg_logprob: Optional[float]) -> Optional[float]:
    """Map an average token log-probability onto a 0.0–1.0 confidence.

    Whisper reports ``avg_logprob`` per segment, roughly in [-2.0, 0.0].
    Exponentiating recovers a probability-like number that is monotonic in the
    model's own certainty — good enough to rank two decodes of the same audio
    against each other, which is what the Taglish two-pass merge will need.

    Not calibrated across engines: never compare a Whisper confidence to a
    sherpa-onnx one.
    """
    if avg_logprob is None:
        return None
    try:
        return max(0.0, min(1.0, math.exp(avg_logprob)))
    except (OverflowError, ValueError):
        return None


def _require_file(audio_path: str) -> str:
    """Resolve ``audio_path`` and raise if it does not exist."""
    resolved = str(Path(audio_path).resolve())
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return resolved


def resample(audio, orig_rate: int, target_rate: int):
    """Resample a 1-D audio array, preferring scipy when it is installed."""
    if orig_rate == target_rate:
        return audio
    try:
        import math  # noqa: PLC0415

        from scipy.signal import resample_poly  # noqa: PLC0415

        gcd = math.gcd(target_rate, orig_rate)
        return resample_poly(audio, target_rate // gcd, orig_rate // gcd)
    except ImportError:
        import numpy as np  # noqa: PLC0415

        ratio = target_rate / orig_rate
        new_length = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio) - 1, new_length),
            np.arange(len(audio)),
            audio,
        )
