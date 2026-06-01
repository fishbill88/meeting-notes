"""Transcription module – uses OpenAI Whisper for offline, multi-language transcription."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from meeting_notes.config import DEFAULT_SAMPLE_RATE, WHISPER_MODELS


@dataclass
class TranscriptSegment:
    """A single timed segment of the transcript."""

    start: float   # seconds
    end: float     # seconds
    text: str

    def __str__(self) -> str:
        minutes_start, secs_start = divmod(int(self.start), 60)
        minutes_end, secs_end = divmod(int(self.end), 60)
        return f"[{minutes_start:02d}:{secs_start:02d} – {minutes_end:02d}:{secs_end:02d}] {self.text.strip()}"


@dataclass
class TranscriptionResult:
    """Full transcription result including detected language and segments."""

    text: str
    language: str
    language_name: str
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def timed_transcript(self) -> str:
        """Return transcript with timestamps for each segment."""
        return "\n".join(str(s) for s in self.segments)


class Transcriber:
    """Offline speech-to-text transcription using OpenAI Whisper.

    Whisper supports 99+ languages including Filipino/Tagalog (``tl``).
    All processing is local – no internet required after the model is
    downloaded.

    Args:
        model_size: One of ``tiny``, ``base``, ``small``, ``medium``, ``large``.
            Larger models are more accurate but require more RAM/disk.
        language: ISO 639-1 language code (e.g. ``"tl"`` for Filipino,
            ``"en"`` for English).  Pass ``None`` to let Whisper auto-detect.
        device: ``"cpu"`` or ``"cuda"`` (if a CUDA GPU is available).
    """

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        if model_size not in WHISPER_MODELS:
            raise ValueError(
                f"Invalid model_size '{model_size}'. "
                f"Choose from: {', '.join(WHISPER_MODELS)}"
            )
        self.model_size = model_size
        self.language = language
        self.device = device
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file (WAV, MP3, M4A, …).

        Args:
            audio_path: Path to the audio file.

        Returns:
            :class:`TranscriptionResult` with full text, detected language,
            and timed segments.
        """
        audio_path = str(Path(audio_path).resolve())
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._load_model()

        options: dict = {
            "fp16": False,  # Use FP32 for CPU compatibility
            "verbose": False,
        }
        if self.language:
            options["language"] = self.language

        result = model.transcribe(audio_path, **options)

        return self._parse_result(result)

    def transcribe_array(self, audio_array, sample_rate: int = DEFAULT_SAMPLE_RATE) -> TranscriptionResult:
        """Transcribe a raw NumPy float32 audio array.

        Args:
            audio_array: 1-D float32 NumPy array (mono, normalised to [-1, 1]).
            sample_rate: Sample rate in Hz.

        Returns:
            :class:`TranscriptionResult`.
        """
        import numpy as np  # noqa: PLC0415

        model = self._load_model()

        if sample_rate != DEFAULT_SAMPLE_RATE:
            audio_array = _resample(audio_array, sample_rate, DEFAULT_SAMPLE_RATE)

        options: dict = {
            "fp16": False,
            "verbose": False,
        }
        if self.language:
            options["language"] = self.language

        result = model.transcribe(audio_array.astype(np.float32), **options)

        return self._parse_result(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy-load the Whisper model (downloads on first use)."""
        if self._model is None:
            try:
                import whisper  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "openai-whisper is required for transcription. "
                    "Install it with: pip install openai-whisper"
                ) from exc
            self._model = whisper.load_model(self.model_size, device=self.device)
        return self._model

    def _parse_result(self, result: dict) -> TranscriptionResult:
        """Parse the raw Whisper output dict into a :class:`TranscriptionResult`."""
        language_code = result.get("language", "unknown")
        language_name = _language_name(language_code)

        segments = [
            TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
            )
            for seg in result.get("segments", [])
        ]

        return TranscriptionResult(
            text=result.get("text", "").strip(),
            language=language_code,
            language_name=language_name,
            segments=segments,
        )


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------


def _language_name(code: str) -> str:
    """Return a human-readable language name for an ISO code."""
    from meeting_notes.config import SUPPORTED_LANGUAGES  # noqa: PLC0415
    return SUPPORTED_LANGUAGES.get(code, code)


def _resample(audio, orig_rate: int, target_rate: int):
    """Simple linear resampling when scipy is not available."""
    try:
        from scipy.signal import resample_poly  # noqa: PLC0415
        import math
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
