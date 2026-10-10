"""faster-whisper (CTranslate2) backend — the English engine.

Replaces the reference ``openai-whisper`` implementation: same weights, same
accuracy, roughly 4x faster on CPU at int8 and about half the memory.  That
speedup is what makes ``large-v3-turbo`` viable on clinic hardware without a
GPU.

Model choice is deliberate — see ``MODELS.md``.  ``large-v3`` runs at well
under real time on CPU; ``large-v3-turbo`` keeps a large-v2-class error rate
with a 4-layer decoder instead of 32.
"""

from __future__ import annotations

from typing import Optional

from meeting_notes.backends.base import (
    BackendUnavailableError,
    TranscriptionBackend,
    TranscriptionResult,
    TranscriptSegment,
    _require_file,
    language_name,
    logprob_to_confidence,
    resample,
)
from meeting_notes.config import DEFAULT_SAMPLE_RATE

_INSTALL_HINT = (
    "Install it with:\n"
    "    pip install faster-whisper\n"
    "or reinstall the project:\n"
    "    pip install -e ."
)


class FasterWhisperBackend(TranscriptionBackend):
    """Offline speech-to-text via faster-whisper.

    Args:
        model_size: A faster-whisper model identifier, e.g. ``"large-v3-turbo"``.
        language: ISO 639-1 code to force, or ``None`` to auto-detect.
        device: ``"cpu"``, ``"cuda"``, or ``"auto"``.
        compute_type: CTranslate2 quantisation. ``"int8"`` on CPU,
            ``"float16"`` on CUDA. ``"auto"`` picks per device.
        beam_size: Beam width. 1 is greedy and noticeably faster.
        initial_prompt: Optional text prepended as decoder context. This is
            Whisper's contextual-biasing mechanism — the hook the medical
            lexicon will use.
    """

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "auto",
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = _resolve_compute_type(compute_type, device)
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # TranscriptionBackend
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    def warmup(self) -> None:
        self._load_model()

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file.

        faster-whisper reads the file itself (via PyAV), so non-WAV formats
        work without a separate ffmpeg step.
        """
        return self._transcribe(_require_file(audio_path))

    def transcribe_array(
        self,
        audio_array,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> TranscriptionResult:
        """Transcribe a 1-D float32 mono array normalised to [-1, 1]."""
        import numpy as np  # noqa: PLC0415

        audio_array = resample(audio_array, sample_rate, DEFAULT_SAMPLE_RATE)
        return self._transcribe(np.asarray(audio_array, dtype=np.float32))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy-load the model (downloads on first use)."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel  # noqa: PLC0415
            except ImportError as exc:
                raise BackendUnavailableError(
                    self.name,
                    "the faster-whisper package is not installed",
                    _INSTALL_HINT,
                ) from exc

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def _transcribe(self, audio) -> TranscriptionResult:
        model = self._load_model()

        segments_iter, info = model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            initial_prompt=self.initial_prompt,
        )

        detected = getattr(info, "language", None) or self.language or "unknown"

        # faster-whisper streams segments lazily; decoding happens on iteration.
        segments = [
            TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                lang=detected,
                confidence=logprob_to_confidence(getattr(seg, "avg_logprob", None)),
            )
            for seg in segments_iter
        ]

        return TranscriptionResult(
            text=" ".join(s.text.strip() for s in segments).strip(),
            language=detected,
            language_name=language_name(detected),
            segments=segments,
            backend=self.name,
        )


def _resolve_compute_type(compute_type: str, device: str) -> str:
    """Pick a sensible CTranslate2 quantisation when set to ``"auto"``."""
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"
