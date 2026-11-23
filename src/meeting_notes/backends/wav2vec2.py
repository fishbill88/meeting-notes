"""A wav2vec2 CTC model (Hugging Face) as a benchmark engine — built to MEASURE Cebuano candidates.

The first target is ``kylegregory/wav2vec2-large-xlsr-53-ceb`` (Apache-2.0, 0.3B parameters, reports 5.5% WER
on its own Cebuano split). It is not wired into language routing: nothing about it has been measured on
meeting audio yet, and the model card itself says code-switched speech is a weak spot. Select it explicitly::

    meeting-notes bench --dataset fleurs-ceb --engine omnilingual-ctc --engine wav2vec2-ctc

The model is read from ``MEETING_NOTES_WAV2VEC2_MODEL`` (a Hugging Face repo id or a local folder, which is
what to use for a gated repo you have already downloaded), else the default below.

Needs ``torch`` and ``transformers``, which the engine does NOT ship and does not import until this backend
runs: they are the reason the sidecar is ~230 MB rather than ~1.5 GB. If this model wins, the next step is an
ONNX export, not bundling torch.

Long audio is cut into fixed windows. CTC has no context across a cut, so a word on the boundary can be
damaged; that is acceptable for a benchmark and is why this is not a production path.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from meeting_notes.backends.base import (
    BackendUnavailableError,
    TranscriptionBackend,
    TranscriptionResult,
    TranscriptSegment,
    language_name,
    resample,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "kylegregory/wav2vec2-large-xlsr-53-ceb"
MODEL_ENV = "MEETING_NOTES_WAV2VEC2_MODEL"

#: What the model was trained on.
SAMPLE_RATE = 16_000

#: Window length for long audio, seconds. Short enough for memory on a laptop, long enough that cuts are rare.
WINDOW_SECONDS = 20.0


class Wav2Vec2CtcBackend(TranscriptionBackend):
    name = "wav2vec2-ctc"

    def __init__(self, model: Optional[str] = None, language: str = "ceb", device: Optional[str] = None) -> None:
        self.model_id = model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
        self.language = language
        self._device = device
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------ availability

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401, PLC0415
            import transformers  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    def _require(self) -> None:
        if not self.is_available():
            raise BackendUnavailableError(
                self.name,
                "torch and transformers are not installed",
                'pip install torch transformers   (benchmark use only; the engine does not ship them)',
            )

    def warmup(self) -> None:
        self._load()

    def _load(self) -> None:
        if self._model is not None:
            return
        self._require()
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCTC, AutoProcessor  # noqa: PLC0415

        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForCTC.from_pretrained(self.model_id)
        except OSError as exc:
            raise BackendUnavailableError(
                self.name,
                f"could not load '{self.model_id}': {exc}",
                "If the repo is gated, accept its terms on huggingface.co, run `huggingface-cli login`, "
                f"or download it and point {MODEL_ENV} at the folder.",
            ) from exc
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._model.to(device).eval()

    # ------------------------------------------------------------------ transcription

    def transcribe_array(self, audio_array, sample_rate: int = SAMPLE_RATE) -> TranscriptionResult:
        import numpy as np  # noqa: PLC0415

        self._load()
        import torch  # noqa: PLC0415

        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            audio = resample(audio, sample_rate, SAMPLE_RATE)

        step = int(WINDOW_SECONDS * SAMPLE_RATE)
        segments: list[TranscriptSegment] = []
        for start in range(0, max(len(audio), 1), step):
            piece = audio[start:start + step]
            if len(piece) < SAMPLE_RATE // 10:      # under 100 ms: nothing to say
                continue
            inputs = self._processor(piece, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(inputs.input_values.to(self._device)).logits
            text = self._processor.batch_decode(torch.argmax(logits, dim=-1))[0]
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                segments.append(TranscriptSegment(
                    start=start / SAMPLE_RATE, end=min(start + step, len(audio)) / SAMPLE_RATE,
                    text=text, lang=self.language,
                ))

        return TranscriptionResult(
            text=" ".join(s.text for s in segments),
            language=self.language,
            language_name=language_name(self.language),
            segments=segments,
            backend=self.name,
        )
