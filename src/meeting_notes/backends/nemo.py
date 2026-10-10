"""NeMo FastConformer backend via sherpa-onnx — the Tagalog engine.

Wraps ``NCSpeech/stt_tl_fastconformer_hybrid_large`` (115M params, ~520h of
Tagalog, CC-BY-4.0).  Its published FLEURS ``fil_ph`` WER of 9.34% is the best
number available for Tagalog by a wide margin; its own conversational
benchmark (16.10%) is the more realistic expectation for clinic audio.

Running it through sherpa-onnx rather than the NeMo framework keeps PyTorch
out of the desktop bundle and gives us VAD from the same runtime.  The model
must be exported to ONNX first — see ``scripts/export_nemo_to_onnx.py``.

The model is Tagalog-only.  It has no documented English or code-switching
handling, so expect it to mangle English medical terms; that asymmetry is
exactly why the router keeps a separate English engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from meeting_notes.backends.base import (
    BackendUnavailableError,
    TranscriptionBackend,
    TranscriptionResult,
    TranscriptSegment,
    language_name,
    resample,
)
from meeting_notes.config import DEFAULT_SAMPLE_RATE, models_dir

logger = logging.getLogger(__name__)

#: Sub-directory under the models dir holding the exported ONNX files.
MODEL_SUBDIR = "stt_tl_fastconformer"

#: Files sherpa-onnx needs for an offline NeMo transducer (RNN-T head).
TRANSDUCER_FILES = ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")

#: Files needed for the CTC head. Simpler to export, marginally less accurate,
#: and the default the export script produces.
CTC_FILES = ("model.onnx", "tokens.txt")

#: Optional Silero VAD model — enables per-utterance segmentation.
VAD_FILENAME = "silero_vad.onnx"

_EXPORT_HINT = (
    "Export the model first:\n"
    "    python scripts/export_nemo_to_onnx.py\n"
    "This downloads NCSpeech/stt_tl_fastconformer_hybrid_large and writes the\n"
    "ONNX files sherpa-onnx needs. It is a one-time step and needs the `nemo`\n"
    "extra:\n"
    "    pip install -e '.[nemo]'"
)

_INSTALL_HINT = (
    "Install it with:\n"
    "    pip install sherpa-onnx\n"
    "or reinstall the project with the Tagalog engine:\n"
    "    pip install -e '.[tagalog]'"
)


class NemoTagalogBackend(TranscriptionBackend):
    """Offline Tagalog speech-to-text via sherpa-onnx.

    Args:
        model_dir: Directory holding the exported ONNX files. Defaults to
            ``<models_dir>/stt_tl_fastconformer``.
        num_threads: ONNX Runtime intra-op threads.
        use_vad: Segment with Silero VAD when the model is present. Without
            it the whole file decodes as a single segment, which loses
            timestamps.
        hotwords: Terms to bias decoding toward. Forces modified beam search.
            The hook the bilingual medical lexicon will use.
        hotwords_score: Bias strength. Higher favours hotwords more.
    """

    name = "nemo-tagalog"
    language = "tl"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        num_threads: int = 4,
        use_vad: bool = True,
        hotwords: Optional[list[str]] = None,
        hotwords_score: float = 1.5,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else models_dir() / MODEL_SUBDIR
        self.num_threads = num_threads
        self.use_vad = use_vad
        self.hotwords = hotwords or []
        self.hotwords_score = hotwords_score
        self._recognizer = None
        self._vad = None

    # ------------------------------------------------------------------
    # TranscriptionBackend
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            import sherpa_onnx  # noqa: F401, PLC0415
        except ImportError:
            return False
        return self.model_type() is not None

    def model_type(self) -> Optional[str]:
        """Detect which head was exported: ``"transducer"``, ``"ctc"``, or None.

        The checkpoint is hybrid, so either head is valid. Transducer is
        slightly more accurate and is the only one supporting hotword biasing;
        CTC is what the export script produces by default because exporting it
        is a single documented call.
        """
        if all((self.model_dir / f).exists() for f in TRANSDUCER_FILES):
            return "transducer"
        if all((self.model_dir / f).exists() for f in CTC_FILES):
            return "ctc"
        return None

    def missing_files(self) -> list[str]:
        """Return what is missing for the closest-matching export."""
        if self.model_type() is not None:
            return []
        # Report against CTC — the smaller, default export.
        return [f for f in CTC_FILES if not (self.model_dir / f).exists()]

    def warmup(self) -> None:
        self._load_recognizer()

    def transcribe_array(
        self,
        audio_array,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> TranscriptionResult:
        """Transcribe a 1-D float32 mono array normalised to [-1, 1]."""
        import numpy as np  # noqa: PLC0415

        audio_array = resample(audio_array, sample_rate, DEFAULT_SAMPLE_RATE)
        audio = np.asarray(audio_array, dtype=np.float32)
        recognizer = self._load_recognizer()

        regions = self._speech_regions(audio)
        segments: list[TranscriptSegment] = []

        for start_sample, region in regions:
            text = self._decode(recognizer, region)
            if not text.strip():
                continue
            start = start_sample / DEFAULT_SAMPLE_RATE
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=start + len(region) / DEFAULT_SAMPLE_RATE,
                    text=text,
                    lang=self.language,
                    # sherpa-onnx's offline transducer exposes no per-segment
                    # score; leave it unset rather than inventing one.
                    confidence=None,
                )
            )

        return TranscriptionResult(
            text=" ".join(s.text.strip() for s in segments).strip(),
            language=self.language,
            language_name=language_name(self.language),
            segments=segments,
            backend=self.name,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_sherpa(self):
        try:
            import sherpa_onnx  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailableError(
                self.name,
                "the sherpa-onnx package is not installed",
                _INSTALL_HINT,
            ) from exc
        return sherpa_onnx

    def _load_recognizer(self):
        """Lazy-load the sherpa-onnx offline recognizer."""
        if self._recognizer is not None:
            return self._recognizer

        sherpa_onnx = self._require_sherpa()

        model_type = self.model_type()
        if model_type is None:
            raise BackendUnavailableError(
                self.name,
                f"no exported model in {self.model_dir} "
                f"(missing: {', '.join(self.missing_files())})",
                _EXPORT_HINT,
            )

        common = {
            "tokens": str(self.model_dir / "tokens.txt"),
            "num_threads": self.num_threads,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "feature_dim": 80,
        }

        if model_type == "transducer":
            kwargs = {
                **common,
                "encoder": str(self.model_dir / "encoder.onnx"),
                "decoder": str(self.model_dir / "decoder.onnx"),
                "joiner": str(self.model_dir / "joiner.onnx"),
                "model_type": "nemo_transducer",
            }
            if self.hotwords:
                # Hotword biasing needs modified beam search; greedy ignores it.
                kwargs["decoding_method"] = "modified_beam_search"
                kwargs["hotwords_score"] = self.hotwords_score
                hotwords_path = self._write_hotwords_file()
                if hotwords_path:
                    kwargs["hotwords_file"] = hotwords_path
            else:
                kwargs["decoding_method"] = "greedy_search"

            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(**kwargs)
        else:
            if self.hotwords:
                logger.warning(
                    "Hotword biasing is ignored for the CTC head. Re-export "
                    "with --transducer to enable it."
                )
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(self.model_dir / "model.onnx"),
                decoding_method="greedy_search",
                **common,
            )

        return self._recognizer

    def _write_hotwords_file(self) -> Optional[str]:
        """Write hotwords to a file next to the model, as sherpa-onnx expects."""
        if not self.hotwords:
            return None
        path = self.model_dir / "hotwords.txt"
        try:
            path.write_text("\n".join(self.hotwords), encoding="utf-8")
        except OSError:
            return None
        return str(path)

    def _load_vad(self):
        """Lazy-load Silero VAD; returns None when the model is absent."""
        if self._vad is not None:
            return self._vad
        if not self.use_vad:
            return None

        vad_path = self.model_dir / VAD_FILENAME
        if not vad_path.exists():
            return None

        sherpa_onnx = self._require_sherpa()
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(vad_path)
        config.silero_vad.min_silence_duration = 0.25
        config.sample_rate = DEFAULT_SAMPLE_RATE

        self._vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=100
        )
        return self._vad

    def _speech_regions(self, audio) -> list[tuple[int, "object"]]:
        """Split audio into (start_sample, samples) speech regions.

        Falls back to a single whole-file region when VAD is unavailable, so
        the backend still works before the VAD model is downloaded.
        """
        vad = self._load_vad()
        if vad is None:
            return [(0, audio)]

        # Silero VAD must be fed in window-sized blocks — see drain_vad.
        from meeting_notes.backends.omnilingual import drain_vad  # noqa: PLC0415

        return drain_vad(vad, audio) or [(0, audio)]

    def _decode(self, recognizer, samples) -> str:
        """Run one decode pass over a block of samples."""
        stream = recognizer.create_stream()
        stream.accept_waveform(DEFAULT_SAMPLE_RATE, samples)
        recognizer.decode_stream(stream)
        return stream.result.text


def default_model_dir() -> Path:
    """Where the exported Tagalog ONNX files are expected to live."""
    return models_dir() / MODEL_SUBDIR
