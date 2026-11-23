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

import logging
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

logger = logging.getLogger(__name__)

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
            lexicon will use. It only reaches the first minute or so; see
            ``hotwords``.
        hotwords: The host's title, names and terms
            (:func:`meeting_notes.context.hint_text`). Unlike
            ``initial_prompt`` it is in front of every 30-second window, so it
            helps all the way through. Used ONLY with faster-whisper's speech
            filter on, and left out when the filter cannot load (see
            ``_transcribe``), and segments are then cut at sentence ends
            (:func:`split_at_sentences`). A segment that only reads it back
            is dropped (:func:`meeting_notes.context.is_echo`).
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
        hotwords: Optional[str] = None,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = _resolve_compute_type(compute_type, device)
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.hotwords = hotwords or None
        self._model = None  # lazy-loaded
        self._vad_ok: Optional[bool] = None  # asked once, when a hint first needs it

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

    def _vad_ready(self) -> bool:
        """Whether faster-whisper's own Silero speech filter loads.

        A frozen engine has the filter's model only when the build collected faster-whisper's package data
        (``scripts/build_sidecar.py`` does now; the engine built 2026-10-13 did not). Asked once per backend.
        faster-whisper caches the model, so the transcription that follows reuses this load.
        """
        if self._vad_ok is None:
            try:
                from faster_whisper.vad import get_vad_model  # noqa: PLC0415

                get_vad_model()
                self._vad_ok = True
            except Exception as exc:  # a missing file, onnxruntime refusing it, an older faster-whisper
                logger.warning("speech filter unavailable, transcribing without the context hint: %s", exc)
                self._vad_ok = False
        return self._vad_ok

    def _transcribe(self, audio) -> TranscriptionResult:
        model = self._load_model()

        # The hint is used only with the speech filter on. Measured on the Taglish fixture (RESEARCH.md §3d): with
        # a hint in every window, Whisper decoded the silence at the end as "Thank you very much.", timed at
        # 70.6-87.6 s of a 69.2 s file. The filter never lets silence reach the decoder, and with it on the same
        # run invented nothing and kept every spelling the hint fixed. Without the filter the hint is left out,
        # which is exactly how the engine behaved before hints existed.
        hotwords = self.hotwords if self.hotwords and self._vad_ready() else None
        options = {}
        if hotwords:
            # Only when there is something to say: arguments this backend never used stay out of the call.
            options["hotwords"] = hotwords
            options["vad_filter"] = True
            # With a hint Whisper writes fewer, longer segments, and the speaker step labels a WHOLE segment, so a
            # short reply sharing one with the next turn went to the wrong person (the Bisaya fixture: "Ready na
            # ni unya. Perfecto. ..."). Word times let each segment be cut at its sentence ends below.
            options["word_timestamps"] = True
        segments_iter, info = model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            initial_prompt=self.initial_prompt,
            **options,
        )

        detected = getattr(info, "language", None) or self.language or "unknown"
        duration = getattr(info, "duration", None)
        duration = duration if isinstance(duration, (int, float)) and duration > 0 else None

        # faster-whisper streams segments lazily; decoding happens on iteration, so each segment's end is how
        # far the decode has really got.
        segments = []
        echoes = 0
        for seg in segments_iter:
            self.report_seconds(seg.end)
            if duration is not None and seg.start >= duration:
                # Text timed after the audio ends cannot be speech: Whisper writing something for the silence.
                continue
            if hotwords:
                from meeting_notes.context import is_echo  # noqa: PLC0415

                # On the WHOLE segment, before it is cut: cut first, an echo's short pieces would each pass.
                if is_echo(seg.text, hotwords):
                    echoes += 1
                    continue
            confidence = logprob_to_confidence(getattr(seg, "avg_logprob", None))
            for start, end, text in split_at_sentences(seg) if hotwords else [(seg.start, seg.end, seg.text)]:
                if duration is not None and start >= duration:
                    continue
                segments.append(
                    TranscriptSegment(start=start, end=end, text=text, lang=detected, confidence=confidence)
                )

        if echoes:
            # A count only: the hint holds people's names.
            logger.info("dropped %d segment(s) that read the context hint back", echoes)

        return TranscriptionResult(
            text=" ".join(s.text.strip() for s in segments).strip(),
            language=detected,
            language_name=language_name(detected),
            segments=segments,
            backend=self.name,
        )


#: What ends a sentence in what Whisper writes for these meetings, Latin and CJK punctuation alike.
_SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")
#: Closers that may follow the full stop: `ready."` still ends a sentence.
_CLOSERS = "\"')]»”’"


def split_at_sentences(seg) -> list[tuple[float, float, str]]:
    """A decoded segment as ``(start, end, text)`` pieces, one per sentence, timed by its words.

    Turns change at sentence ends, so cutting there gives the speaker step one turn per piece. Cutting too
    often is harmless (both halves of "the U.S. office" keep one speaker); not cutting is what mislabels a
    reply. A segment without word times, or with one sentence, is returned whole and unchanged.
    """
    whole = [(seg.start, seg.end, seg.text)]
    words = list(getattr(seg, "words", None) or [])
    if len(words) < 2:
        return whole
    pieces: list[list] = [[]]
    for i, word in enumerate(words):
        pieces[-1].append(word)
        ends = word.word.rstrip().rstrip(_CLOSERS).endswith(_SENTENCE_END)
        if ends and i < len(words) - 1:
            pieces.append([])
    if len(pieces) == 1:
        return whole
    return [(p[0].start, p[-1].end, "".join(w.word for w in p)) for p in pieces]


def _resolve_compute_type(compute_type: str, device: str) -> str:
    """Pick a sensible CTranslate2 quantisation when set to ``"auto"``."""
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"
