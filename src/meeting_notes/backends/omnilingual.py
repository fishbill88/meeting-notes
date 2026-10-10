"""Meta Omnilingual ASR (CTC) via sherpa-onnx — the regional-language engine.

Wraps ``omniASR_CTC_300M`` from facebookresearch/omnilingual-asr (Apache-2.0),
packaged as int8 ONNX by the sherpa-onnx project. It was trained on 1,600+
languages including every major Philippine one — Cebuano (196.8 h of training
audio), Ilocano, Hiligaynon, Waray, Bikol, Kapampangan, Pangasinan — which
neither Whisper nor the Tagalog FastConformer cover at all.

Why the CTC head and not the LLM one: CTC has **no language token**. It emits
characters for whatever it hears, so a Bisaya meeting that drifts into English
or Tagalog mid-sentence is decoded by one model with no up-front language
choice. That also means the ``language`` on the result is the *requested* tag,
not a detection — the model does not know what language it just wrote.

Accuracy expectation: Meta publishes per-language numbers only for the 7B
LLM model (Cebuano 2.0 CER on read speech). The 300M CTC is roughly 2x worse
than the 7B on their benchmarks and still ahead of Whisper large-v3 on most
low-resource sets. Nothing has been measured on Cebuano *meeting* audio yet;
see RESEARCH.md section 9b.

Runtime: ~292 MB download on first use (auto), ~350 MB on disk, CPU-only via
onnxruntime, vendor RTF 0.225 on 2 threads. No torch.
"""

from __future__ import annotations

import logging
import tarfile
import urllib.request
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

#: Sub-directory under the models dir holding the ONNX files.
MODEL_SUBDIR = "omnilingual"

#: The sherpa-onnx release archive. int8 300M CTC — the smallest published
#: variant; a 1B CTC exists at roughly 3x the size if accuracy warrants it.
MODEL_RELEASE = "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{MODEL_RELEASE}.tar.bz2"
)
#: Approximate compressed download, for progress messages.
MODEL_DOWNLOAD_MB = 292

#: Files sherpa-onnx needs, as named inside the archive.
MODEL_FILES = ("model.int8.onnx", "tokens.txt")

#: Optional Silero VAD (~0.6 MB) — gives per-utterance segments and keeps each
#: decode under the model's training window. Same asset the mobile app uses.
VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "silero_vad.onnx"
)
VAD_FILENAME = "silero_vad.onnx"

#: Omnilingual models were trained on <=40 s utterances. Cap every decode well
#: under that; VAD honours it via ``max_speech_duration`` and the no-VAD path
#: falls back to fixed windows of this length.
MAX_SEGMENT_SECONDS = 30.0

#: Default language tag stamped on results. Cebuano is the first regional
#: language the app exposes; the model itself is not language-specific.
DEFAULT_LANGUAGE = "ceb"

_INSTALL_HINT = (
    "Install it with:\n"
    "    pip install sherpa-onnx\n"
    "or reinstall the project with the desktop extra:\n"
    "    pip install -e \".[desktop]\""
)

_DOWNLOAD_HINT = (
    f"Download the model (~{MODEL_DOWNLOAD_MB} MB, one time):\n"
    "    meeting-notes transcribe <file> --language ceb   # downloads on first use\n"
    "or fetch it by hand from\n"
    f"    {MODEL_URL}\n"
    "and unpack model.int8.onnx and tokens.txt into the models directory."
)


class OmnilingualBackend(TranscriptionBackend):
    """Offline regional-language speech-to-text via sherpa-onnx.

    Args:
        language: ISO code to stamp on the result. The model is
            language-agnostic; this is what the UI and the notes step see.
        model_dir: Directory holding the ONNX files. Defaults to
            ``<models_dir>/omnilingual``.
        num_threads: ONNX Runtime intra-op threads.
        use_vad: Segment with Silero VAD when its model is present or can be
            fetched. Without it audio is cut into fixed 30 s windows.
        auto_download: Fetch the model on first use when it is missing.
            ``False`` makes a missing model a hard error, for tests and for
            installs that must never touch the network.
    """

    name = "omnilingual-ctc"

    def __init__(
        self,
        language: str = DEFAULT_LANGUAGE,
        model_dir: Optional[str] = None,
        num_threads: int = 4,
        use_vad: bool = True,
        auto_download: bool = True,
    ) -> None:
        self.language = language
        self.model_dir = Path(model_dir) if model_dir else models_dir() / MODEL_SUBDIR
        self.num_threads = num_threads
        self.use_vad = use_vad
        self.auto_download = auto_download
        self._recognizer = None
        self._vad = None
        self._vad_window = 512
        self._vad_failed = False

    # ------------------------------------------------------------------
    # TranscriptionBackend
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True when sherpa-onnx imports and the model is present or fetchable.

        Deliberately does not download: this is called from diagnostics and
        the desktop startup check, which must stay fast and offline.
        """
        try:
            import sherpa_onnx  # noqa: F401, PLC0415
        except ImportError:
            return False
        return self.is_downloaded() or self.auto_download

    def is_downloaded(self) -> bool:
        """True when both model files are on disk."""
        return all((self.model_dir / f).exists() for f in MODEL_FILES)

    def missing_files(self) -> list[str]:
        return [f for f in MODEL_FILES if not (self.model_dir / f).exists()]

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

        segments: list[TranscriptSegment] = []
        for start_sample, region in self._speech_regions(audio):
            text = clean_latin_text(self._decode(recognizer, region))
            if not text.strip():
                continue
            start = start_sample / DEFAULT_SAMPLE_RATE
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=start + len(region) / DEFAULT_SAMPLE_RATE,
                    text=text.strip(),
                    lang=self.language,
                    # Greedy CTC in sherpa-onnx exposes no per-segment score.
                    confidence=None,
                )
            )

        return TranscriptionResult(
            text=" ".join(s.text for s in segments).strip(),
            language=self.language,
            language_name=language_name(self.language),
            segments=segments,
            backend=self.name,
        )

    # ------------------------------------------------------------------
    # Model files
    # ------------------------------------------------------------------

    def ensure_models(self) -> None:
        """Make sure the model files exist, downloading them when allowed.

        Raises:
            BackendUnavailableError: When files are missing and downloading
                is disabled or fails.
        """
        if self.is_downloaded():
            return
        if not self.auto_download:
            raise BackendUnavailableError(
                self.name,
                f"model files missing from {self.model_dir} "
                f"({', '.join(self.missing_files())})",
                _DOWNLOAD_HINT,
            )

        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            archive = self.model_dir / f"{MODEL_RELEASE}.tar.bz2.partial"
            logger.info(
                "Downloading Omnilingual ASR (~%d MB) to %s",
                MODEL_DOWNLOAD_MB,
                self.model_dir,
            )
            urllib.request.urlretrieve(MODEL_URL, archive, _progress)  # noqa: S310
            self._extract(archive)
            archive.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 — any failure is "unavailable"
            raise BackendUnavailableError(
                self.name,
                f"could not download the model: {exc}",
                _DOWNLOAD_HINT,
            ) from exc

        if not self.is_downloaded():
            raise BackendUnavailableError(
                self.name,
                f"archive did not contain {', '.join(self.missing_files())}",
                _DOWNLOAD_HINT,
            )

    def _extract(self, archive: Path) -> None:
        """Pull just the model files out of the release tarball, flattened."""
        with tarfile.open(archive, "r:bz2") as tar:
            for member in tar.getmembers():
                base = Path(member.name).name
                if base not in MODEL_FILES or not member.isfile():
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                dest = self.model_dir / base
                with source, open(dest, "wb") as out:
                    while True:
                        chunk = source.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)

    def _ensure_vad_model(self) -> Optional[Path]:
        """Local path of the Silero VAD model, fetching it once if allowed."""
        path = self.model_dir / VAD_FILENAME
        if path.exists():
            return path
        if not self.auto_download:
            return None
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial")
            urllib.request.urlretrieve(VAD_URL, partial)  # noqa: S310
            partial.replace(path)
            return path
        except Exception:  # noqa: BLE001 — VAD is optional
            logger.warning(
                "could not download Silero VAD; using fixed windows", exc_info=True
            )
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_sherpa(self):
        try:
            import sherpa_onnx  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailableError(
                self.name, "the sherpa-onnx package is not installed", _INSTALL_HINT
            ) from exc
        return sherpa_onnx

    def _load_recognizer(self):
        """Lazy-load the sherpa-onnx offline recognizer."""
        if self._recognizer is not None:
            return self._recognizer

        sherpa_onnx = self._require_sherpa()
        self.ensure_models()

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=str(self.model_dir / MODEL_FILES[0]),
            tokens=str(self.model_dir / MODEL_FILES[1]),
            num_threads=self.num_threads,
            decoding_method="greedy_search",
        )
        return self._recognizer

    def _load_vad(self):
        """Lazy-load Silero VAD; None when disabled or unavailable."""
        if self._vad is not None:
            return self._vad
        if not self.use_vad or self._vad_failed:
            return None

        vad_path = self._ensure_vad_model()
        if vad_path is None:
            self._vad_failed = True
            return None

        sherpa_onnx = self._require_sherpa()
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(vad_path)
        config.silero_vad.min_silence_duration = 0.25
        config.silero_vad.max_speech_duration = MAX_SEGMENT_SECONDS
        config.sample_rate = DEFAULT_SAMPLE_RATE

        # The buffer only has to hold the longest single speech region plus
        # some slack; finished regions are drained as the audio streams in.
        self._vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=MAX_SEGMENT_SECONDS * 2
        )
        self._vad_window = int(config.silero_vad.window_size) or 512
        return self._vad

    def _speech_regions(self, audio) -> list[tuple[int, "object"]]:
        """Split audio into (start_sample, samples) regions to decode.

        VAD when available; otherwise fixed windows, because a single decode
        over a long meeting would exceed the model's training length and eat
        memory.
        """
        vad = self._load_vad()
        if vad is None:
            return _fixed_windows(audio)
        regions = drain_vad(vad, audio, self._vad_window)
        return regions or _fixed_windows(audio)

    def _decode(self, recognizer, samples) -> str:
        stream = recognizer.create_stream()
        stream.accept_waveform(DEFAULT_SAMPLE_RATE, samples)
        recognizer.decode_stream(stream)
        return stream.result.text


#: Highest code point kept by :func:`clean_latin_text` — covers ASCII,
#: Latin-1 and the Latin Extended blocks (ñ, accented loanwords), nothing more.
_LATIN_MAX_CODEPOINT = 0x024F


def clean_latin_text(text: str) -> str:
    """Drop non-Latin characters a language-agnostic CTC model may emit.

    With no language token, the 1,600-language model occasionally decodes a
    short or noisy stretch of Bisaya as Arabic or other scripts — whole
    segments of ``ل`` and ``مردينني`` showed up on a 70 s meeting clip.
    **[MEASURED 2026-09-07]** Every language this backend serves today is
    written in Latin script, so anything else is noise, not content. Keeps
    whitespace and punctuation (including the General Punctuation block, for
    dashes and quotes), collapses the gaps left behind.

    A segment where most letters were foreign script is discarded outright —
    the few Latin letters left over (``n o an n``) are debris from the same
    mis-decode, not words.
    """
    latin_letters = sum(1 for ch in text if ch.isalpha() and ord(ch) <= _LATIN_MAX_CODEPOINT)
    other_letters = sum(1 for ch in text if ch.isalpha() and ord(ch) > _LATIN_MAX_CODEPOINT)
    if other_letters and other_letters >= latin_letters:
        return ""

    kept = (
        ch if (_is_latin_or_punct(ch) or ch.isspace()) else " "
        for ch in text
    )
    return " ".join("".join(kept).split())


def _is_latin_or_punct(ch: str) -> bool:
    code = ord(ch)
    return code <= _LATIN_MAX_CODEPOINT or 0x2000 <= code <= 0x206F


def drain_vad(vad, audio, window_size: int = 512) -> list[tuple[int, "object"]]:
    """Run a sherpa-onnx VAD over ``audio`` and return (start_sample, samples).

    Silero VAD is a streaming model: sherpa-onnx scores exactly one
    ``window_size`` block per call, so the audio has to be pushed in blocks of
    that size. Handing it the whole file in one ``accept_waveform`` looks like
    it works but scores only the tail — on a 70 s meeting recording it
    returned a single 0.3 s region at the very end. **[MEASURED 2026-09-07]**
    Finished regions are drained as we go so the internal buffer never fills.
    """
    import numpy as np  # noqa: PLC0415

    audio = np.asarray(audio, dtype=np.float32)
    regions: list[tuple[int, object]] = []

    def _drain() -> None:
        while not vad.empty():
            segment = vad.front
            regions.append((segment.start, segment.samples))
            vad.pop()

    vad.reset()
    for offset in range(0, len(audio), window_size):
        block = audio[offset : offset + window_size]
        if len(block) < window_size:
            block = np.pad(block, (0, window_size - len(block)))
        vad.accept_waveform(block)
        _drain()
    vad.flush()
    _drain()
    return regions


def _fixed_windows(audio) -> list[tuple[int, "object"]]:
    """Cut audio into consecutive MAX_SEGMENT_SECONDS windows."""
    step = int(MAX_SEGMENT_SECONDS * DEFAULT_SAMPLE_RATE)
    if len(audio) == 0:
        return []
    return [(start, audio[start : start + step]) for start in range(0, len(audio), step)]


def _progress(block: int, block_size: int, total: int) -> None:
    """urlretrieve hook: log every ~50 MB so a 290 MB fetch is visibly alive."""
    done = block * block_size
    if total > 0 and block > 0 and done % (50 * 1024 * 1024) < block_size:
        logger.info("  %d / %d MB", done >> 20, total >> 20)


def default_model_dir() -> Path:
    """Where the Omnilingual ONNX files are expected to live."""
    return models_dir() / MODEL_SUBDIR
