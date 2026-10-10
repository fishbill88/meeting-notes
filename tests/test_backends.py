"""Tests for the individual transcription backends."""

from __future__ import annotations

import sys

import pytest

from meeting_notes.backends.base import (
    BackendUnavailableError,
    logprob_to_confidence,
)
from meeting_notes.backends.nemo import CTC_FILES, TRANSDUCER_FILES, NemoTagalogBackend
from meeting_notes.backends.omnilingual import MODEL_FILES, OmnilingualBackend
from meeting_notes.backends.whisper import FasterWhisperBackend, _resolve_compute_type


# ---------------------------------------------------------------------------
# Confidence mapping
# ---------------------------------------------------------------------------


def test_logprob_to_confidence_maps_zero_to_one():
    assert logprob_to_confidence(0.0) == 1.0


def test_logprob_to_confidence_is_monotonic():
    assert logprob_to_confidence(-0.1) > logprob_to_confidence(-1.0)


def test_logprob_to_confidence_stays_in_range():
    for logprob in (-100.0, -5.0, -0.5, 0.0):
        value = logprob_to_confidence(logprob)
        assert 0.0 <= value <= 1.0


def test_logprob_to_confidence_passes_through_none():
    assert logprob_to_confidence(None) is None


# ---------------------------------------------------------------------------
# faster-whisper backend
# ---------------------------------------------------------------------------


def test_compute_type_auto_picks_int8_on_cpu():
    assert _resolve_compute_type("auto", "cpu") == "int8"


def test_compute_type_auto_picks_float16_on_cuda():
    assert _resolve_compute_type("auto", "cuda") == "float16"


def test_compute_type_explicit_is_respected():
    assert _resolve_compute_type("float32", "cpu") == "float32"


def test_whisper_backend_parses_segments(fake_faster_whisper, wav_file):
    backend = FasterWhisperBackend(model_size="base")
    result = backend.transcribe_file(wav_file)

    assert result.backend == "faster-whisper"
    assert result.language == "en"
    assert len(result.segments) == 2
    assert result.text == "This is a test transcript."


def test_whisper_backend_populates_confidence(fake_faster_whisper, wav_file):
    backend = FasterWhisperBackend(model_size="base")
    result = backend.transcribe_file(wav_file)

    # avg_logprob of -0.1 and -0.2 must survive as usable confidences; the
    # Taglish merge depends on this being populated.
    assert all(s.confidence is not None for s in result.segments)
    assert result.segments[0].confidence > result.segments[1].confidence


def test_whisper_backend_tags_segments_with_language(fake_faster_whisper, wav_file):
    backend = FasterWhisperBackend(model_size="base", language="tl")
    result = backend.transcribe_file(wav_file)
    assert all(s.lang == "en" for s in result.segments)  # from the fake's info


def test_whisper_backend_forwards_language_and_prompt(fake_faster_whisper, wav_file):
    backend = FasterWhisperBackend(
        model_size="base", language="tl", initial_prompt="lagnat, ubo"
    )
    backend.transcribe_file(wav_file)

    kwargs = fake_faster_whisper._model.transcribe.call_args[1]
    assert kwargs["language"] == "tl"
    assert kwargs["initial_prompt"] == "lagnat, ubo"


def test_whisper_backend_unavailable_without_package(monkeypatch, wav_file):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    backend = FasterWhisperBackend(model_size="base")

    with pytest.raises(BackendUnavailableError, match="faster-whisper"):
        backend.transcribe_file(wav_file)


def test_whisper_backend_is_available_reflects_import(monkeypatch):
    backend = FasterWhisperBackend()
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    assert backend.is_available() is False


# ---------------------------------------------------------------------------
# NeMo Tagalog backend
# ---------------------------------------------------------------------------


def test_nemo_detects_no_model_in_empty_dir(tmp_path):
    backend = NemoTagalogBackend(model_dir=str(tmp_path))
    assert backend.model_type() is None
    assert backend.is_available() is False


def test_nemo_detects_ctc_export(tmp_path):
    for name in CTC_FILES:
        (tmp_path / name).write_text("x")
    backend = NemoTagalogBackend(model_dir=str(tmp_path))
    assert backend.model_type() == "ctc"


def test_nemo_detects_transducer_export(tmp_path):
    for name in TRANSDUCER_FILES:
        (tmp_path / name).write_text("x")
    backend = NemoTagalogBackend(model_dir=str(tmp_path))
    assert backend.model_type() == "transducer"


def test_nemo_transducer_preferred_when_both_present(tmp_path):
    for name in set(CTC_FILES) | set(TRANSDUCER_FILES):
        (tmp_path / name).write_text("x")
    backend = NemoTagalogBackend(model_dir=str(tmp_path))
    assert backend.model_type() == "transducer"


def test_nemo_missing_files_reports_ctc_set(tmp_path):
    backend = NemoTagalogBackend(model_dir=str(tmp_path))
    assert set(backend.missing_files()) == set(CTC_FILES)


def test_nemo_missing_files_empty_when_exported(tmp_path):
    for name in CTC_FILES:
        (tmp_path / name).write_text("x")
    backend = NemoTagalogBackend(model_dir=str(tmp_path))
    assert backend.missing_files() == []


def test_nemo_raises_with_export_hint_when_unexported(tmp_path, monkeypatch):
    # Pretend sherpa-onnx is installed so we exercise the missing-files path.
    import types

    monkeypatch.setitem(sys.modules, "sherpa_onnx", types.ModuleType("sherpa_onnx"))
    backend = NemoTagalogBackend(model_dir=str(tmp_path))

    with pytest.raises(BackendUnavailableError, match="export_nemo_to_onnx"):
        backend.warmup()


def test_nemo_reports_language_as_tagalog(tmp_path):
    backend = NemoTagalogBackend(model_dir=str(tmp_path))
    assert backend.language == "tl"


# ---------------------------------------------------------------------------
# Omnilingual CTC backend (Cebuano / regional languages)
# ---------------------------------------------------------------------------


def _touch_omnilingual_model(model_dir):
    for name in MODEL_FILES:
        (model_dir / name).write_bytes(b"stub")


def _install_fake_sherpa(monkeypatch, text="maayong buntag sa tanan"):
    """A stand-in ``sherpa_onnx`` whose CTC recognizer returns canned text."""
    import types
    from unittest.mock import MagicMock

    stream = MagicMock()
    stream.result.text = text
    recognizer = MagicMock()
    recognizer.create_stream.return_value = stream

    module = types.ModuleType("sherpa_onnx")
    module.OfflineRecognizer = MagicMock()
    module.OfflineRecognizer.from_omnilingual_asr_ctc = MagicMock(return_value=recognizer)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    return module


def test_omnilingual_reports_missing_files_in_empty_dir(tmp_path):
    backend = OmnilingualBackend(model_dir=str(tmp_path), auto_download=False)
    assert backend.is_downloaded() is False
    assert set(backend.missing_files()) == set(MODEL_FILES)


def test_omnilingual_unavailable_without_sherpa(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    assert OmnilingualBackend(model_dir=str(tmp_path)).is_available() is False


def test_omnilingual_available_when_download_allowed(monkeypatch, tmp_path):
    # Availability must not trigger a download — it is called at app startup.
    _install_fake_sherpa(monkeypatch)
    backend = OmnilingualBackend(model_dir=str(tmp_path), auto_download=True)
    assert backend.is_available() is True
    assert backend.is_downloaded() is False


def test_omnilingual_raises_with_download_hint_when_offline(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch)
    backend = OmnilingualBackend(model_dir=str(tmp_path), auto_download=False)
    with pytest.raises(BackendUnavailableError, match="--language ceb"):
        backend.warmup()


def test_omnilingual_download_failure_is_unavailable_not_a_crash(monkeypatch, tmp_path):
    # conftest forbids urlretrieve; the backend must translate that into the
    # shared error type the CLI and sidecar already render.
    _install_fake_sherpa(monkeypatch)
    backend = OmnilingualBackend(model_dir=str(tmp_path), auto_download=True)
    with pytest.raises(BackendUnavailableError, match="could not download"):
        backend.warmup()


def test_omnilingual_transcribes_and_stamps_language(monkeypatch, tmp_path, wav_file):
    module = _install_fake_sherpa(monkeypatch)
    _touch_omnilingual_model(tmp_path)
    backend = OmnilingualBackend(model_dir=str(tmp_path), use_vad=False, auto_download=False)

    result = backend.transcribe_file(wav_file)

    assert result.backend == "omnilingual-ctc"
    assert result.language == "ceb"
    assert result.language_name == "Cebuano / Bisaya"
    assert result.text == "maayong buntag sa tanan"
    assert [s.lang for s in result.segments] == ["ceb"]
    assert result.segments[0].confidence is None

    kwargs = module.OfflineRecognizer.from_omnilingual_asr_ctc.call_args.kwargs
    assert kwargs["model"].endswith("model.int8.onnx")
    assert kwargs["tokens"].endswith("tokens.txt")


def test_omnilingual_cuts_long_audio_into_windows_without_vad(monkeypatch, tmp_path):
    import numpy as np

    _install_fake_sherpa(monkeypatch)
    _touch_omnilingual_model(tmp_path)
    backend = OmnilingualBackend(model_dir=str(tmp_path), use_vad=False, auto_download=False)

    seventy_seconds = np.zeros(70 * 16000, dtype=np.float32)
    result = backend.transcribe_array(seventy_seconds)

    # 30 s + 30 s + 10 s — never one 70 s decode, which exceeds the model's
    # training window.
    assert [round(s.end - s.start) for s in result.segments] == [30, 30, 10]
    assert [s.start for s in result.segments] == [0.0, 30.0, 60.0]


def test_omnilingual_default_dir_follows_models_dir(monkeypatch, tmp_path):
    from meeting_notes.config import MODELS_DIR_ENV

    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path))
    assert OmnilingualBackend().model_dir == tmp_path / "omnilingual"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("para dili maguba ang layout", "para dili maguba ang layout"),
        ("dpل atana  staging بن usr", "dp atana staging usr"),  # script drift, seen live
        ("مردينني نيا", ""),  # a whole segment of drift disappears
        ("كn oقين anتناn", ""),  # mostly drift: the Latin debris goes with it
        ("señor Peña — ok", "señor Peña — ok"),  # Latin-1 and punctuation survive
    ],
)
def test_clean_latin_text_strips_script_drift(raw, expected):
    from meeting_notes.backends.omnilingual import clean_latin_text

    assert clean_latin_text(raw) == expected


def test_omnilingual_drops_segments_that_were_only_script_drift(monkeypatch, tmp_path, wav_file):
    _install_fake_sherpa(monkeypatch, text="مردينني نيا")
    _touch_omnilingual_model(tmp_path)
    backend = OmnilingualBackend(model_dir=str(tmp_path), use_vad=False, auto_download=False)

    result = backend.transcribe_file(wav_file)

    assert result.segments == []
    assert result.text == ""


class _FakeVad:
    """Mimics sherpa-onnx's VoiceActivityDetector: one block scored per call.

    Emits a "speech region" every ``every`` blocks so the drain loop has
    something to collect mid-stream, and records every block it was fed.
    """

    def __init__(self, every: int = 10):
        self.blocks: list[int] = []
        self._queue: list = []
        self._every = every
        self.flushed = False

    def reset(self):
        self.blocks.clear()

    def accept_waveform(self, block):
        self.blocks.append(len(block))
        if len(self.blocks) % self._every == 0:
            from unittest.mock import MagicMock

            seg = MagicMock()
            seg.start = (len(self.blocks) - self._every) * 512
            seg.samples = block
            self._queue.append(seg)

    def flush(self):
        self.flushed = True

    def empty(self):
        return not self._queue

    @property
    def front(self):
        return self._queue[0]

    def pop(self):
        self._queue.pop(0)


def test_drain_vad_feeds_window_sized_blocks_and_collects_regions():
    # Regression: pushing a whole 70 s file into Silero VAD in one call
    # scored only the tail (a single 0.3 s region), so the transcript came out
    # empty. The VAD must see the audio in window-sized blocks.
    import numpy as np

    from meeting_notes.backends.omnilingual import drain_vad

    vad = _FakeVad(every=10)
    audio = np.zeros(512 * 45 + 100, dtype=np.float32)  # not a whole number of blocks

    regions = drain_vad(vad, audio, window_size=512)

    assert vad.blocks and set(vad.blocks) == {512}  # never the whole file, tail padded
    assert len(vad.blocks) == 46
    assert vad.flushed
    assert [start for start, _ in regions] == [0, 5120, 10240, 15360]
