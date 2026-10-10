"""Tests for the transcription facade and shared result types."""

from __future__ import annotations

import pytest

from meeting_notes.transcriber import (
    TranscriptSegment,
    TranscriptionResult,
    Transcriber,
    _language_name,
    _resample,
)


# ---------------------------------------------------------------------------
# TranscriptSegment
# ---------------------------------------------------------------------------


def test_segment_str_format():
    seg = TranscriptSegment(start=65.0, end=70.5, text="Hello world.")
    result = str(seg)
    assert "01:05" in result
    assert "01:10" in result
    assert "Hello world." in result


def test_segment_defaults_to_english_and_no_confidence():
    seg = TranscriptSegment(start=0.0, end=1.0, text="Hi")
    assert seg.lang == "en"
    assert seg.confidence is None


def test_segment_carries_language_and_confidence():
    seg = TranscriptSegment(0.0, 1.0, "Kumusta", lang="tl", confidence=0.82)
    assert seg.lang == "tl"
    assert seg.confidence == 0.82


# ---------------------------------------------------------------------------
# TranscriptionResult
# ---------------------------------------------------------------------------


def test_timed_transcript_joins_segments():
    result = TranscriptionResult(
        text="Hello world.",
        language="en",
        language_name="English",
        segments=[
            TranscriptSegment(0.0, 2.0, "Hello"),
            TranscriptSegment(2.0, 4.0, "world."),
        ],
    )
    timed = result.timed_transcript
    assert "Hello" in timed
    assert "world." in timed
    assert "\n" in timed


def test_timed_transcript_empty_segments():
    result = TranscriptionResult(
        text="Hello.", language="en", language_name="English", segments=[]
    )
    assert result.timed_transcript == ""


def test_result_records_backend():
    result = TranscriptionResult(
        text="Hi", language="en", language_name="English", backend="faster-whisper"
    )
    assert result.backend == "faster-whisper"


# ---------------------------------------------------------------------------
# Transcriber – now a thin facade over a backend
# ---------------------------------------------------------------------------


def test_transcribe_file_delegates_to_backend(fake_backend, wav_file):
    t = Transcriber(model_size="base", backend=fake_backend)
    result = t.transcribe_file(wav_file)

    assert result.text == "This is a test transcript."
    assert result.language == "en"
    assert len(result.segments) == 1
    assert fake_backend.calls == [("transcribe_file", wav_file)]


def test_transcribe_array_delegates_to_backend(fake_backend):
    import numpy as np

    t = Transcriber(backend=fake_backend)
    t.transcribe_array(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    assert fake_backend.calls == [("transcribe_array", 16000)]


def test_transcribe_file_not_found(fake_backend):
    from meeting_notes.backends.base import _require_file

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        _require_file("/nonexistent/audio.wav")


def test_invalid_model_size_raises():
    with pytest.raises(ValueError, match="Invalid model_size"):
        Transcriber(model_size="mega")


def test_backend_lazily_constructed():
    t = Transcriber(model_size="base")
    assert t._backend is None  # not built until first use


def test_large_v3_turbo_is_a_valid_model_size():
    # The default English engine — guards against the choices list regressing.
    Transcriber(model_size="large-v3-turbo")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def test_language_name_known():
    assert _language_name("en") == "English"
    assert _language_name("tl") == "Filipino / Tagalog"


def test_language_name_unknown_returns_code():
    assert _language_name("xx") == "xx"


def test_resample_increases_length():
    import numpy as np

    audio = np.zeros(8000, dtype=np.float32)
    resampled = _resample(audio, orig_rate=8000, target_rate=16000)
    assert len(resampled) == 16000


def test_resample_decreases_length():
    import numpy as np

    audio = np.zeros(16000, dtype=np.float32)
    resampled = _resample(audio, orig_rate=16000, target_rate=8000)
    assert len(resampled) == 8000


def test_resample_noop_when_rates_match():
    import numpy as np

    audio = np.zeros(16000, dtype=np.float32)
    assert _resample(audio, 16000, 16000) is audio
