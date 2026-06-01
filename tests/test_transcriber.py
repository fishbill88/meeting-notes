"""Tests for the transcription module."""

from __future__ import annotations

import os
import tempfile
import wave
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# Transcriber – unit tests with mocked Whisper
# ---------------------------------------------------------------------------


def _make_wav(path: str, n_frames: int = 8000) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * n_frames * 2)


FAKE_WHISPER_RESULT = {
    "text": "This is a test transcript.",
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 2.5, "text": "This is a test"},
        {"start": 2.5, "end": 4.0, "text": " transcript."},
    ],
}


@patch("whisper.load_model")
def test_transcribe_file_success(mock_load_model):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = FAKE_WHISPER_RESULT
    mock_load_model.return_value = mock_model

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        _make_wav(path)
        t = Transcriber(model_size="base")
        result = t.transcribe_file(path)

        assert result.text == "This is a test transcript."
        assert result.language == "en"
        assert len(result.segments) == 2
    finally:
        os.unlink(path)


@patch("whisper.load_model")
def test_transcribe_file_with_language_hint(mock_load_model):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {**FAKE_WHISPER_RESULT, "language": "tl"}
    mock_load_model.return_value = mock_model

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        _make_wav(path)
        t = Transcriber(model_size="base", language="tl")
        result = t.transcribe_file(path)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("language") == "tl"
        assert result.language == "tl"
    finally:
        os.unlink(path)


def test_transcribe_file_not_found():
    t = Transcriber(model_size="base")
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        t.transcribe_file("/nonexistent/audio.wav")


def test_invalid_model_size_raises():
    with pytest.raises(ValueError, match="Invalid model_size"):
        Transcriber(model_size="mega")


def test_model_lazy_loaded():
    t = Transcriber(model_size="base")
    assert t._model is None  # not loaded yet


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
