"""Tests for the audio module."""

from __future__ import annotations

import os
import struct
import tempfile
import wave

import numpy as np
import pytest

from meeting_notes.audio import _save_wav, load_audio_file


# ---------------------------------------------------------------------------
# _save_wav
# ---------------------------------------------------------------------------


def test_save_wav_creates_file():
    audio = np.zeros(16000, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name

    try:
        result = _save_wav(audio, sample_rate=16000, channels=1, output_path=path)
        assert os.path.exists(result)
        assert result == path
    finally:
        os.unlink(path)


def test_save_wav_auto_tmp_path():
    audio = np.zeros(16000, dtype=np.float32)
    path = _save_wav(audio, sample_rate=16000, channels=1)
    try:
        assert os.path.exists(path)
        assert path.endswith(".wav")
    finally:
        os.unlink(path)


def test_save_wav_correct_headers():
    audio = np.zeros(8000, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name

    try:
        _save_wav(audio, sample_rate=16000, channels=1, output_path=path)
        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2  # 16-bit PCM
            assert wf.getnframes() == 8000
    finally:
        os.unlink(path)


def test_save_wav_clips_values():
    """Values outside [-1, 1] should be clipped to 16-bit range."""
    audio = np.array([2.0, -3.0], dtype=np.float32)
    path = _save_wav(audio, sample_rate=16000, channels=1)
    try:
        with wave.open(path, "rb") as wf:
            raw = wf.readframes(2)
        samples = struct.unpack("<2h", raw)
        assert samples[0] == 32767   # clipped +
        assert samples[1] == -32768  # clipped −
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# load_audio_file
# ---------------------------------------------------------------------------


def _make_wav(path: str, n_frames: int = 16000, sample_rate: int = 16000) -> None:
    """Write a minimal silent WAV file for testing."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00" * n_frames * 2)


def test_load_audio_file_returns_float32_array():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        _make_wav(path)
        audio, sr = load_audio_file(path)
        assert audio.dtype == np.float32
        assert sr == 16000
        assert len(audio) == 16000
    finally:
        os.unlink(path)


def test_load_audio_file_stereo_mixed_to_mono():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        # Write a 2-channel WAV
        n = 1000
        with wave.open(path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * n * 2 * 2)  # 2 channels × 2 bytes × n frames
        audio, sr = load_audio_file(path)
        assert audio.ndim == 1  # mono output
        assert len(audio) == n
    finally:
        os.unlink(path)


def test_load_audio_file_not_found():
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        load_audio_file("/nonexistent/path/audio.wav")
