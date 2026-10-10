"""Shared test fixtures.

No test may download a model or touch audio hardware. Engines are faked at the
module boundary so the suite runs offline and fast.
"""

from __future__ import annotations

import sys
import types
import wave
from typing import Optional
from unittest.mock import MagicMock

import pytest

from meeting_notes.backends.base import (
    TranscriptionBackend,
    TranscriptionResult,
    TranscriptSegment,
)


class FakeBackend(TranscriptionBackend):
    """A backend that returns a canned result and records how it was called."""

    name = "fake"

    def __init__(
        self,
        text: str = "This is a test transcript.",
        language: str = "en",
        available: bool = True,
    ) -> None:
        self.text = text
        self.language = language
        self._available = available
        self.calls: list[tuple[str, object]] = []

    def is_available(self) -> bool:
        return self._available

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        self.calls.append(("transcribe_file", audio_path))
        return self._result()

    def transcribe_array(self, audio_array, sample_rate: int = 16000):
        self.calls.append(("transcribe_array", sample_rate))
        return self._result()

    def _result(self) -> TranscriptionResult:
        return TranscriptionResult(
            text=self.text,
            language=self.language,
            language_name=self.language,
            segments=[
                TranscriptSegment(
                    start=0.0,
                    end=2.0,
                    text=self.text,
                    lang=self.language,
                    confidence=0.9,
                )
            ],
            backend=self.name,
        )


def _forbid_download(*args, **kwargs):
    raise RuntimeError("Tests must not download models")


@pytest.fixture(autouse=True)
def offline_diarization(monkeypatch, tmp_path):
    """Enforce the no-download rule for the diarization models.

    Points the models dir at an empty temp dir, makes any download attempt
    raise (so diarization degrades to "no speakers", its designed failure
    mode), and resets the module's process-wide caches between tests.
    """
    import urllib.request

    import meeting_notes.diarize as diarize
    from meeting_notes.config import MODELS_DIR_ENV

    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path / "models"))
    monkeypatch.setattr(urllib.request, "urlretrieve", _forbid_download)
    monkeypatch.setattr(diarize, "_diarizer", None)
    monkeypatch.setattr(diarize, "_diarizer_failed", False)
    monkeypatch.setattr(diarize, "_denoiser", None)
    monkeypatch.setattr(diarize, "_denoiser_failed", False)
    monkeypatch.setattr(diarize, "_embedder", None)
    monkeypatch.setattr(diarize, "_embedder_failed", False)


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def wav_file(tmp_path) -> str:
    """A tiny valid 16 kHz mono WAV file."""
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 16000 * 2)  # 1 second of silence
    return str(path)


def make_fake_faster_whisper(
    segments: Optional[list] = None,
    language: str = "en",
) -> types.ModuleType:
    """Build a stand-in ``faster_whisper`` module.

    faster-whisper is a heavy optional dependency; injecting a fake module
    lets the backend's parsing logic be tested without installing it.
    """
    if segments is None:
        segments = [
            _segment(0.0, 2.5, "This is a test", -0.1),
            _segment(2.5, 4.0, " transcript.", -0.2),
        ]

    info = MagicMock()
    info.language = language
    info.language_probability = 0.99

    model = MagicMock()
    model.transcribe.return_value = (iter(segments), info)

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = MagicMock(return_value=model)
    module._model = model  # exposed so tests can assert on transcribe kwargs
    return module


def _segment(start: float, end: float, text: str, avg_logprob: float):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    seg.avg_logprob = avg_logprob
    seg.no_speech_prob = 0.01
    return seg


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    """Install a fake ``faster_whisper`` into ``sys.modules`` for one test."""
    module = make_fake_faster_whisper()
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return module
