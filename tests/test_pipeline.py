"""Tests for the shared transcribe → summarize pipeline."""

from __future__ import annotations

import pytest

from meeting_notes.config import AppConfig, LanguageMode
from meeting_notes.pipeline import run_pipeline


@pytest.fixture
def routed(monkeypatch, fake_backend):
    """Route every request to the fake backend."""
    monkeypatch.setattr(
        "meeting_notes.backends.router.resolve_backend",
        lambda config: fake_backend,
    )
    return fake_backend


def test_pipeline_returns_transcript_and_notes(routed, wav_file):
    config = AppConfig(use_ollama=False)
    result = run_pipeline(wav_file, config)

    assert result.transcript.text == "This is a test transcript."
    assert result.notes.title
    assert result.notes.summary


def test_pipeline_exposes_which_engine_ran(routed, wav_file):
    result = run_pipeline(wav_file, AppConfig(use_ollama=False))
    assert result.backend == "fake"


def test_pipeline_reports_progress(routed, wav_file):
    stages = []
    run_pipeline(
        wav_file,
        AppConfig(use_ollama=False),
        on_progress=lambda stage, message: stages.append(stage),
    )
    assert stages == ["transcribe", "diarize", "summarize"]


def test_pipeline_skips_diarization_when_disabled(routed, wav_file):
    stages = []
    run_pipeline(
        wav_file,
        AppConfig(use_ollama=False, diarize=False),
        on_progress=lambda stage, message: stages.append(stage),
    )
    assert stages == ["transcribe", "summarize"]


def test_pipeline_labels_speakers_when_diarization_works(monkeypatch, wav_file):
    from meeting_notes.backends.base import TranscriptionResult, TranscriptSegment
    from meeting_notes.diarize import SpeakerSpan

    class TwoSegments:
        name = "fake"

        def transcribe_file(self, path):
            return TranscriptionResult(
                text="Hello. Hi there.",
                language="en",
                language_name="English",
                segments=[
                    TranscriptSegment(start=0.0, end=1.0, text="Hello."),
                    TranscriptSegment(start=1.2, end=2.0, text="Hi there."),
                ],
                backend=self.name,
            )

    monkeypatch.setattr(
        "meeting_notes.backends.router.resolve_backend",
        lambda config: TwoSegments(),
    )
    monkeypatch.setattr(
        "meeting_notes.diarize._diarize",
        lambda path: (
            [
                SpeakerSpan(start=0.0, end=1.1, speaker=0),
                SpeakerSpan(start=1.1, end=2.0, speaker=1),
            ],
            [0.0] * 32000,
            16000,
        ),
    )
    result = run_pipeline(wav_file, AppConfig(use_ollama=False))
    assert [s.speaker for s in result.transcript.segments] == [
        "Person 1",
        "Person 2",
    ]


def test_pipeline_survives_unavailable_diarization(routed, wav_file):
    # The autouse offline_diarization fixture makes model downloads fail, so
    # this exercises the real degradation path: transcript intact, no labels.
    result = run_pipeline(wav_file, AppConfig(use_ollama=False))
    assert result.transcript.text == "This is a test transcript."
    assert result.transcript.segments[0].speaker is None


def test_pipeline_works_without_progress_callback(routed, wav_file):
    assert run_pipeline(wav_file, AppConfig(use_ollama=False)) is not None


def test_pipeline_timed_transcript_falls_back_to_text(monkeypatch, wav_file):
    from meeting_notes.backends.base import TranscriptionResult

    class NoSegments:
        name = "no-segments"

        def transcribe_file(self, path):
            return TranscriptionResult(
                text="Plain text only.",
                language="en",
                language_name="English",
                segments=[],
                backend=self.name,
            )

    monkeypatch.setattr(
        "meeting_notes.backends.router.resolve_backend",
        lambda config: NoSegments(),
    )
    result = run_pipeline(wav_file, AppConfig(use_ollama=False))
    assert result.timed_transcript == "Plain text only."


def test_pipeline_passes_detected_language_to_summarizer(monkeypatch, wav_file):
    from tests.conftest import FakeBackend

    monkeypatch.setattr(
        "meeting_notes.backends.router.resolve_backend",
        lambda config: FakeBackend(text="Kumusta po kayo.", language="tl"),
    )
    result = run_pipeline(wav_file, AppConfig(use_ollama=False))
    assert result.notes.language == "tl"


def test_pipeline_rejects_taglish(wav_file):
    config = AppConfig(use_ollama=False)
    config.language_mode = LanguageMode.TAGLISH  # bypass validate()
    with pytest.raises(NotImplementedError, match="Taglish"):
        run_pipeline(wav_file, config)
