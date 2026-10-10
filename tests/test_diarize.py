"""Tests for speaker diarization.

sherpa-onnx and its models are faked per the conftest rule: no test may
download a model. The autouse ``offline_diarization`` fixture points the
models dir at an empty temp dir and makes downloads raise.
"""

from __future__ import annotations

import sys
import types

import numpy as np

import meeting_notes.diarize as diarize
from meeting_notes.backends.base import TranscriptSegment
from meeting_notes.diarize import (
    SpeakerSpan,
    assign_speakers,
    diarize_file,
    ensure_models,
    label_segments,
)


def _segments(*times: tuple[float, float]) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(start=t0, end=t1, text=f"segment {i}")
        for i, (t0, t1) in enumerate(times)
    ]


# ---------------------------------------------------------------------------
# label_segments — the span→segment assignment rules
# ---------------------------------------------------------------------------


class TestLabelSegments:
    def test_labels_by_largest_overlap(self):
        segments = _segments((0.0, 2.0), (2.0, 4.0))
        spans = [
            SpeakerSpan(start=0.0, end=2.1, speaker=0),
            SpeakerSpan(start=2.1, end=4.0, speaker=1),
        ]
        assert label_segments(segments, spans) == 2
        assert [s.speaker for s in segments] == ["Person 1", "Person 2"]

    def test_numbering_follows_first_appearance(self):
        # Cluster ids are arbitrary; whoever speaks first is Person 1.
        segments = _segments((0.0, 1.0), (1.0, 2.0))
        spans = [
            SpeakerSpan(start=0.0, end=1.0, speaker=7),
            SpeakerSpan(start=1.0, end=2.0, speaker=3),
        ]
        label_segments(segments, spans)
        assert [s.speaker for s in segments] == ["Person 1", "Person 2"]

    def test_gap_segment_takes_nearest_span(self):
        # The middle segment sits in a diarization gap; its midpoint (2.5s)
        # is nearer the second span.
        segments = _segments((0.0, 1.0), (2.0, 3.0), (3.5, 4.5))
        spans = [
            SpeakerSpan(start=0.0, end=1.0, speaker=0),
            SpeakerSpan(start=3.4, end=4.5, speaker=1),
        ]
        label_segments(segments, spans)
        assert [s.speaker for s in segments] == ["Person 1", "Person 2", "Person 2"]

    def test_monologue_clears_labels(self):
        segments = _segments((0.0, 1.0), (1.0, 2.0))
        segments[0].speaker = "stale"
        spans = [SpeakerSpan(start=0.0, end=2.0, speaker=0)]
        assert label_segments(segments, spans) == 1
        assert all(s.speaker is None for s in segments)

    def test_empty_inputs(self):
        assert label_segments([], [SpeakerSpan(0.0, 1.0, 0)]) == 0
        assert label_segments(_segments((0.0, 1.0)), []) == 0


# ---------------------------------------------------------------------------
# assign_speakers — the pipeline-facing wrapper
# ---------------------------------------------------------------------------


class TestAssignSpeakers:
    def test_returns_zero_when_diarization_unavailable(self, wav_file):
        # Autouse fixture: sherpa may import, but the model download fails.
        segments = _segments((0.0, 1.0), (1.0, 2.0))
        assert assign_speakers(segments, wav_file) == 0
        assert all(s.speaker is None for s in segments)

    @staticmethod
    def _fake_diarize(monkeypatch, spans, voices=None):
        monkeypatch.setattr(
            diarize, "_diarize", lambda path: (spans, np.zeros(16000), 16000)
        )
        monkeypatch.setattr(
            diarize, "segment_voices", lambda segments, audio, rate: voices
        )

    def test_returns_zero_for_monologue(self, monkeypatch, wav_file):
        self._fake_diarize(monkeypatch, [SpeakerSpan(0.0, 2.0, 0)])
        segments = _segments((0.0, 1.0), (1.0, 2.0))
        assert assign_speakers(segments, wav_file) == 0
        assert all(s.speaker is None for s in segments)

    def test_labels_two_speakers(self, monkeypatch, wav_file):
        self._fake_diarize(
            monkeypatch, [SpeakerSpan(0.0, 1.0, 0), SpeakerSpan(1.0, 2.0, 1)]
        )
        segments = _segments((0.0, 1.0), (1.0, 2.0))
        assert assign_speakers(segments, wav_file) == 2
        assert [s.speaker for s in segments] == ["Person 1", "Person 2"]

    def test_voice_recheck_fixes_resumed_phrase(self, monkeypatch, wav_file):
        # The Taglish fixture's bug: B pauses and resumes with a short phrase
        # that the diarizer folds into A's next turn.
        segments = _segments((0.0, 3.0), (3.0, 6.0), (6.4, 7.3), (7.9, 10.0))
        spans = [
            SpeakerSpan(0.0, 3.0, 0),
            SpeakerSpan(3.0, 6.2, 1),
            SpeakerSpan(6.2, 10.0, 0),
        ]
        self._fake_diarize(monkeypatch, spans, [A, B, _voice(0.22, 0.57), A])
        assert assign_speakers(segments, wav_file) == 2
        assert [s.speaker for s in segments] == [
            "Person 1", "Person 2", "Person 2", "Person 1",
        ]

    def test_embedder_unavailable_keeps_diarizer_labels(self, monkeypatch, wav_file):
        segments = _segments((0.0, 3.0), (3.0, 6.0), (6.4, 7.3))
        self._fake_diarize(
            monkeypatch,
            [SpeakerSpan(0.0, 3.0, 0), SpeakerSpan(3.0, 6.2, 1), SpeakerSpan(6.2, 7.3, 0)],
        )
        assert assign_speakers(segments, wav_file) == 2
        assert segments[2].speaker == "Person 1"


# ---------------------------------------------------------------------------
# Voice re-check of short phrases (label_segments with voices)
# ---------------------------------------------------------------------------


def _voice(*components: float) -> np.ndarray:
    """A unit vector whose dot products with A/B/C are the given components.

    The remainder goes into an unused fourth axis, so ``_voice(0.22, 0.57)``
    scores 0.22 against A and 0.57 against B.
    """
    head = list(components) + [0.0] * (3 - len(components))
    rest = max(0.0, 1.0 - sum(c * c for c in head)) ** 0.5
    return np.array(head + [rest], dtype=np.float32)


A = _voice(1.0)
B = _voice(0.0, 1.0)
C = _voice(0.0, 0.0, 1.0)


class TestVoiceRecheck:
    def test_short_phrase_moves_to_the_speaker_it_sounds_like(self):
        segments = _segments((0.0, 3.0), (3.0, 6.0), (6.4, 7.3))
        spans = [SpeakerSpan(0.0, 3.0, 0), SpeakerSpan(3.0, 6.2, 1), SpeakerSpan(6.2, 7.3, 0)]
        label_segments(segments, spans, [A, B, _voice(0.22, 0.57)])
        assert segments[2].speaker == "Person 2"

    def test_ambiguous_phrase_keeps_diarizer_label(self):
        # Closer to B, but not by the margin: too close to call.
        segments = _segments((0.0, 3.0), (3.0, 6.0), (6.4, 7.3))
        spans = [SpeakerSpan(0.0, 3.0, 0), SpeakerSpan(3.0, 6.2, 1), SpeakerSpan(6.2, 7.3, 0)]
        label_segments(segments, spans, [A, B, _voice(0.45, 0.55)])
        assert segments[2].speaker == "Person 1"

    def test_weak_match_keeps_diarizer_label(self):
        # Clear winner by margin, but too dissimilar to everyone to trust.
        segments = _segments((0.0, 3.0), (3.0, 6.0), (6.4, 7.3))
        spans = [SpeakerSpan(0.0, 3.0, 0), SpeakerSpan(3.0, 6.2, 1), SpeakerSpan(6.2, 7.3, 0)]
        label_segments(segments, spans, [A, B, _voice(0.0, 0.35)])
        assert segments[2].speaker == "Person 1"

    def test_long_segments_are_never_moved(self):
        # A 3 s segment defines a profile; its label is the diarizer's call.
        segments = _segments((0.0, 3.0), (3.0, 6.0), (6.0, 9.0))
        spans = [SpeakerSpan(0.0, 3.0, 0), SpeakerSpan(3.0, 6.0, 1), SpeakerSpan(6.0, 9.0, 0)]
        label_segments(segments, spans, [A, B, B])
        assert segments[2].speaker == "Person 1"

    def test_ties_between_two_other_speakers_stay_put(self):
        segments = _segments((0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.2, 10.0))
        spans = [
            SpeakerSpan(0.0, 3.0, 0),
            SpeakerSpan(3.0, 6.0, 1),
            SpeakerSpan(6.0, 9.0, 2),
            SpeakerSpan(9.0, 10.0, 0),
        ]
        label_segments(segments, spans, [A, B, C, _voice(0.1, 0.6, 0.6)])
        assert segments[3].speaker == "Person 1"

    def test_phantom_speaker_is_absorbed_and_numbering_stays_contiguous(self):
        # The diarizer invented a third speaker for one short phrase that is
        # really B; once it moves, only two speakers remain, as Person 1-2.
        segments = _segments((0.0, 3.0), (3.2, 4.0), (4.0, 7.0))
        spans = [SpeakerSpan(0.0, 3.0, 5), SpeakerSpan(3.2, 4.0, 9), SpeakerSpan(4.0, 7.0, 2)]
        assert label_segments(segments, spans, [A, _voice(0.1, 0.8), B]) == 2
        assert [s.speaker for s in segments] == ["Person 1", "Person 2", "Person 2"]

    def test_unjudgeable_phrase_keeps_diarizer_label(self):
        segments = _segments((0.0, 3.0), (3.0, 6.0), (6.4, 6.7))
        spans = [SpeakerSpan(0.0, 3.0, 0), SpeakerSpan(3.0, 6.2, 1), SpeakerSpan(6.2, 6.7, 0)]
        label_segments(segments, spans, [A, B, None])
        assert segments[2].speaker == "Person 1"

    def test_one_profiled_speaker_means_nothing_to_compare(self):
        segments = _segments((0.0, 3.0), (3.2, 4.0))
        spans = [SpeakerSpan(0.0, 3.0, 0), SpeakerSpan(3.2, 4.0, 1)]
        label_segments(segments, spans, [A, _voice(0.9)])
        assert [s.speaker for s in segments] == ["Person 1", "Person 2"]

    def test_segment_voices_none_without_embedder(self):
        # Autouse fixture: the embedding model cannot be downloaded.
        assert diarize.segment_voices(_segments((0.0, 1.0)), np.zeros(16000), 16000) is None


# ---------------------------------------------------------------------------
# ensure_models — download-on-first-use behaviour
# ---------------------------------------------------------------------------


class TestEnsureModels:
    def test_uses_existing_files_without_downloading(self):
        # The autouse fixture makes any download raise, so this passing at
        # all proves no network was touched.
        target = diarize.diarization_dir()
        target.mkdir(parents=True)
        (target / diarize.SEGMENTATION_FILENAME).write_bytes(b"seg")
        (target / diarize.EMBEDDING_FILENAME).write_bytes(b"emb")

        models = ensure_models()
        assert models is not None
        segmentation, embedding = models
        assert segmentation.endswith(diarize.SEGMENTATION_FILENAME)
        assert embedding.endswith(diarize.EMBEDDING_FILENAME)

    def test_returns_none_when_download_fails(self):
        assert ensure_models() is None

    def test_zero_byte_file_is_re_downloaded(self, monkeypatch):
        # An interrupted earlier download can leave an empty file behind;
        # it must not be trusted as a model.
        target = diarize.diarization_dir()
        target.mkdir(parents=True)
        (target / diarize.SEGMENTATION_FILENAME).write_bytes(b"")
        (target / diarize.EMBEDDING_FILENAME).write_bytes(b"emb")

        downloaded = []

        def fake_urlretrieve(url, dest):
            downloaded.append(url)
            with open(dest, "wb") as fh:
                fh.write(b"model bytes")

        monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
        assert ensure_models() is not None
        assert downloaded == [diarize.SEGMENTATION_URL]

    def test_downloads_missing_files(self, monkeypatch):
        downloaded = []

        def fake_urlretrieve(url, dest):
            downloaded.append(url)
            with open(dest, "wb") as fh:
                fh.write(b"model bytes")

        monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
        assert ensure_models() is not None
        assert len(downloaded) == 2

        # A second call finds the files and downloads nothing further.
        assert ensure_models() is not None
        assert len(downloaded) == 2


# ---------------------------------------------------------------------------
# diarize_file — sherpa-onnx integration, module faked
# ---------------------------------------------------------------------------


def make_fake_sherpa_onnx(spans: list[SpeakerSpan]) -> types.ModuleType:
    """A stand-in ``sherpa_onnx`` returning canned diarization spans."""
    module = types.ModuleType("sherpa_onnx")

    class AnyConfig:
        def __init__(self, *args, **kwargs):
            pass

    for name in (
        "OfflineSpeakerDiarizationConfig",
        "OfflineSpeakerSegmentationModelConfig",
        "OfflineSpeakerSegmentationPyannoteModelConfig",
        "SpeakerEmbeddingExtractorConfig",
        "FastClusteringConfig",
    ):
        setattr(module, name, AnyConfig)

    class FakeResult:
        def sort_by_start_time(self):
            return spans

    class FakeDiarizer:
        sample_rate = 16000

        def __init__(self, config):
            pass

        def process(self, samples, callback=None):
            return FakeResult()

    module.OfflineSpeakerDiarization = FakeDiarizer
    return module


class TestDiarizeFile:
    def _install(self, monkeypatch, tmp_path, spans):
        seg = tmp_path / "seg.onnx"
        emb = tmp_path / "emb.onnx"
        seg.write_bytes(b"x")
        emb.write_bytes(b"x")
        monkeypatch.setattr(diarize, "ensure_models", lambda: (str(seg), str(emb)))
        monkeypatch.setitem(
            sys.modules, "sherpa_onnx", make_fake_sherpa_onnx(spans)
        )

    def test_returns_spans(self, monkeypatch, tmp_path, wav_file):
        expected = [SpeakerSpan(0.0, 0.5, 0), SpeakerSpan(0.5, 1.0, 1)]
        self._install(monkeypatch, tmp_path, expected)
        assert diarize_file(wav_file) == expected

    def test_caches_the_diarizer(self, monkeypatch, tmp_path, wav_file):
        self._install(monkeypatch, tmp_path, [SpeakerSpan(0.0, 1.0, 0)])
        assert diarize_file(wav_file) is not None
        first = diarize._diarizer
        assert diarize_file(wav_file) is not None
        assert diarize._diarizer is first

    def test_returns_none_when_models_unavailable(
        self, monkeypatch, tmp_path, wav_file
    ):
        monkeypatch.setattr(diarize, "ensure_models", lambda: None)
        monkeypatch.setitem(sys.modules, "sherpa_onnx", make_fake_sherpa_onnx([]))
        assert diarize_file(wav_file) is None

    def test_returns_none_for_unreadable_audio(self, monkeypatch, tmp_path):
        # A corrupt file must degrade to "no speakers", never raise.
        self._install(monkeypatch, tmp_path, [SpeakerSpan(0.0, 1.0, 0)])
        bad = tmp_path / "corrupt.wav"
        bad.write_bytes(b"this is not audio")
        assert diarize_file(str(bad)) is None

    def test_failure_is_remembered_for_the_process(
        self, monkeypatch, tmp_path, wav_file
    ):
        calls = []

        def counting_ensure():
            calls.append(1)
            return None

        monkeypatch.setattr(diarize, "ensure_models", counting_ensure)
        monkeypatch.setitem(sys.modules, "sherpa_onnx", make_fake_sherpa_onnx([]))
        assert diarize_file(wav_file) is None
        assert diarize_file(wav_file) is None
        assert len(calls) == 1  # the second call did not retry the download
