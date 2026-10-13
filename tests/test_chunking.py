"""Tests for the chunk planner, speaker linking across parts, and the chunked pipeline.

The planner is exercised on synthetic audio at a low sample rate (a "recording" of half an hour is under two
million samples, so these run in milliseconds). The linker takes synthetic voice fingerprints with known
identities. The pipeline runs with a fake backend and a fake diarizer, so no model is loaded.
What these guard: parts tile the recording exactly and are cut in pauses, a speaker keeps one identity across
parts, short recordings are untouched, cancelling stops between parts, and a failing part fails by name.
"""

from __future__ import annotations

import numpy as np
import pytest

from meeting_notes import chunking, pipeline
from meeting_notes.backends.base import TranscriptionResult, TranscriptSegment
from meeting_notes.cancel import PipelineCancelled
from meeting_notes.chunking import Chunk, needs_chunking, plan_chunks
from meeting_notes.config import AppConfig, LanguageMode
from meeting_notes.diarize import ChunkSpeakers, SpeakerLinker

SR = 1000   # a low rate keeps the synthetic recordings tiny; the planner only cares about seconds


def _speech(seconds: float, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(seconds * SR)) * 0.2).astype(np.float32)


def _recording(total: float, silences: list[tuple[float, float]], seed: int = 1) -> np.ndarray:
    """Noise ("speech") everywhere except digital silence in the given (start, end) intervals."""
    audio = _speech(total, seed)
    for start, end in silences:
        audio[int(start * SR):int(end * SR)] = 0.0
    return audio


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------


def test_a_short_recording_is_one_part():
    parts = plan_chunks(_speech(300), SR)
    assert parts == [Chunk(0, 0.0, 300.0)]


def test_a_recording_just_under_the_maximum_is_still_one_part():
    parts = plan_chunks(_speech(chunking.MAX_SECONDS - 1), SR)
    assert len(parts) == 1


def test_parts_tile_the_recording_exactly():
    audio = _speech(1800, seed=3)
    parts = plan_chunks(audio, SR)
    assert len(parts) >= 5
    assert parts[0].start == 0.0
    assert parts[-1].end == pytest.approx(len(audio) / SR)
    for before, after in zip(parts, parts[1:]):
        assert before.end == after.start                 # no gap, no overlap
    assert [p.index for p in parts] == list(range(len(parts)))


def test_no_part_is_longer_than_the_maximum_or_shorter_than_the_minimum():
    parts = plan_chunks(_speech(2500, seed=4), SR)
    assert all(p.duration <= chunking.MAX_SECONDS + 1e-6 for p in parts)
    assert all(p.duration >= chunking.MIN_SECONDS - 1e-6 for p in parts[:-1])
    assert parts[-1].duration >= chunking.MAX_SECONDS - chunking.TARGET_SECONDS - chunking.SEARCH_SECONDS - 1e-6


def test_cuts_land_in_the_pauses():
    # A pause near each five-minute mark, but not exactly on it.
    pauses = [(291.0, 293.0), (583.0, 585.0), (880.0, 882.5), (1188.0, 1190.0)]
    parts = plan_chunks(_recording(1500, pauses), SR)
    cuts = [p.start for p in parts[1:]]
    assert len(cuts) >= 3
    for cut in cuts:
        assert any(start - 0.3 <= cut <= end + 0.3 for start, end in pauses), f"cut at {cut:.1f}s is mid-speech"


def test_the_cut_is_in_the_middle_of_a_long_pause_nearest_the_aim():
    # One long pause spanning the aim: the cut should be near the aim, not at the pause's edge.
    audio = _recording(900, [(270.0, 330.0)])
    first_cut = plan_chunks(audio, SR)[0].end
    assert 285.0 <= first_cut <= 315.0


def test_with_no_pause_at_all_the_parts_still_respect_the_bounds():
    parts = plan_chunks(_speech(2000, seed=9), SR)
    assert all(chunking.MIN_SECONDS - 1e-6 <= p.duration <= chunking.MAX_SECONDS + 1e-6 for p in parts[:-1])


def test_parts_are_near_the_target_when_pauses_are_everywhere():
    # Short pauses every 10 s: the search window always finds one, so parts stay close to the target.
    pauses = [(t, t + 0.6) for t in np.arange(10, 1800, 10.0)]
    parts = plan_chunks(_recording(1800, pauses, seed=5), SR)
    assert all(abs(p.duration - chunking.TARGET_SECONDS) <= chunking.SEARCH_SECONDS + 1.0 for p in parts[:-1])


def test_planning_is_deterministic():
    audio = _speech(1600, seed=6)
    assert plan_chunks(audio, SR) == plan_chunks(audio, SR)


def test_silent_or_tiny_audio_does_not_crash():
    assert len(plan_chunks(np.zeros(10, dtype=np.float32), SR)) == 1
    assert len(plan_chunks(np.zeros(int(1000 * SR), dtype=np.float32), SR)) >= 2   # 1000 s of silence


# ---------------------------------------------------------------------------
# When to chunk
# ---------------------------------------------------------------------------


def test_the_threshold():
    assert not needs_chunking(None)                                  # length unknown: process whole
    assert not needs_chunking(chunking.CHUNK_THRESHOLD_SECONDS)      # at the threshold: not yet
    assert needs_chunking(chunking.CHUNK_THRESHOLD_SECONDS + 1)


def test_probing_a_missing_file_is_none_not_an_error(tmp_path):
    assert chunking.probe_duration(str(tmp_path / "nope.webm")) is None


# ---------------------------------------------------------------------------
# Speaker linking across parts
# ---------------------------------------------------------------------------


def _voice(identity: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """A unit fingerprint for a made-up person: a fixed direction per identity plus a little noise."""
    base = np.zeros(64, dtype=np.float32)
    base[identity * 8:(identity + 1) * 8] = 1.0
    vec = base + np.random.default_rng(seed).standard_normal(64).astype(np.float32) * noise
    return vec / np.linalg.norm(vec)


def _part(*identities: int, noise: float = 0.05, seed: int = 0, weight: int = 3) -> ChunkSpeakers:
    """A part whose local clusters 0, 1, … are the given people, in the order they first speak."""
    return ChunkSpeakers(
        clusters=list(range(len(identities))),
        fingerprints={i: _voice(who, noise, seed + i) for i, who in enumerate(identities)},
        weights={i: weight for i in range(len(identities))},
    )


def test_the_same_people_keep_one_identity_across_parts():
    linker = SpeakerLinker()
    first = linker.link(_part(0, 1, 2, seed=1))
    assert first == {0: 0, 1: 1, 2: 2}
    # Part two: the same three people, but the diarizer numbered them differently and they speak in another order.
    second = linker.link(_part(2, 0, 1, seed=10))
    assert second == {0: 2, 1: 0, 2: 1}
    assert linker.speaker_count == 3


def test_a_new_voice_becomes_a_new_speaker_numbered_after_the_known_ones():
    linker = SpeakerLinker()
    linker.link(_part(0, 1))
    mapping = linker.link(_part(1, 3))          # person 1 returns, a stranger (3) joins
    assert mapping[0] == 1
    assert mapping[1] == 2
    assert linker.speaker_count == 3


def test_two_similar_but_distinct_speakers_in_one_part_are_not_merged():
    """Both are alike a known speaker, but not alike enough to be one voice split in two: the diarizer's
    verdict that they are different people stands."""
    linker = SpeakerLinker()
    linker.link(_part(0))
    near = _voice(0, 0.0, 1)
    other = _voice(1)
    # 0.7 alike the known speaker: above the link threshold, below the merge threshold.
    middling = near * 0.7 + other * np.sqrt(1 - 0.49)
    middling /= np.linalg.norm(middling)
    closer = near * 0.9 + other * np.sqrt(1 - 0.81)
    closer /= np.linalg.norm(closer)
    assert 0.55 < float(middling @ near) < 0.78 and 0.55 < float(closer @ near) < 0.95
    mapping = linker.link(ChunkSpeakers(
        clusters=[0, 1], fingerprints={0: closer, 1: middling}, weights={0: 3, 1: 3}))
    assert mapping[0] != mapping[1]


def test_a_diarizer_split_of_one_voice_is_rejoined():
    """A host's interjections often come out as a cluster of their own. Near-identical fingerprints in one
    part are one person, however many clusters the diarizer made of them."""
    linker = SpeakerLinker()
    linker.link(_part(0, 1))
    both_like_zero = ChunkSpeakers(
        clusters=[0, 1],
        fingerprints={0: _voice(0, 0.02, 1), 1: _voice(0, 0.02, 2)},
        weights={0: 9, 1: 2},
    )
    mapping = linker.link(both_like_zero)
    assert mapping[0] == mapping[1] == 0
    assert linker.speaker_count == 2


def test_a_split_voice_inside_the_first_part_is_one_speaker():
    linker = SpeakerLinker()
    split = ChunkSpeakers(
        clusters=[0, 1, 2],
        fingerprints={0: _voice(0, 0.02, 1), 1: _voice(1, 0.02, 2), 2: _voice(0, 0.02, 3)},
        weights={0: 2, 1: 5, 2: 8},
    )
    mapping = linker.link(split)
    assert mapping[0] == mapping[2] != mapping[1]
    assert linker.speaker_count == 2


def test_the_best_match_wins_when_two_compete_for_one_known_speaker():
    linker = SpeakerLinker()
    linker.link(_part(0))
    close = _voice(0, 0.01, 1)
    far = (_voice(0, 0.01, 2) * 0.5 + _voice(1) * 0.5)      # ~0.7 alike: competes, but is not a fragment
    far /= np.linalg.norm(far)
    mapping = linker.link(ChunkSpeakers(clusters=[0, 1], fingerprints={0: far, 1: close}, weights={0: 3, 1: 3}))
    assert mapping[1] == 0          # the closer one is speaker 0
    assert mapping[0] == 1          # the other is somebody new


def test_a_cluster_with_no_fingerprint_is_left_out_not_guessed():
    linker = SpeakerLinker()
    part = ChunkSpeakers(clusters=[0, 1], fingerprints={0: _voice(0)}, weights={0: 3})
    mapping = linker.link(part)
    assert 1 not in mapping


def test_the_global_fingerprint_steadies_as_parts_accumulate():
    """After several parts the running mean is a better reference than any one noisy part."""
    linker = SpeakerLinker()
    for seed in range(1, 8):
        linker.link(_part(0, 1, noise=0.15, seed=seed))
    mapping = linker.link(_part(1, 0, noise=0.15, seed=99))
    assert mapping == {0: 1, 1: 0}


def test_the_link_threshold_separates_the_measured_same_and_different_speaker_scores():
    # From the fixtures (diarize.py): same speaker 0.70-0.96, best wrong speaker 0.51.
    assert 0.51 < SpeakerLinker().threshold < 0.70
    # Diarizer-split fragments of one voice scored 0.81-0.94; different people never passed 0.59.
    assert 0.59 < SpeakerLinker().merge_threshold < 0.81


# ---------------------------------------------------------------------------
# The chunked pipeline, with a fake backend and a fake diarizer
# ---------------------------------------------------------------------------


class FakeBackend:
    """Transcribes each part as two segments of known text, with part-relative times."""

    name = "fake"
    language = None

    def __init__(self, fail_part: int | None = None, fail_times: int = 99):
        self.calls: list[float] = []
        self.languages_used: list[str] = []
        self.done = 0
        self.fail_part, self.fail_times = fail_part, fail_times
        self._failures = 0

    def transcribe_array(self, audio, sample_rate=16000):
        index = self.done                  # the part being worked on: it does not advance on a failed attempt
        self.calls.append(len(audio) / sample_rate)
        if self.fail_part == index and self._failures < self.fail_times:
            self._failures += 1
            raise RuntimeError("the model fell over")
        self.done += 1
        # A real backend honours the language it was told to use and otherwise detects; here detection
        # would flip to "tl" after the first part, which is exactly what the lock exists to prevent.
        detected = self.language or ("en" if index == 0 else "tl")
        self.languages_used.append(detected)
        return TranscriptionResult(
            text=f"part {index}",
            language=detected,
            language_name="x",
            backend="fake",
            segments=[
                TranscriptSegment(start=1.0, end=4.0, text=f"part{index} first"),
                TranscriptSegment(start=10.0, end=14.0, text=f"part{index} second"),
            ],
        )


@pytest.fixture
def long_audio(monkeypatch):
    """A 20-minute "recording" at 16 kHz is 19 M samples, so use a low rate: the pipeline takes whatever
    load_audio_file reports."""
    audio = _recording(1200, [(298.0, 301.0), (598.0, 601.0), (898.0, 901.0)])
    monkeypatch.setattr("meeting_notes.audio.load_audio_file", lambda path: (audio, SR))
    monkeypatch.setattr(chunking, "probe_duration", lambda path: 1200.0)
    return audio


def _config(**kw):
    return AppConfig(language_mode=LanguageMode.AUTO, use_ollama=False, diarize=kw.pop("diarize", False), **kw)


def _run(backend, config=None, **kw):
    import meeting_notes.backends.router as router

    router_resolve = router.resolve_backend
    router.resolve_backend = lambda cfg: backend
    try:
        return pipeline.run_pipeline("long.webm", config or _config(), **kw)
    finally:
        router.resolve_backend = router_resolve


def test_segments_from_every_part_are_stitched_in_order_with_absolute_times(long_audio):
    backend = FakeBackend()
    result = _run(backend)
    parts = len(backend.calls)
    assert parts >= 3
    segments = result.transcript.segments
    assert len(segments) == parts * 2
    starts = [s.start for s in segments]
    assert starts == sorted(starts)                       # in order across parts
    assert segments[0].start == pytest.approx(1.0)
    assert segments[2].start > 200                         # part two's "1.0 s" is at ITS offset in the recording
    assert "part0 first" in result.transcript.text and f"part{parts - 1} second" in result.transcript.text


def test_the_first_parts_language_is_kept_for_the_whole_recording(long_audio):
    backend = FakeBackend()
    result = _run(backend)
    assert backend.language == "en"                         # locked after part one
    assert backend.languages_used[0] == "en"
    assert set(backend.languages_used[1:]) == {"en"}        # and every later part was decoded as English
    assert result.transcript.language == "en"


def test_progress_is_reported_per_part_and_ends_at_one(long_audio):
    messages, fractions = [], []
    _run(FakeBackend(), on_progress=lambda s, m: messages.append(m), on_fraction=fractions.append)
    assert any("part 1 of" in m for m in messages)
    assert any("part 2 of" in m for m in messages)
    assert fractions == sorted(fractions)                   # never goes backwards
    assert fractions[-1] == 1.0
    assert 0 < fractions[0] < 0.5


def test_a_short_recording_takes_the_single_pass_untouched(monkeypatch):
    monkeypatch.setattr(chunking, "probe_duration", lambda path: 120.0)
    called = {}

    class Whole(FakeBackend):
        def transcribe_file(self, path):
            called["file"] = path
            return TranscriptionResult(text="whole", language="en", language_name="English", backend="fake", segments=[])

    backend = Whole()
    fractions = []
    result = _run(backend, on_fraction=fractions.append)
    assert called["file"] == "long.webm"                    # the old path: the backend read the file itself
    assert backend.calls == []                              # never asked for parts
    assert result.transcript.text == "whole"
    assert fractions == [1.0]                               # no part-by-part progress without parts


def test_cancelling_stops_between_parts(long_audio):
    backend = FakeBackend()
    state = {"stop": False}

    def on_progress(stage, message):
        if "part 2 of" in message:
            state["stop"] = True

    with pytest.raises(PipelineCancelled):
        _run(backend, on_progress=on_progress, should_cancel=lambda: state["stop"])
    assert len(backend.calls) < 4                           # did not carry on to the end of the meeting


def test_a_part_that_fails_once_is_retried(long_audio):
    backend = FakeBackend(fail_part=1, fail_times=1)
    result = _run(backend)
    assert result.transcript.segments
    clean = FakeBackend()
    _run(clean)
    assert len(backend.calls) == len(clean.calls) + 1       # exactly one extra attempt, then on to the end


def test_a_part_that_fails_twice_fails_the_recording_by_name(long_audio):
    with pytest.raises(RuntimeError, match=r"part 2 of \d+"):
        _run(FakeBackend(fail_part=1, fail_times=2))


def test_speakers_keep_one_identity_across_parts_and_are_numbered_by_first_appearance(long_audio, monkeypatch):
    """Diarize each part with a fake that numbers its local speakers arbitrarily, and check the stitched
    labels are consistent. Two real people, A and B; the fake swaps their local ids in every other part."""
    import meeting_notes.diarize as diarize

    person_of_segment = {0: "A", 1: "B"}          # first segment of a part is A, second is B

    def fake_diarize_chunk(segments, audio, sample_rate):
        parity = fake_diarize_chunk.calls % 2
        fake_diarize_chunk.calls += 1
        # local cluster ids swap every other part, as a real diarizer's arbitrary numbering does
        clusters = [(i + parity) % 2 for i, _ in enumerate(segments)]
        identity = {"A": 0, "B": 1}
        return ChunkSpeakers(
            clusters=clusters,
            fingerprints={c: _voice(identity[person_of_segment[i]], 0.03, 7 + c)
                          for i, c in enumerate(clusters)},
            weights={c: 3 for c in clusters},
        )

    fake_diarize_chunk.calls = 0
    monkeypatch.setattr(diarize, "diarize_chunk", fake_diarize_chunk)

    result = _run(FakeBackend(), config=_config(diarize=True))
    labels = [s.speaker for s in result.transcript.segments]
    assert set(labels) == {"Person 1", "Person 2"}
    assert labels[0] == "Person 1"                       # the first to speak
    assert labels[0::2] == ["Person 1"] * (len(labels) // 2)    # A is always the same person, in every part
    assert labels[1::2] == ["Person 2"] * (len(labels) // 2)


def test_one_voice_in_the_whole_recording_means_no_labels(long_audio, monkeypatch):
    import meeting_notes.diarize as diarize

    monkeypatch.setattr(diarize, "diarize_chunk", lambda segs, audio, sr: ChunkSpeakers(
        clusters=[0] * len(segs), fingerprints={0: _voice(0, 0.02, 1)}, weights={0: 3}))
    result = _run(FakeBackend(), config=_config(diarize=True))
    assert all(s.speaker is None for s in result.transcript.segments)


def test_a_part_whose_diarization_fails_is_left_unlabelled_not_dropped(long_audio, monkeypatch):
    import meeting_notes.diarize as diarize

    monkeypatch.setattr(diarize, "diarize_chunk", lambda segs, audio, sr: None)
    result = _run(FakeBackend(), config=_config(diarize=True))
    assert result.transcript.segments                      # the transcript is the product; labels are a bonus
    assert all(s.speaker is None for s in result.transcript.segments)
