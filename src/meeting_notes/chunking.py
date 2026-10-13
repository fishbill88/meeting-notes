"""Cutting a long recording into parts that can be transcribed and diarized one at a time.

**Why.** One pass over an hour of audio gives no progress, holds everything in memory at once, and asks
the speaker diarizer to cluster an hour of voices in one go, where it drifts. Parts of about five minutes,
cut where nobody is speaking, can each be done and reported on, and let a long meeting be cancelled
without waiting for the end.

**Where to cut.** At the quietest moment near each five-minute mark. A cut inside a word loses or garbles
it in both neighbours; a cut in a pause costs nothing. The search is a window either side of the mark, so a
part may be a minute or so shorter or longer than five. No speech model is involved: a smoothed energy
curve is enough to find a pause, and cannot fail to load.

**What this does not do.** It does not detect speech or silence as such. If the whole window is loud (music,
a noisy room) the cut lands at the least-loud point, which is the best available. And a sentence that runs
across a cut is transcribed in two halves: the price of bounded parts, kept small by cutting in pauses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("meeting_notes.chunking")

#: Recordings up to this long are processed in one pass, exactly as before chunking existed. Below it the
#: single pass is fast enough, and splitting adds boundaries for no benefit.
CHUNK_THRESHOLD_SECONDS = 600.0

#: Aim for parts this long.
TARGET_SECONDS = 300.0
#: How far either side of the aim a cut may move to find a pause.
SEARCH_SECONDS = 45.0
#: No part shorter than this, except the last (whatever is left, which is never under MAX - TARGET - SEARCH).
MIN_SECONDS = 120.0
#: No part longer than this.
MAX_SECONDS = 420.0

#: Energy is measured in 20 ms frames and smoothed over 300 ms, so a cut lands in a real pause and not
#: between two syllables.
_FRAME_SECONDS = 0.02
_SMOOTH_FRAMES = 15


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float   # seconds
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def probe_duration(path: str) -> Optional[float]:
    """The length of an audio or video file in seconds, without decoding it; ``None`` if unknown.

    Uses PyAV (already a dependency, via faster-whisper). ``None`` means "treat it as short": a file whose
    length cannot be read is better processed whole than refused.
    """
    try:
        import av  # noqa: PLC0415

        with av.open(path) as container:
            if container.duration:
                return float(container.duration) / 1_000_000.0
            for stream in container.streams.audio:
                if stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
    except Exception:  # noqa: BLE001
        logger.debug("could not probe the duration of %s", path, exc_info=True)
    return None


def needs_chunking(duration_seconds: Optional[float]) -> bool:
    return duration_seconds is not None and duration_seconds > CHUNK_THRESHOLD_SECONDS


def plan_chunks(
    audio,
    sample_rate: int,
    target: float = TARGET_SECONDS,
    search: float = SEARCH_SECONDS,
    minimum: float = MIN_SECONDS,
    maximum: float = MAX_SECONDS,
) -> list[Chunk]:
    """Split ``audio`` (mono float32) into contiguous parts, cutting at pauses.

    Returns parts that tile the recording exactly: the first starts at 0, each starts where the one before
    ended, the last ends at the recording's length. Audio no longer than ``maximum`` is one part.
    """
    import numpy as np  # noqa: PLC0415

    total = len(audio) / float(sample_rate)
    if total <= maximum:
        return [Chunk(0, 0.0, total)]

    energy = _smoothed_energy(np.asarray(audio, dtype=np.float32), sample_rate)

    def time_of(frame: int) -> float:
        return frame * _FRAME_SECONDS

    cuts: list[float] = []
    position = 0.0
    while total - position > maximum:
        ideal = position + target
        low = max(position + minimum, ideal - search)
        high = min(position + maximum, ideal + search)
        first = int(low / _FRAME_SECONDS)
        last = max(first + 1, min(len(energy), int(high / _FRAME_SECONDS)))
        window = energy[first:last]

        floor = float(window.min())
        # Digital silence is a run of equal minima; of those, the one nearest the aim keeps parts even.
        quiet = np.flatnonzero(window <= floor + 1e-9 + abs(floor) * 1e-6)
        nearest = quiet[np.argmin(np.abs((first + quiet) * _FRAME_SECONDS - ideal))]
        cut = time_of(first + int(nearest))
        # Never a degenerate part, whatever the energy did.
        cut = min(max(cut, position + minimum), position + maximum)
        cuts.append(cut)
        position = cut

    # No tail handling is needed: a cut is only made while more than ``maximum`` remains, and a cut lands at
    # most ``target + search`` in, so what is left after the last cut is at least ``maximum - target - search``
    # (75 s with the defaults), never a sliver.
    edges = [0.0, *cuts, total]
    return [Chunk(i, edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _smoothed_energy(audio, sample_rate: int):
    import numpy as np  # noqa: PLC0415

    frame = max(1, int(sample_rate * _FRAME_SECONDS))
    usable = (len(audio) // frame) * frame
    if usable == 0:
        return np.zeros(1, dtype=np.float32)
    frames = audio[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    kernel = np.ones(_SMOOTH_FRAMES, dtype=np.float32) / _SMOOTH_FRAMES
    return np.convolve(rms, kernel, mode="same")
