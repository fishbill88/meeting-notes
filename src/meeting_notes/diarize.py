"""Speaker diarization via sherpa-onnx — who spoke each transcript segment.

The Python counterpart of the mobile app's ``diarize.ts`` +
``DiarizationModule.kt``: the same two ONNX models (pyannote segmentation-3.0
finds speaker-change points acoustically; a CAM++ embedding identifies who is
who), run through sherpa-onnx's offline speaker diarization pipeline, so
desktop and mobile label the same audio the same way. Desktop additionally
runs a GTCRN denoise pre-pass when available — background beds destabilise
the speaker embeddings — which mobile does not have yet. Desktop also
re-checks each short transcript phrase against every speaker's voice after
diarization (``_refine_clusters``), which mobile does not do either.

Everything here degrades to "no speakers" rather than failing the request: a
recorded meeting must never be lost because an optional model could not be
downloaded. Callers check for ``None``/``[]`` and move on.

Models live under ``config.models_dir()/diarization`` (~34 MB total) and are
downloaded on first use — the one network touch, same trade-off the mobile
app makes.
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from meeting_notes.backends.base import TranscriptSegment

logger = logging.getLogger("meeting_notes.diarize")

#: Same artifacts the mobile app downloads (see mobile/lib/models.ts and
#: MODELS.md). pyannote segmentation-3.0 is MIT, rehosted ungated by the
#: sherpa-onnx project; the misspelt "recongition" is in the real release tag.
SEGMENTATION_URL = (
    "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0"
    "/resolve/main/model.onnx"
)
SEGMENTATION_FILENAME = "pyannote-segmentation-3-0.onnx"
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download"
    "/speaker-recongition-models"
    "/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)
EMBEDDING_FILENAME = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
#: GTCRN speech denoiser (~0.5 MB) — strips background beds/music before
#: diarization. Optional: when it cannot be fetched or built, diarization
#: runs on the raw audio (the threshold below works for both).
DENOISER_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download"
    "/speech-enhancement-models/gtcrn_simple.onnx"
)
DENOISER_FILENAME = "gtcrn_simple.onnx"

#: -1 lets sherpa-onnx pick the speaker count.
AUTO_SPEAKER_COUNT = -1
#: Clustering threshold: larger = fewer, bigger clusters. Benchmarked
#: 2026-08-28 against a labelled 3-voice fixture of edited audio with a
#: background bed: the earlier VoxCeleb-EN CAM++ model never recovered the
#: right speaker count at any threshold, while this zh-en "advanced" CAM++
#: gets exactly 3 speakers at 0.6 both raw (94% purity) and GTCRN-denoised
#: (97.5%). 0.8 over-merges denoised audio; sherpa's 0.5 default
#: over-splits. Mobile passes the same 0.6 (see mobile/lib/diarize.ts).
CLUSTER_THRESHOLD = 0.6

#: Voice re-check of short phrases (see ``_refine_clusters``). Transcript
#: segments at least ANCHOR_SECONDS long build each speaker's voice profile;
#: shorter ones may move to another speaker when their own voice matches that
#: profile at RELABEL_MIN_SIMILARITY or better AND beats every other speaker
#: by RELABEL_MARGIN. Tuned 2026-09-25 on the Recordings/ fixtures: the
#: mislabelled Taglish "Ready na yun mamaya." (0.9 s) scores 0.57 against its
#: real speaker vs 0.22 against the diarized one. Phrases of 2 s or more
#: score 0.70–0.96 against their own speaker; a short phrase's best score
#: against a wrong speaker reached 0.51, but always far behind its own.
ANCHOR_SECONDS = 2.0
RELABEL_MIN_SIMILARITY = 0.4
RELABEL_MARGIN = 0.2
#: Below this a phrase's embedding is noise; it keeps the diarizer's label.
MIN_VOICE_SECONDS = 0.5

#: Loaded diarizer, cached for the process lifetime — the models take longer
#: to load than to run on a short consultation. The lock keeps concurrent
#: first calls from racing the download and building duplicate diarizers;
#: today's FastAPI routes happen to serialize on one event-loop thread, but
#: that is an accident this module must not depend on.
_diarizer = None
_diarizer_failed = False
_denoiser = None
_denoiser_failed = False
_embedder = None
_embedder_failed = False
_diarizer_lock = threading.Lock()


@dataclass
class SpeakerSpan:
    """One diarized stretch of audio: cluster ``speaker`` spoke start→end."""

    start: float   # seconds
    end: float     # seconds
    speaker: int   # 0-based cluster id


def diarization_dir() -> Path:
    """Where the diarization models live."""
    from meeting_notes.config import models_dir  # noqa: PLC0415

    return models_dir() / "diarization"


def ensure_models() -> Optional[tuple[str, str]]:
    """Return local paths of (segmentation, embedding), downloading on
    first use. ``None`` when either cannot be fetched — callers fall back
    to a speakerless transcript, mirroring ``ensureDiarizationModels`` on
    mobile.
    """
    target = diarization_dir()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("cannot create %s", target)
        return None

    paths = []
    for url, filename in (
        (SEGMENTATION_URL, SEGMENTATION_FILENAME),
        (EMBEDDING_URL, EMBEDDING_FILENAME),
    ):
        dest = target / filename
        if not (dest.exists() and dest.stat().st_size > 0):
            partial = dest.with_suffix(".part")
            try:
                logger.info("downloading %s", url)
                urllib.request.urlretrieve(url, partial)  # noqa: S310
                partial.replace(dest)
            except Exception:  # noqa: BLE001 - any failure means "unavailable"
                logger.warning("could not download %s", url, exc_info=True)
                partial.unlink(missing_ok=True)
                return None
        paths.append(str(dest))

    return paths[0], paths[1]


def _load_diarizer():
    """Build (or return the cached) sherpa-onnx diarizer, or ``None``."""
    global _diarizer, _diarizer_failed
    with _diarizer_lock:
        if _diarizer is not None:
            return _diarizer
        if _diarizer_failed:
            # A missing wheel or model will not fix itself mid-process; do
            # not re-attempt a 34 MB download on every request.
            return None

        try:
            import sherpa_onnx  # noqa: PLC0415
        except ImportError:
            logger.info("sherpa-onnx not installed; speaker diarization disabled")
            _diarizer_failed = True
            return None

        models = ensure_models()
        if models is None:
            _diarizer_failed = True
            return None
        segmentation, embedding = models

        try:
            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=segmentation
                    ),
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=embedding
                ),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=AUTO_SPEAKER_COUNT,
                    threshold=CLUSTER_THRESHOLD,
                ),
            )
            _diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
        except Exception:  # noqa: BLE001 - corrupt model file, ABI mismatch, …
            logger.exception("failed to initialise the diarizer")
            _diarizer_failed = True
            return None

        return _diarizer


def _ensure_denoiser_model() -> Optional[str]:
    """Local path of the GTCRN model, downloading on first use, or ``None``."""
    target = diarization_dir()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("cannot create %s", target)
        return None

    dest = target / DENOISER_FILENAME
    if not (dest.exists() and dest.stat().st_size > 0):
        partial = dest.with_suffix(".part")
        try:
            logger.info("downloading %s", DENOISER_URL)
            urllib.request.urlretrieve(DENOISER_URL, partial)  # noqa: S310
            partial.replace(dest)
        except Exception:  # noqa: BLE001 - any failure means "unavailable"
            logger.warning("could not download %s", DENOISER_URL, exc_info=True)
            partial.unlink(missing_ok=True)
            return None
    return str(dest)


def _load_denoiser():
    """Build (or return the cached) GTCRN denoiser, or ``None``.

    The denoiser is a nice-to-have on top of diarization: embeddings are far
    more stable once a background bed (music, hum, room noise) is stripped.
    Any failure — old sherpa-onnx wheel without ``OfflineSpeechDenoiser``,
    download refused — means diarization proceeds on the raw audio.
    """
    global _denoiser, _denoiser_failed
    with _diarizer_lock:
        if _denoiser is not None:
            return _denoiser
        if _denoiser_failed:
            return None

        try:
            import sherpa_onnx  # noqa: PLC0415

            model = _ensure_denoiser_model()
            if model is None:
                raise RuntimeError("denoiser model unavailable")
            _denoiser = sherpa_onnx.OfflineSpeechDenoiser(
                sherpa_onnx.OfflineSpeechDenoiserConfig(
                    model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                        gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                            model=model
                        )
                    )
                )
            )
        except Exception:  # noqa: BLE001
            logger.info(
                "speech denoiser unavailable; diarizing raw audio", exc_info=True
            )
            _denoiser_failed = True
            return None

        return _denoiser


def _load_embedder():
    """Build (or return the cached) speaker-embedding extractor, or ``None``.

    The same CAM++ model the diarizer clusters with, loaded standalone so a
    transcript phrase can be embedded on its own. Optional like the
    denoiser: without it the diarizer's labels stand unrefined.
    """
    global _embedder, _embedder_failed
    with _diarizer_lock:
        if _embedder is not None:
            return _embedder
        if _embedder_failed:
            return None

        try:
            import sherpa_onnx  # noqa: PLC0415

            models = ensure_models()
            if models is None:
                raise RuntimeError("embedding model unavailable")
            _embedder = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=models[1])
            )
        except Exception:  # noqa: BLE001
            logger.info(
                "speaker embedder unavailable; keeping diarizer labels",
                exc_info=True,
            )
            _embedder_failed = True
            return None

        return _embedder


def _diarize(audio_path: str):
    """Diarize ``audio_path``.

    Returns ``(spans, audio, sample_rate)`` — the audio as the diarizer saw
    it (resampled, denoised when possible), so the phrase re-check embeds
    exactly the same signal — or ``None`` on any failure.
    """
    try:
        from meeting_notes.audio import load_audio_file  # noqa: PLC0415

        audio, sample_rate = load_audio_file(audio_path)
    except Exception:  # noqa: BLE001
        logger.exception("could not read %s for diarization", audio_path)
        return None
    return _diarize_samples(audio, sample_rate, audio_path)


def _diarize_samples(audio, sample_rate: int, label: str = "audio"):
    """Diarize samples already in memory (mono float32). Same result shape as :func:`_diarize`.

    This is what a CHUNK goes through: a long recording is decoded once and each part diarized from its
    slice, with no temporary file per part.
    """
    diarizer = _load_diarizer()
    if diarizer is None:
        return None

    try:
        import numpy as np  # noqa: PLC0415

        from meeting_notes.backends.base import resample  # noqa: PLC0415

        audio = np.asarray(
            resample(audio, sample_rate, diarizer.sample_rate), dtype=np.float32
        )

        denoiser = _load_denoiser()
        if denoiser is not None:
            try:
                denoised = denoiser(audio, diarizer.sample_rate)
                audio = np.asarray(
                    resample(
                        np.asarray(denoised.samples, dtype=np.float32),
                        denoised.sample_rate,
                        diarizer.sample_rate,
                    ),
                    dtype=np.float32,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "denoising failed; diarizing raw audio", exc_info=True
                )

        result = diarizer.process(audio)
        segments = result.sort_by_start_time()
    except Exception:  # noqa: BLE001
        logger.exception("diarization failed for %s", label)
        return None

    spans = [SpeakerSpan(start=s.start, end=s.end, speaker=s.speaker) for s in segments]
    return spans, audio, diarizer.sample_rate


def diarize_file(audio_path: str) -> Optional[list[SpeakerSpan]]:
    """Diarize an audio file into speaker spans.

    Returns ``None`` when diarization is unavailable or fails for any
    reason — never raises, because the transcript matters more than the
    speaker labels.
    """
    diarized = _diarize(audio_path)
    return diarized[0] if diarized else None


def segment_voices(segments: list[TranscriptSegment], audio, sample_rate: int):
    """Embed each transcript segment's own audio (unit vectors).

    An entry is ``None`` for a segment too short to judge. Returns ``None``
    overall when the embedder is unavailable or fails — the caller then
    keeps the diarizer's labels as they are.
    """
    extractor = _load_embedder()
    if extractor is None:
        return None

    try:
        import numpy as np  # noqa: PLC0415

        voices = []
        for seg in segments:
            if seg.end - seg.start < MIN_VOICE_SECONDS:
                voices.append(None)
                continue
            stream = extractor.create_stream()
            stream.accept_waveform(
                sample_rate,
                audio[int(seg.start * sample_rate):int(seg.end * sample_rate)],
            )
            stream.input_finished()
            if not extractor.is_ready(stream):
                voices.append(None)
                continue
            vector = np.asarray(extractor.compute(stream), dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            voices.append(vector / norm if norm > 0 else None)
    except Exception:  # noqa: BLE001
        logger.warning(
            "phrase embedding failed; keeping diarizer labels", exc_info=True
        )
        return None
    return voices


def _refine_clusters(
    clusters: list[int],
    segments: list[TranscriptSegment],
    voices: list,
) -> list[int]:
    """Move short phrases whose own voice clearly belongs to another speaker.

    The diarizer places turn boundaries from the audio alone, and its
    embedding of a brief stretch is unreliable, so a short phrase at a turn
    edge — typically someone resuming after a pause just before the next
    person speaks — can land in the neighbour's span. Whisper's segment
    boundaries follow those pauses, so embedding each short segment on its
    own and comparing it to every speaker's profile (built from their long
    segments, where the diarizer is reliable) catches the flip.

    Long segments are never moved: they define the profiles. A short phrase
    that is ambiguous or weakly matched keeps the diarizer's call.
    """
    import numpy as np  # noqa: PLC0415

    anchors: dict[int, list] = {}
    for cluster, seg, voice in zip(clusters, segments, voices):
        if voice is not None and seg.end - seg.start >= ANCHOR_SECONDS:
            anchors.setdefault(cluster, []).append(voice)
    if len(anchors) < 2:
        return clusters

    profiles = {}
    for cluster, vectors in anchors.items():
        mean = np.mean(vectors, axis=0)
        profiles[cluster] = mean / np.linalg.norm(mean)

    refined = list(clusters)
    for i, (seg, voice) in enumerate(zip(segments, voices)):
        if voice is None or seg.end - seg.start >= ANCHOR_SECONDS:
            continue
        scores = sorted(
            ((float(voice @ profile), cluster) for cluster, profile in profiles.items()),
            reverse=True,
        )
        (best, cluster), (runner_up, _) = scores[0], scores[1]
        if (
            cluster != refined[i]
            and best >= RELABEL_MIN_SIMILARITY
            and best - runner_up >= RELABEL_MARGIN
        ):
            refined[i] = cluster
    return refined


def _cluster_segments(
    segments: list[TranscriptSegment],
    spans: list[SpeakerSpan],
    voices: Optional[list] = None,
) -> list[int]:
    """The diarizer's cluster id for each segment: the span it overlaps most (or, in a gap, the nearest),
    then the short-phrase voice re-check. Shared by whole-recording labelling and per-chunk labelling."""
    clusters: list[int] = []
    for seg in segments:
        best = spans[0].speaker
        best_overlap = 0.0
        best_dist = float("inf")
        for span in spans:
            overlap = min(span.end, seg.end) - max(span.start, seg.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = span.speaker
            elif best_overlap <= 0:
                mid = (seg.start + seg.end) / 2
                dist = max(span.start - mid, mid - span.end, 0.0)
                if dist < best_dist:
                    best_dist = dist
                    best = span.speaker
        clusters.append(best)

    if voices:
        clusters = _refine_clusters(clusters, segments, voices)
    return clusters


def label_segments(
    segments: list[TranscriptSegment],
    spans: list[SpeakerSpan],
    voices: Optional[list] = None,
) -> int:
    """Assign each transcript segment a ``Person N`` speaker label.

    Each segment takes the span it overlaps most; a segment sitting in a
    diarization gap takes the nearest span (the mobile app's
    ``speakersFromNativeSpans`` rule). When ``voices`` (from
    :func:`segment_voices`) is given, short phrases are then re-checked
    against each speaker's voice — see :func:`_refine_clusters`. Labels are
    numbered by order of first appearance, after that re-check, so
    ``Person 1`` always speaks first — that ordering is what keys the
    user's saved ``speaker_names``.

    A one-speaker verdict clears every label (the apps' monologue
    convention: no labels shown for a single voice).

    Returns the number of distinct speakers found.
    """
    if not segments or not spans:
        return 0

    clusters = _cluster_segments(segments, spans, voices)

    order: dict[int, int] = {}
    for cluster in clusters:
        if cluster not in order:
            order[cluster] = len(order) + 1

    if len(order) < 2:
        for seg in segments:
            seg.speaker = None
        return len(order)

    for seg, cluster in zip(segments, clusters):
        seg.speaker = f"Person {order[cluster]}"
    return len(order)


def assign_speakers(segments: list[TranscriptSegment], audio_path: str) -> int:
    """Diarize ``audio_path`` and label ``segments`` in place.

    The one call sites use: returns the speaker count, ``0`` meaning
    "no labels assigned" (diarization unavailable, failed, or monologue).
    """
    diarized = _diarize(audio_path)
    if not diarized or not diarized[0]:
        return 0
    spans, audio, sample_rate = diarized
    voices = segment_voices(segments, audio, sample_rate)
    found = label_segments(segments, spans, voices)
    return found if found >= 2 else 0


# ---------------------------------------------------------------------------
# Long recordings: diarize a part, then tell which of its speakers are which across parts
# ---------------------------------------------------------------------------
#
# A diarizer labels speakers WITHIN what it was given: "speaker 0" in part 3 is not "speaker 0" in part 1.
# So each part is clustered on its own, every local speaker is summarised by a voice fingerprint (the mean of
# the embeddings of their longer phrases, the same anchors the short-phrase re-check uses), and the
# fingerprints are matched across parts by cosine similarity. Matching is greedy and in order, so the global
# numbering is "by first appearance" exactly as for a short recording: Person 1 still speaks first.

#: How alike two fingerprints must be to be the same person. Measured on the Recordings/ fixtures: phrases of
#: two seconds or more score 0.70-0.96 against their own speaker, and the best wrong-speaker score on a short
#: phrase was 0.51. A fingerprint averages several anchors so is steadier than a phrase; 0.55 sits above the
#: wrong-speaker ceiling with margin, and below the same-speaker floor.
LINK_THRESHOLD = 0.55


@dataclass
class ChunkSpeakers:
    """One part's diarization: a local cluster id per segment, and each cluster's voice fingerprint."""

    clusters: list[int]
    #: local cluster id -> unit-length fingerprint, in order of first appearance. A cluster with no phrase
    #: long enough to embed has no entry, and cannot be linked.
    fingerprints: dict[int, "object"]
    #: how many anchor phrases each fingerprint is built from (its weight when merged into a global one)
    weights: dict[int, int]


def diarize_chunk(
    segments: list[TranscriptSegment],
    audio,
    sample_rate: int,
) -> Optional[ChunkSpeakers]:
    """Diarize one part of a long recording from its samples. ``segments`` are relative to the part.

    ``None`` when diarization is unavailable or fails (the part is then left unlabelled; the transcript
    matters more than the labels). Unlike :func:`assign_speakers` this never applies the monologue rule or
    numbers anybody: a part with one speaker is still a speaker, and who "Person 1" is is decided across
    all the parts by :class:`SpeakerLinker`.
    """
    if not segments:
        return None
    diarized = _diarize_samples(audio, sample_rate, "a part of a long recording")
    if not diarized or not diarized[0]:
        return None
    spans, resampled, rate = diarized

    voices = segment_voices(segments, resampled, rate)
    clusters = _cluster_segments(segments, spans, voices)
    if not voices:
        return ChunkSpeakers(clusters=clusters, fingerprints={}, weights={})

    import numpy as np  # noqa: PLC0415

    # Fingerprint from the longer phrases, where the embedding is reliable; fall back to any embeddable
    # phrase for a speaker who only ever said short things in this part.
    anchors: dict[int, list] = {}
    others: dict[int, list] = {}
    for cluster, seg, voice in zip(clusters, segments, voices):
        if voice is None:
            continue
        bucket = anchors if seg.end - seg.start >= ANCHOR_SECONDS else others
        bucket.setdefault(cluster, []).append(voice)

    fingerprints: dict[int, object] = {}
    weights: dict[int, int] = {}
    for cluster in dict.fromkeys(clusters):          # first-appearance order
        vectors = anchors.get(cluster) or others.get(cluster)
        if not vectors:
            continue
        mean = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            fingerprints[cluster] = mean / norm
            weights[cluster] = len(vectors)
    return ChunkSpeakers(clusters=clusters, fingerprints=fingerprints, weights=weights)


#: How alike two fingerprints must be to be the SAME person even when the diarizer said they were two
#: speakers in one part. Diarizers over-split: a host's interjections ("mm", "yes") become a cluster of their
#: own. On the fixtures (3 real voices, 5 parts) those fragments scored 0.81-0.94 against their own speaker
#: while different people never passed 0.59, so 0.78 sits in the gap with room on both sides.
MERGE_THRESHOLD = 0.78


class SpeakerLinker:
    """Gives speakers one identity across the parts of a long recording.

    Feed it each part's :class:`ChunkSpeakers` in order; it answers with a global speaker number for every
    local cluster. Two rules, from how real diarizers behave:

    * A local speaker joins the most similar known speaker once they are ``threshold`` alike, matched
      best-first and one-to-one within a part (the diarizer said these are different people).
    * Except when they are near-identical (``merge_threshold``): then the diarizer split one voice in two and
      the pieces rejoin. Left alone, every short interjection became an extra "person".

    Whoever matches nobody is new; the biggest unmatched cluster is placed first so a small one that sounds
    like it can join it instead of becoming a stranger.
    """

    def __init__(self, threshold: float = LINK_THRESHOLD, merge_threshold: float = MERGE_THRESHOLD) -> None:
        self.threshold = threshold
        self.merge_threshold = merge_threshold
        self._sums: list = []      # running weighted sum of fingerprints, per global speaker

    @property
    def speaker_count(self) -> int:
        return len(self._sums)

    def link(self, part: ChunkSpeakers) -> dict[int, int]:
        """Map each local cluster id in ``part`` to a global speaker index (0-based, in creation order).

        A local cluster with no fingerprint is absent from the result; the caller leaves its segments
        unlabelled rather than guess. Numbering as "Person N" is by first appearance in the transcript, which
        the caller does once everything is linked.
        """
        import numpy as np  # noqa: PLC0415

        # What the known speakers sounded like BEFORE this part: folding this part's matches in first would
        # drag a centroid towards them and let a different voice look like a fragment of it.
        known = [total / np.linalg.norm(total) for total in self._sums]

        def similarity_to(index: int, fingerprint) -> float:
            if index < len(known):
                return float(fingerprint @ known[index])
            total = self._sums[index]                 # created in this part
            return float(fingerprint @ (total / np.linalg.norm(total)))

        def fold(local: int, index: int) -> None:
            self._sums[index] = self._sums[index] + part.fingerprints[local] * part.weights.get(local, 1)

        pairs = sorted(
            (
                (similarity_to(index, fingerprint), local, index)
                for local, fingerprint in part.fingerprints.items()
                for index in range(len(self._sums))
            ),
            reverse=True,
        )
        mapping: dict[int, int] = {}
        taken: set[int] = set()
        for similarity, local, index in pairs:
            if similarity < self.threshold:
                break
            if local in mapping or index in taken:
                continue
            mapping[local] = index
            taken.add(index)
        for local, index in mapping.items():
            fold(local, index)

        # What is left: a near-identical voice is the same person split by the diarizer, anything else is
        # new. Biggest first, so fragments find the main voice rather than the other way round.
        leftover = sorted(
            (local for local in part.fingerprints if local not in mapping),
            key=lambda local: -part.weights.get(local, 1),
        )
        for local in leftover:
            fingerprint = part.fingerprints[local]
            scores = [(similarity_to(index, fingerprint), index) for index in range(len(self._sums))]
            best, best_index = max(scores, default=(-1.0, -1))
            if best >= self.merge_threshold:
                mapping[local] = best_index
            else:
                self._sums.append(np.zeros_like(fingerprint))
                mapping[local] = len(self._sums) - 1
            fold(local, mapping[local])
        return mapping
