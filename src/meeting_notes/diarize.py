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
    diarizer = _load_diarizer()
    if diarizer is None:
        return None

    try:
        import numpy as np  # noqa: PLC0415

        from meeting_notes.audio import load_audio_file  # noqa: PLC0415
        from meeting_notes.backends.base import resample  # noqa: PLC0415

        audio, sample_rate = load_audio_file(audio_path)
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
        logger.exception("diarization failed for %s", audio_path)
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
