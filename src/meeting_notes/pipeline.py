"""The shared transcribe → summarize pipeline.

Both the CLI and the REST server call :func:`run_pipeline`. Keeping one
implementation here is what stops them drifting apart — before this module
they each had their own copy, and the server's had already lost the CLI's
language handling.

This function is deliberately free of terminal output and process exits so it
is safe to call from a server request handler, a test, or the desktop sidecar.
Progress is reported through an optional callback instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Optional

from meeting_notes.backends.base import TranscriptionResult
from meeting_notes.cancel import PipelineCancelled
from meeting_notes.config import AppConfig
from meeting_notes.summarizer import MeetingNotes, Summarizer

#: ``(stage, message)`` — stage is one of ``"transcribe"``, ``"diarize"``,
#: or ``"summarize"``.
ProgressCallback = Callable[[str, str], None]

logger = logging.getLogger("meeting_notes.pipeline")


@dataclass
class PipelineResult:
    """Everything the pipeline produced, ready to render or persist."""

    transcript: TranscriptionResult
    notes: MeetingNotes

    @property
    def backend(self) -> str:
        """Which engine actually ran.

        Worth surfacing in the UI: the Tagalog router falls back to Whisper
        when the NeMo model has not been exported, and a clinician should be
        able to see that rather than silently getting the weaker engine.
        """
        return self.transcript.backend

    @property
    def timed_transcript(self) -> str:
        """Timestamped transcript, falling back to plain text."""
        return self.transcript.timed_transcript or self.transcript.text


#: Share of the overall progress bar that transcribing and diarizing take; the rest is writing the notes.
#: A long meeting's notes (one model call per part, then a merge) are a real fraction of the total.
_AUDIO_SHARE = 0.85

#: Of one part's share, how much is transcribing (the rest is diarizing it).
_TRANSCRIBE_SHARE = 0.7


def run_pipeline(
    audio_path: str,
    config: AppConfig,
    on_progress: Optional[ProgressCallback] = None,
    on_fraction: Optional[Callable[[float], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> PipelineResult:
    """Transcribe ``audio_path`` and generate structured meeting notes.

    A recording longer than ``chunking.CHUNK_THRESHOLD_SECONDS`` is processed in parts cut at pauses
    (see ``chunking.py``), which gives progress, bounded memory, and cancellation between parts. A shorter
    one goes through the single pass it always did.

    Args:
        audio_path: Path to an audio file in any format the backend accepts.
        config: Validated app configuration. Its ``language_mode`` selects
            the engine.
        on_progress: Optional ``(stage, message)`` callback for UI feedback.
        on_fraction: Optional callback given overall progress, 0-1. Only called for a recording that is
            processed in parts; a single pass has no measure of how far through it is.
        should_cancel: Optional callback polled between steps; true raises
            :class:`~meeting_notes.cancel.PipelineCancelled`.

    Returns:
        :class:`PipelineResult` with the transcript and the notes.

    Raises:
        NotImplementedError: If ``config.language_mode`` is Taglish.
        BackendUnavailableError: If the selected engine cannot run.
        PipelineCancelled: If ``should_cancel`` turned true.
    """
    from meeting_notes.backends.router import resolve_backend  # noqa: PLC0415
    from meeting_notes.chunking import needs_chunking, probe_duration  # noqa: PLC0415

    report = on_progress or (lambda stage, message: None)
    fraction = on_fraction or (lambda value: None)
    cancelled = should_cancel or (lambda: False)

    backend = resolve_backend(config)
    duration = probe_duration(audio_path)
    chunked = needs_chunking(duration)

    if chunked:
        transcript = _transcribe_in_parts(
            audio_path, backend, config, report, fraction, cancelled
        )
    else:
        report("transcribe", f"Transcribing with {backend.name}…")
        transcript = backend.transcribe_file(audio_path)

        if config.diarize and transcript.segments:
            from meeting_notes.diarize import assign_speakers  # noqa: PLC0415

            report("diarize", "Identifying speakers…")
            assign_speakers(transcript.segments, audio_path)

    if cancelled():
        raise PipelineCancelled()

    # The bundled model when it is installed (use_ollama=False still means "no model at all").
    provider = None
    if config.use_ollama:
        from meeting_notes.llm import default_provider  # noqa: PLC0415

        provider = default_provider()
    engine = "the bundled model" if provider else "Ollama" if config.use_ollama else "extractive"
    report("summarize", f"Writing the notes ({engine})…")

    def on_part(done: int, total: int) -> None:
        report("summarize", f"Writing the notes ({engine}), part {done} of {total}…")
        fraction(_AUDIO_SHARE + (1 - _AUDIO_SHARE) * done / max(total, 1) * 0.9)

    summarizer = Summarizer(
        use_ollama=config.use_ollama,
        ollama_url=config.ollama_url,
        ollama_model=config.ollama_model,
        provider=provider,
        on_part=on_part if chunked else None,
        should_cancel=cancelled,
    )
    from meeting_notes.summarizer import speaker_text  # noqa: PLC0415

    notes = summarizer.summarize(
        transcript.text,
        language=transcript.language,
        template=config.template,
        speaker_text=speaker_text(transcript.segments),
    )

    fraction(1.0)
    return PipelineResult(transcript=transcript, notes=notes)


def _transcribe_in_parts(
    audio_path: str,
    backend,
    config: AppConfig,
    report: ProgressCallback,
    fraction: Callable[[float], None],
    cancelled: Callable[[], bool],
) -> TranscriptionResult:
    """Transcribe and diarize a long recording one part at a time, then stitch the parts together.

    The recording is decoded ONCE; each part is a slice of that array, so there is no temporary file per
    part and nothing is read twice. Speakers are labelled within each part and then given one identity
    across parts by their voice (``diarize.SpeakerLinker``), so "Person 2" is the same person in minute 5 and
    minute 55.
    """
    from collections import Counter  # noqa: PLC0415

    from meeting_notes.audio import load_audio_file  # noqa: PLC0415
    from meeting_notes.backends.base import language_name  # noqa: PLC0415
    from meeting_notes.chunking import plan_chunks  # noqa: PLC0415

    report("transcribe", "Reading the recording…")
    audio, sample_rate = load_audio_file(audio_path)
    chunks = plan_chunks(audio, sample_rate)
    total_seconds = len(audio) / float(sample_rate)
    parts = len(chunks)
    logger.info("long recording: %.0f s in %d parts", total_seconds, parts)

    linker = None
    if config.diarize:
        from meeting_notes.diarize import SpeakerLinker  # noqa: PLC0415

        linker = SpeakerLinker()

    segments: list = []
    languages: Counter = Counter()
    # Placeholder labels ("#<global index>") until every part is done; see the renumbering below.
    for chunk in chunks:
        if cancelled():
            raise PipelineCancelled()

        done_before = chunk.start / total_seconds
        span = chunk.duration / total_seconds
        label = f"part {chunk.index + 1} of {parts}"
        piece = audio[int(chunk.start * sample_rate):int(chunk.end * sample_rate)]

        report("transcribe", f"Transcribing {label} with {backend.name}…")
        result = _transcribe_piece(backend, piece, sample_rate, label)
        fraction(_AUDIO_SHARE * (done_before + span * _TRANSCRIBE_SHARE))

        # One language for the whole recording, as a single pass would have: Whisper detects from the start
        # of what it is given, and a part that opens with silence or music would otherwise flip it.
        if (
            chunk.index == 0
            and result.segments
            and config.language_mode.value == "auto"
            and getattr(backend, "language", "x") is None
        ):
            backend.language = result.language

        if result.segments:
            languages[result.language] += chunk.duration

        if linker is not None and result.segments:
            if cancelled():
                raise PipelineCancelled()
            report("diarize", f"Identifying speakers in {label}…")
            _label_part(linker, result.segments, piece, sample_rate)

        for seg in result.segments:
            seg.start += chunk.start
            seg.end += chunk.start
        segments.extend(result.segments)
        fraction(_AUDIO_SHARE * (done_before + span))

    _renumber_speakers(segments)

    language = languages.most_common(1)[0][0] if languages else "unknown"
    return TranscriptionResult(
        text=" ".join(s.text.strip() for s in segments).strip(),
        language=language,
        language_name=language_name(language),
        segments=segments,
        backend=backend.name,
    )


def _transcribe_piece(backend, piece, sample_rate: int, label: str):
    """Transcribe one part, retrying once. A part that fails twice fails the recording, by name: dropping
    five minutes of a meeting silently and carrying on would be a worse answer than an error."""
    for attempt in (1, 2):
        try:
            return backend.transcribe_array(piece, sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed (attempt %d): %s", label, attempt, exc)
            if attempt == 2:
                raise RuntimeError(f"Couldn't transcribe {label}: {exc}") from exc
    raise AssertionError("unreachable")


def _label_part(linker, segments: list, piece, sample_rate: int) -> None:
    """Diarize one part and write global speaker placeholders onto its segments.

    A part whose diarization is unavailable, or a speaker with nothing long enough to fingerprint, is left
    unlabelled rather than guessed at.
    """
    from meeting_notes.diarize import diarize_chunk  # noqa: PLC0415

    part = diarize_chunk(segments, piece, sample_rate)
    if part is None or not part.fingerprints:
        return
    mapping = linker.link(part)
    for seg, cluster in zip(segments, part.clusters):
        index = mapping.get(cluster)
        if index is not None:
            seg.speaker = f"#{index}"


def _renumber_speakers(segments: list) -> None:
    """Turn the placeholders into ``Person N``, numbered by first appearance across the WHOLE recording
    (so Person 1 speaks first, which is what keys a person's saved speaker names), and apply the monologue
    rule: one voice means no labels at all."""
    order: dict[str, int] = {}
    for seg in segments:
        if seg.speaker and seg.speaker not in order:
            order[seg.speaker] = len(order) + 1

    if len(order) < 2:
        for seg in segments:
            seg.speaker = None
        return
    for seg in segments:
        if seg.speaker:
            seg.speaker = f"Person {order[seg.speaker]}"
