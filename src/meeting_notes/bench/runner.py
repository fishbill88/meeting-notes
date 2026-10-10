"""Runs a backend over a dataset and aggregates the scores."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

from meeting_notes.backends.base import TranscriptionBackend
from meeting_notes.bench.datasets import Dataset, Utterance
from meeting_notes.bench.metrics import (
    BenchmarkMetrics,
    character_error_rate,
    term_recall,
    word_error_rate,
)


@dataclass
class UtteranceResult:
    """Per-item detail, kept so failures can be inspected afterwards."""

    id: str
    reference: str
    hypothesis: str
    wer: float
    duration: Optional[float] = None
    compute_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class BenchmarkRun:
    """Everything one (backend, dataset) pairing produced."""

    metrics: BenchmarkMetrics
    utterances: list[UtteranceResult] = field(default_factory=list)

    def worst(self, n: int = 10) -> list[UtteranceResult]:
        """The n highest-WER utterances — where to look first."""
        scored = [u for u in self.utterances if u.error is None]
        return sorted(scored, key=lambda u: u.wer, reverse=True)[:n]

    def to_dict(self) -> dict:
        metrics = self.metrics
        return {
            "backend": metrics.backend,
            "dataset": metrics.dataset,
            "language": metrics.language,
            "utterances": metrics.utterances,
            "failures": metrics.failures,
            "wer": metrics.wer,
            "cer": metrics.cer,
            "rtf": metrics.rtf,
            "speed_factor": metrics.speed_factor,
            "term_recall": metrics.term_recall,
            "audio_seconds": metrics.audio_seconds,
            "compute_seconds": metrics.compute_seconds,
            "detail": [asdict(u) for u in self.utterances],
        }


def run_benchmark(
    backend: TranscriptionBackend,
    dataset: Dataset,
    terms: Optional[Sequence[str]] = None,
    on_progress: Optional[Callable[[int, UtteranceResult], None]] = None,
    keep_detail: bool = True,
) -> BenchmarkRun:
    """Transcribe every utterance in ``dataset`` and score the results.

    The model is warmed up before timing starts, so the reported RTF measures
    steady-state throughput rather than a one-off model load. That matters:
    including a 30-second load in a 400-utterance run would understate speed,
    and including it in a 5-utterance smoke run would wildly overstate cost.

    A failing utterance is recorded and skipped rather than aborting the run —
    one unreadable file should not cost you a 400-item benchmark.

    Args:
        backend: The engine under test.
        dataset: What to run it over.
        terms: Terms to score recall against, e.g. drug names.
        on_progress: Called with ``(index, result)`` after each utterance.
        keep_detail: Retain per-utterance rows. Turn off for very large runs.

    Returns:
        A :class:`BenchmarkRun` with aggregate metrics and per-item detail.
    """
    terms = list(terms or [])
    metrics = BenchmarkMetrics(
        backend=backend.name,
        dataset=dataset.name,
        language=dataset.language,
    )
    run = BenchmarkRun(metrics=metrics)

    backend.warmup()

    for index, utterance in enumerate(dataset):
        result = _score_one(backend, utterance, terms, metrics)

        if keep_detail:
            run.utterances.append(result)
        if on_progress is not None:
            on_progress(index, result)

    return run


def _score_one(
    backend: TranscriptionBackend,
    utterance: Utterance,
    terms: Sequence[str],
    metrics: BenchmarkMetrics,
) -> UtteranceResult:
    """Transcribe and score a single utterance, accumulating into ``metrics``."""
    duration = utterance.resolved_duration()

    started = time.perf_counter()
    try:
        if utterance.audio_array is not None:
            transcription = backend.transcribe_array(
                utterance.audio_array, utterance.sample_rate
            )
        elif utterance.audio_path:
            transcription = backend.transcribe_file(utterance.audio_path)
        else:
            raise ValueError("utterance has neither audio_array nor audio_path")
    except Exception as exc:  # noqa: BLE001 – one bad file must not end the run
        metrics.failures += 1
        return UtteranceResult(
            id=utterance.id,
            reference=utterance.reference,
            hypothesis="",
            wer=1.0,
            duration=duration,
            compute_seconds=time.perf_counter() - started,
            error=str(exc),
        )

    elapsed = time.perf_counter() - started
    hypothesis = transcription.text

    word_counts = word_error_rate(utterance.reference, hypothesis)
    char_counts = character_error_rate(utterance.reference, hypothesis)

    metrics.utterances += 1
    metrics.word_counts = metrics.word_counts + word_counts
    metrics.char_counts = metrics.char_counts + char_counts
    metrics.compute_seconds += elapsed
    if duration:
        metrics.audio_seconds += duration

    if terms:
        found, expected = term_recall(utterance.reference, hypothesis, terms)
        metrics.terms_found += found
        metrics.terms_expected += expected

    return UtteranceResult(
        id=utterance.id,
        reference=utterance.reference,
        hypothesis=hypothesis,
        wer=word_counts.rate,
        duration=duration,
        compute_seconds=elapsed,
    )
