"""Benchmark harness for comparing transcription backends.

Answers one question: which engine should ship for which language, measured
rather than assumed. Published vendor numbers are a starting point, not
evidence — this reproduces them on the same public test sets.

Layout::

    normalize.py  — text normalisation (WER is meaningless without it)
    metrics.py    — WER, CER, term recall, RTF
    datasets.py   — FLEURS / Common Voice / local JSONL manifests
    runner.py     — run one backend over one dataset
    report.py     — markdown results table

Run it via ``meeting-notes bench``.
"""

from __future__ import annotations

from meeting_notes.bench.metrics import (
    BenchmarkMetrics,
    character_error_rate,
    term_recall,
    word_error_rate,
)
from meeting_notes.bench.normalize import normalize_text

__all__ = [
    "BenchmarkMetrics",
    "character_error_rate",
    "normalize_text",
    "term_recall",
    "word_error_rate",
]
