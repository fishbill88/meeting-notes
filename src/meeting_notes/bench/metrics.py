"""Scoring metrics.

Four numbers, because one is not enough to choose a clinical engine:

* **WER** — the standard, and the one to compare against published figures.
* **CER** — needed to compare against papers that report it (Qwen3-ASR does).
  Not interchangeable with WER: CER is normally the more forgiving of the two,
  so a 24% CER is *worse* than a 9% WER, not better.
* **Term recall** — of the terms that must not be missed (drug names,
  conditions), how many survived? Decides clinical usability, and WER
  routinely disagrees with it: an engine can score well overall while dropping
  every drug name.
* **RTF** — seconds of compute per second of audio. Decides whether live
  captions are possible at all.

Implemented with plain Levenshtein rather than a dependency, matching the
project's habit of keeping the scoring path free of optional installs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from meeting_notes.bench.normalize import normalize_text, tokenize


@dataclass
class ErrorCounts:
    """Edit operations between a reference and a hypothesis."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Error rate. Returns 0.0 for an empty reference with empty output."""
        if self.reference_length == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.reference_length

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_length=self.reference_length + other.reference_length,
        )


@dataclass
class BenchmarkMetrics:
    """Aggregate results for one backend over one dataset."""

    backend: str = ""
    dataset: str = ""
    language: str = ""
    utterances: int = 0
    word_counts: ErrorCounts = field(default_factory=ErrorCounts)
    char_counts: ErrorCounts = field(default_factory=ErrorCounts)
    audio_seconds: float = 0.0
    compute_seconds: float = 0.0
    terms_expected: int = 0
    terms_found: int = 0
    failures: int = 0

    @property
    def wer(self) -> float:
        return self.word_counts.rate

    @property
    def cer(self) -> float:
        return self.char_counts.rate

    @property
    def rtf(self) -> Optional[float]:
        """Real-time factor: compute seconds per audio second.

        Below 1.0 is faster than real time. Live captioning needs comfortably
        below 1.0, since the figure here excludes UI and audio-capture cost.
        """
        if self.audio_seconds <= 0:
            return None
        return self.compute_seconds / self.audio_seconds

    @property
    def speed_factor(self) -> Optional[float]:
        """How many times faster than real time. The inverse of :attr:`rtf`."""
        rtf = self.rtf
        if not rtf:
            return None
        return 1.0 / rtf

    @property
    def term_recall(self) -> Optional[float]:
        if self.terms_expected == 0:
            return None
        return self.terms_found / self.terms_expected


def _levenshtein(reference: Sequence, hypothesis: Sequence) -> ErrorCounts:
    """Count edit operations aligning ``hypothesis`` to ``reference``.

    Two-row dynamic programme carrying (substitutions, deletions, insertions)
    alongside the cost, so the breakdown survives — a high deletion count means
    something different (dropped audio, VAD eating speech) from a high
    substitution count (acoustic confusion).
    """
    ref_len, hyp_len = len(reference), len(hypothesis)

    if ref_len == 0:
        return ErrorCounts(insertions=hyp_len, reference_length=0)
    if hyp_len == 0:
        return ErrorCounts(deletions=ref_len, reference_length=ref_len)

    # Each cell: (cost, subs, dels, ins)
    previous = [(j, 0, 0, j) for j in range(hyp_len + 1)]

    for i in range(1, ref_len + 1):
        current = [(i, 0, i, 0)]
        for j in range(1, hyp_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost, subs, dels, ins = previous[j - 1]
                current.append((cost, subs, dels, ins))
                continue

            sub_cost, sub_s, sub_d, sub_i = previous[j - 1]
            del_cost, del_s, del_d, del_i = previous[j]
            ins_cost, ins_s, ins_d, ins_i = current[j - 1]

            best = min(sub_cost, del_cost, ins_cost)
            if best == sub_cost:
                current.append((best + 1, sub_s + 1, sub_d, sub_i))
            elif best == del_cost:
                current.append((best + 1, del_s, del_d + 1, del_i))
            else:
                current.append((best + 1, ins_s, ins_d, ins_i + 1))

        previous = current

    _, subs, dels, ins = previous[hyp_len]
    return ErrorCounts(
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        reference_length=ref_len,
    )


def word_error_rate(reference: str, hypothesis: str) -> ErrorCounts:
    """Word-level edit counts between two strings, after normalisation."""
    return _levenshtein(tokenize(reference), tokenize(hypothesis))


def character_error_rate(reference: str, hypothesis: str) -> ErrorCounts:
    """Character-level edit counts, after normalisation.

    Spaces are kept: they carry word-boundary information, which is exactly
    what a code-switching model tends to get wrong.
    """
    return _levenshtein(
        list(normalize_text(reference)),
        list(normalize_text(hypothesis)),
    )


def term_recall(reference: str, hypothesis: str, terms: Sequence[str]) -> tuple[int, int]:
    """Count how many required terms present in the reference survived.

    Args:
        reference: Ground-truth text.
        hypothesis: Engine output.
        terms: Terms that must not be lost — drug names, conditions.

    Returns:
        ``(found, expected)``. Terms absent from the reference are not counted
        against the engine; only terms that were genuinely said are scored.
    """
    ref = f" {normalize_text(reference)} "
    hyp = f" {normalize_text(hypothesis)} "

    expected = 0
    found = 0
    for term in terms:
        needle = normalize_text(term)
        if not needle:
            continue
        occurrences = ref.count(f" {needle} ")
        if occurrences == 0:
            continue
        expected += occurrences
        found += min(occurrences, hyp.count(f" {needle} "))

    return found, expected
