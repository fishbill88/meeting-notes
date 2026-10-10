"""Tests for the benchmark harness."""

from __future__ import annotations

import json

import pytest

from meeting_notes.bench.datasets import Utterance, manifest_dataset
from meeting_notes.bench.metrics import (
    BenchmarkMetrics,
    ErrorCounts,
    character_error_rate,
    term_recall,
    word_error_rate,
)
from meeting_notes.bench.normalize import normalize_text, tokenize
from meeting_notes.bench.report import render_table
from meeting_notes.bench.runner import run_benchmark


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalize_folds_case_and_strips_punctuation():
    assert normalize_text("Hello, World!") == "hello world"


def test_normalize_collapses_whitespace():
    assert normalize_text("  a   b \n c ") == "a b c"


def test_normalize_preserves_tagalog_affix_hyphen():
    # The whole point: nag-meeting is ONE token. Splitting it would score
    # "nag meeting" as correct and hide the failure mode we care about.
    assert normalize_text("Nag-meeting kami.") == "nag-meeting kami"
    assert tokenize("Nag-meeting kami.") == ["nag-meeting", "kami"]


def test_normalize_preserves_apostrophes():
    assert normalize_text("don't") == "don't"


def test_normalize_drops_dangling_hyphens():
    assert normalize_text("a — b") == "a b"


def test_normalize_handles_empty():
    assert normalize_text("") == ""
    assert tokenize("") == []


def test_normalize_is_unicode_stable():
    # NFKC folds typographic variants that differ between dataset sources.
    assert normalize_text("ﬁle") == "file"


# ---------------------------------------------------------------------------
# WER / CER
# ---------------------------------------------------------------------------


def test_wer_perfect_match_is_zero():
    assert word_error_rate("the cat sat", "the cat sat").rate == 0.0


def test_wer_counts_substitution():
    counts = word_error_rate("the cat sat", "the dog sat")
    assert counts.substitutions == 1
    assert counts.rate == pytest.approx(1 / 3)


def test_wer_counts_deletion():
    counts = word_error_rate("the cat sat", "the sat")
    assert counts.deletions == 1
    assert counts.insertions == 0


def test_wer_counts_insertion():
    counts = word_error_rate("the cat sat", "the big cat sat")
    assert counts.insertions == 1
    assert counts.deletions == 0


def test_wer_ignores_case_and_punctuation():
    assert word_error_rate("The cat, sat.", "the cat sat").rate == 0.0


def test_wer_empty_hypothesis_is_total_loss():
    assert word_error_rate("the cat sat", "").rate == 1.0


def test_wer_empty_reference_with_output_is_capped():
    assert word_error_rate("", "spurious words").rate == 1.0


def test_wer_empty_both_is_zero():
    assert word_error_rate("", "").rate == 0.0


def test_wer_penalises_split_affix():
    # An engine that drops the Tagalog prefix must be scored as wrong.
    assert word_error_rate("nag-meeting kami", "nag meeting kami").rate > 0


def test_cer_is_finer_grained_than_wer():
    # One wrong letter: a whole word by WER, one character by CER.
    wer = word_error_rate("kumusta", "kumustu").rate
    cer = character_error_rate("kumusta", "kumustu").rate
    assert wer == 1.0
    assert 0 < cer < wer


def test_error_counts_add():
    total = ErrorCounts(1, 2, 3, 10) + ErrorCounts(1, 1, 1, 10)
    assert total.substitutions == 2
    assert total.errors == 9
    assert total.reference_length == 20


# ---------------------------------------------------------------------------
# Term recall
# ---------------------------------------------------------------------------


def test_term_recall_finds_present_terms():
    found, expected = term_recall(
        "patient has chest pain and fever", "patient has chest pain and fever",
        ["fever", "chest pain"],
    )
    assert (found, expected) == (2, 2)


def test_term_recall_detects_dropped_term():
    found, expected = term_recall(
        "give amoxicillin now", "give now", ["amoxicillin"]
    )
    assert (found, expected) == (0, 1)


def test_term_recall_ignores_terms_not_in_reference():
    # Only terms actually said are scored — no credit or penalty otherwise.
    found, expected = term_recall("has fever", "has fever", ["ibuprofen"])
    assert expected == 0


def test_term_recall_counts_repeats():
    found, expected = term_recall(
        "fever and fever", "fever and fever", ["fever"]
    )
    assert (found, expected) == (2, 2)


def test_term_recall_does_not_over_credit():
    found, expected = term_recall("fever", "fever fever fever", ["fever"])
    assert found == 1


def test_term_recall_is_case_insensitive():
    found, expected = term_recall("Has Fever", "has fever", ["FEVER"])
    assert (found, expected) == (1, 1)


def test_term_recall_matches_whole_words_only():
    found, expected = term_recall("he is feverish", "he is feverish", ["fever"])
    assert expected == 0


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def test_rtf_and_speed_factor_are_inverses():
    m = BenchmarkMetrics(audio_seconds=100.0, compute_seconds=25.0)
    assert m.rtf == 0.25
    assert m.speed_factor == 4.0


def test_rtf_is_none_without_durations():
    assert BenchmarkMetrics(compute_seconds=5.0).rtf is None


def test_term_recall_property_is_none_without_terms():
    assert BenchmarkMetrics().term_recall is None


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path, rows) -> str:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return str(path)


def test_manifest_loads_rows(tmp_path, wav_file):
    path = _write_manifest(
        tmp_path,
        [
            {"audio": wav_file, "text": "Nag-reklamo siya ng chest pain."},
            {"audio": wav_file, "text": "Ubo at lagnat po.", "id": "two"},
        ],
    )
    items = list(manifest_dataset(path))
    assert len(items) == 2
    assert items[1].id == "two"
    assert items[0].reference.startswith("Nag-reklamo")


def test_manifest_resolves_relative_paths(tmp_path):
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "a.wav").write_bytes(b"")
    path = _write_manifest(tmp_path, [{"audio": "clips/a.wav", "text": "hi"}])

    item = next(iter(manifest_dataset(path)))
    assert item.audio_path == str(tmp_path / "clips" / "a.wav")


def test_manifest_skips_blank_and_comment_lines(tmp_path, wav_file):
    path = tmp_path / "m.jsonl"
    path.write_text(
        "# a comment\n\n" + json.dumps({"audio": wav_file, "text": "hi"}) + "\n",
        encoding="utf-8",
    )
    assert len(list(manifest_dataset(str(path)))) == 1


def test_manifest_respects_limit(tmp_path, wav_file):
    path = _write_manifest(
        tmp_path, [{"audio": wav_file, "text": f"line {i}"} for i in range(5)]
    )
    assert len(list(manifest_dataset(path, limit=2))) == 2


def test_manifest_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        list(manifest_dataset(str(tmp_path / "nope.jsonl")))


def test_fleurs_addresses_the_parquet_export_directly():
    # Regression guard: load_dataset('google/fleurs', 'fil_ph') hangs because
    # the repo has no loading script and ~1000 files to resolve. The parquet
    # path must stay.
    from meeting_notes.bench.datasets import fleurs_parquet_url

    url = fleurs_parquet_url("fil_ph")
    assert url.startswith("hf://datasets/google/fleurs/parquet-data/")
    assert url.endswith("test-00000-of-00001.parquet")


def test_named_datasets_declare_language_and_description():
    from meeting_notes.bench.datasets import DATASET_SPECS

    for name, spec in DATASET_SPECS.items():
        assert spec["language"] in ("en", "tl", "ceb"), name
        assert spec["description"], name


def test_fleurs_datasets_warn_about_download_size():
    from meeting_notes.bench.datasets import get_dataset

    assert get_dataset("fleurs-fil").download_mb > 500


def test_unknown_dataset_name_raises():
    from meeting_notes.bench.datasets import get_dataset

    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset("fleurs-cebuano")


def test_utterance_derives_duration_from_array():
    import numpy as np

    utt = Utterance(
        id="1", reference="x", audio_array=np.zeros(32000), sample_rate=16000
    )
    assert utt.resolved_duration() == 2.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_runner_scores_a_dataset(tmp_path, wav_file, fake_backend):
    path = _write_manifest(
        tmp_path,
        [{"audio": wav_file, "text": "This is a test transcript.", "duration": 2.0}],
    )
    run = run_benchmark(fake_backend, manifest_dataset(path))

    assert run.metrics.utterances == 1
    assert run.metrics.wer == 0.0
    assert run.metrics.rtf is not None


def test_runner_records_failures_without_aborting(tmp_path, wav_file):
    from tests.conftest import FakeBackend

    class Flaky(FakeBackend):
        def transcribe_file(self, audio_path):
            if "bad" in audio_path:
                raise RuntimeError("corrupt file")
            return super().transcribe_file(audio_path)

    path = _write_manifest(
        tmp_path,
        [
            {"audio": wav_file, "text": "This is a test transcript."},
            {"audio": "bad.wav", "text": "never mind"},
        ],
    )
    run = run_benchmark(Flaky(), manifest_dataset(path))

    assert run.metrics.failures == 1
    assert run.metrics.utterances == 1  # the good one still scored
    assert any(u.error for u in run.utterances)


def test_runner_reports_progress(tmp_path, wav_file, fake_backend):
    path = _write_manifest(
        tmp_path, [{"audio": wav_file, "text": "hi"} for _ in range(3)]
    )
    seen = []
    run_benchmark(
        fake_backend, manifest_dataset(path), on_progress=lambda i, r: seen.append(i)
    )
    assert seen == [0, 1, 2]


def test_runner_worst_ranks_by_wer(tmp_path, wav_file, fake_backend):
    path = _write_manifest(
        tmp_path,
        [
            {"audio": wav_file, "text": "This is a test transcript."},  # perfect
            {"audio": wav_file, "text": "completely different words here"},
        ],
    )
    run = run_benchmark(fake_backend, manifest_dataset(path))
    worst = run.worst(1)
    assert worst[0].reference.startswith("completely")


def test_runner_scores_term_recall(tmp_path, wav_file, fake_backend):
    path = _write_manifest(
        tmp_path, [{"audio": wav_file, "text": "This is a test transcript."}]
    )
    run = run_benchmark(fake_backend, manifest_dataset(path), terms=["test"])
    assert run.metrics.term_recall == 1.0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_render_table_includes_backend_and_wer(tmp_path, wav_file, fake_backend):
    path = _write_manifest(
        tmp_path, [{"audio": wav_file, "text": "This is a test transcript."}]
    )
    table = render_table([run_benchmark(fake_backend, manifest_dataset(path))])

    assert "fake" in table
    assert "0.00%" in table
    assert table.startswith("| Backend |")
