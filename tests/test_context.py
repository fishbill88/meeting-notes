"""Tests for the host's meeting context: parsing it, the hint Whisper is given, and dropping a hint read back."""

from __future__ import annotations

import json

import pytest

from meeting_notes.context import (
    MAX_HINT_CHARS,
    MAX_ITEM_CHARS,
    MAX_NAMES,
    TranscriptionContext,
    hint_text,
    is_echo,
    parse_context,
)


def _ctx(**kw) -> str:
    return json.dumps(kw)


# ---------------------------------------------------------------------------
# parse_context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_no_field_is_no_context(raw):
    assert parse_context(raw) is None


def test_an_object_that_says_nothing_is_no_context():
    assert parse_context(_ctx(title="  ", names=[], terms=["", "​"])) is None


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '"a string"', "42"])
def test_anything_but_a_json_object_is_refused(raw):
    with pytest.raises(ValueError, match="JSON object"):
        parse_context(raw)


def test_title_names_and_terms_are_read():
    ctx = parse_context(_ctx(title="Sprint sync", names=["Ana Cruz", "Marco Reyes"], terms=["UAT"]))
    assert ctx == TranscriptionContext(title="Sprint sync", names=("Ana Cruz", "Marco Reyes"), terms=("UAT",))


def test_items_of_the_wrong_type_are_skipped_not_refused():
    ctx = parse_context(_ctx(title=7, names=["Ana", 3, None, {"x": 1}], terms="UAT"))
    assert ctx == TranscriptionContext(title=None, names=("Ana",), terms=())


def test_unspeakable_characters_go_and_whitespace_collapses():
    # A control character, a zero-width space, an emoji and a run of spaces and tabs.
    ctx = parse_context(_ctx(title="IT\x07 Dept \U0001F525", names=["Ana​  \t Cruz"]))
    assert ctx.title == "IT Dept"
    assert ctx.names == ("Ana Cruz",)


def test_accented_and_non_latin_names_survive():
    ctx = parse_context(_ctx(names=["José Peñaflorida", "Ñino", "王小明"]))
    assert ctx.names == ("José Peñaflorida", "Ñino", "王小明")


def test_repeats_keep_the_first_spelling_ignoring_case():
    ctx = parse_context(_ctx(names=["Ana Cruz", "ana cruz", "ANA CRUZ", "Marco"]))
    assert ctx.names == ("Ana Cruz", "Marco")


def test_a_long_item_is_cut():
    ctx = parse_context(_ctx(terms=["x" * 500]))
    assert len(ctx.terms[0]) == MAX_ITEM_CHARS


def test_too_many_names_are_capped():
    ctx = parse_context(_ctx(names=[f"Person {i}" for i in range(100)]))
    assert len(ctx.names) == MAX_NAMES
    assert ctx.names[0] == "Person 0"


# ---------------------------------------------------------------------------
# hint_text
# ---------------------------------------------------------------------------


def test_no_context_is_no_hint():
    assert hint_text(None) is None
    assert hint_text(TranscriptionContext()) is None


def test_the_hint_reads_title_then_names_then_terms():
    ctx = TranscriptionContext(title="Sprint sync", names=("Ana Cruz", "Marco Reyes"), terms=("UAT",))
    assert hint_text(ctx) == "Sprint sync. Ana Cruz, Marco Reyes, UAT."


def test_a_title_alone_or_names_alone_are_fine():
    assert hint_text(TranscriptionContext(title="Sprint sync.")) == "Sprint sync."
    assert hint_text(TranscriptionContext(names=("Ana",))) == "Ana."


def test_a_name_that_is_the_title_is_not_said_twice():
    # A direct call is titled by the other person's name.
    ctx = TranscriptionContext(title="Ana Cruz", names=("ana cruz", "Marco Reyes"), terms=("Marco Reyes",))
    assert hint_text(ctx) == "Ana Cruz. Marco Reyes."


def test_the_budget_drops_terms_before_names():
    names = tuple(f"Person Number {i}" for i in range(10))
    ctx = TranscriptionContext(title="Sprint sync", names=names, terms=("UAT", "ERP"))
    hint = hint_text(ctx, budget=120)
    assert len(hint) <= 120
    assert hint.startswith("Sprint sync. Person Number 0, ")
    assert "UAT" not in hint
    assert hint.endswith(".")


def test_the_default_budget_holds_for_the_largest_context():
    ctx = parse_context(_ctx(
        title="t" * 500,
        names=[f"{'n' * 50} {i}" for i in range(100)],
        terms=[f"{'w' * 50} {i}" for i in range(100)],
    ))
    assert len(hint_text(ctx)) <= MAX_HINT_CHARS


# ---------------------------------------------------------------------------
# Echoes
# ---------------------------------------------------------------------------

HINT = "Sprint sync. Ana Cruz, Marco Reyes, Joy Santos, UAT."


def test_the_whole_hint_read_back_is_an_echo():
    assert is_echo(" Sprint sync. Ana Cruz, Marco Reyes, Joy Santos, UAT.", HINT)


def test_a_long_run_of_the_hint_is_an_echo_whatever_the_punctuation():
    assert is_echo("ana cruz marco reyes joy santos", HINT)


def test_speech_that_mentions_names_is_not_an_echo():
    assert not is_echo("Ana, can you send the build?", HINT)
    assert not is_echo("Sprint sync, Ana Cruz", HINT)          # a short run of the hint
    assert not is_echo("Ana Cruz and Marco Reyes and Joy", HINT)  # not a run: other words between


def test_a_short_hint_never_drops_anything():
    assert not is_echo("Ana Cruz", "Ana Cruz.")


def test_a_hint_of_three_words_drops_only_itself():
    assert is_echo("Ana Cruz, UAT.", "Ana Cruz, UAT.")
    assert not is_echo("Ana Cruz", "Ana Cruz, UAT.")


def test_no_hint_is_never_an_echo():
    assert not is_echo("Sprint sync. Ana Cruz, Marco Reyes, Joy Santos, UAT.", None)
