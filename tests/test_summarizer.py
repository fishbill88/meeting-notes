"""Tests for the summarizer module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from meeting_notes.summarizer import (
    MeetingNotes,
    Summarizer,
    _build_prompt,
    _derive_title,
    _first_n_words,
    _parse_llm_response,
    _score_sentences,
    _split_sentences,
)


# ---------------------------------------------------------------------------
# MeetingNotes – Markdown rendering
# ---------------------------------------------------------------------------


def test_to_markdown_basic():
    notes = MeetingNotes(
        title="Q1 Planning",
        summary="We discussed Q1 goals.",
        key_topics=["Budget", "Hiring"],
        action_items=["Review budget by Friday"],
        decisions=["Hire 2 engineers"],
        language="en",
    )
    md = notes.to_markdown(date_str="2024-01-15")
    assert "# Q1 Planning" in md
    assert "2024-01-15" in md
    assert "We discussed Q1 goals." in md
    assert "- Budget" in md
    assert "- [ ] Review budget by Friday" in md
    assert "- Hire 2 engineers" in md


def test_to_markdown_with_transcript():
    notes = MeetingNotes(title="Test", summary="Summary.", language="en")
    md = notes.to_markdown(transcript="Speaker: Hello.")
    assert "Full Transcript" in md
    assert "Speaker: Hello." in md


def test_to_markdown_non_english_shows_language():
    notes = MeetingNotes(title="Pulong", summary="Nagpulong kami.", language="tl")
    md = notes.to_markdown()
    assert "Language detected" in md
    assert "tl" in md


def test_to_markdown_english_hides_language():
    notes = MeetingNotes(title="Test", summary="Summary.", language="en")
    md = notes.to_markdown()
    assert "Language detected" not in md


# ---------------------------------------------------------------------------
# Summarizer – extractive backend
# ---------------------------------------------------------------------------


SAMPLE_TRANSCRIPT = (
    "Good morning everyone. Today we are here to discuss the Q1 budget. "
    "John will prepare the financial report by next Monday. "
    "We need to hire two more engineers for the backend team. "
    "Sarah should follow up with the vendors regarding the contract. "
    "We decided to postpone the product launch to March. "
    "The team needs to complete the testing phase before release."
)


def test_extractive_summarizer_returns_meeting_notes():
    s = Summarizer(use_ollama=False)
    notes = s.summarize(SAMPLE_TRANSCRIPT, language="en")
    assert isinstance(notes, MeetingNotes)
    assert notes.title
    assert notes.summary
    assert notes.language == "en"


def test_extractive_detects_action_items():
    s = Summarizer(use_ollama=False)
    notes = s.summarize(SAMPLE_TRANSCRIPT, language="en")
    # Should detect sentences with "will", "should", "need to"
    combined = " ".join(notes.action_items).lower()
    assert any(kw in combined for kw in ["will", "should", "need"])


def test_extractive_empty_transcript_returns_default():
    s = Summarizer(use_ollama=False)
    notes = s.summarize("", language="en")
    assert notes.summary == "(No transcript content to summarize.)"


def test_extractive_uses_filipino_keywords():
    """Extractive summarizer should detect Filipino action items."""
    fil_transcript = (
        "Magandang umaga. Kailangang tapusin ang ulat bago Lunes. "
        "Dapat mag-follow up si Juan sa mga vendor. "
        "Nagpasya kami na ipagpaliban ang paglulunsad ng produkto."
    )
    s = Summarizer(use_ollama=False)
    notes = s.summarize(fil_transcript, language="tl")
    assert notes.language == "tl"


# ---------------------------------------------------------------------------
# Summarizer – Ollama backend
# ---------------------------------------------------------------------------


FAKE_OLLAMA_JSON = """{
  "title": "Q1 Planning Meeting",
  "summary": "Discussed Q1 goals and budget.",
  "key_topics": ["Budget", "Hiring"],
  "decisions": ["Hire 2 engineers"],
  "action_items": ["Prepare financial report by Monday"]
}"""


@patch("requests.post")
def test_ollama_backend_success(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": FAKE_OLLAMA_JSON}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    s = Summarizer(use_ollama=True)
    notes = s.summarize(SAMPLE_TRANSCRIPT, language="en")
    assert notes.title == "Q1 Planning Meeting"
    assert notes.summary == "Discussed Q1 goals and budget."
    assert "Budget" in notes.key_topics
    assert "Hire 2 engineers" in notes.decisions
    assert len(notes.action_items) == 1


@patch("requests.post")
def test_ollama_failure_falls_back_to_extractive(mock_post):
    """If Ollama raises an exception, extractive fallback is used."""
    mock_post.side_effect = Exception("Connection refused")

    s = Summarizer(use_ollama=True)
    notes = s.summarize(SAMPLE_TRANSCRIPT, language="en")
    assert isinstance(notes, MeetingNotes)
    assert notes.summary  # fallback produced something


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_split_sentences_basic():
    text = "Hello world. How are you? Fine."
    sentences = _split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello world."


def test_split_sentences_empty():
    assert _split_sentences("") == []


def test_score_sentences_returns_list():
    sentences = ["The quick brown fox.", "The fox jumps."]
    scores = _score_sentences(sentences)
    assert len(scores) == len(sentences)
    assert all(isinstance(s, float) for s in scores)


def test_first_n_words_short():
    assert _first_n_words("Hello world", 10) == "Hello world"


def test_first_n_words_truncates():
    result = _first_n_words("One two three four five six seven eight", 5)
    assert result.endswith("…")
    assert "One" in result


def test_derive_title_nonempty():
    title = _derive_title("Good morning everyone today we discuss")
    assert title
    assert len(title.split()) <= 6


def test_build_prompt_contains_transcript():
    prompt = _build_prompt("Hello world.", "en")
    assert "Hello world." in prompt
    assert "JSON" in prompt


def test_parse_llm_response_plain_json():
    notes = _parse_llm_response(FAKE_OLLAMA_JSON, language="en")
    assert notes.title == "Q1 Planning Meeting"
    assert notes.language == "en"


def test_parse_llm_response_wrapped_in_fences():
    wrapped = f"```json\n{FAKE_OLLAMA_JSON}\n```"
    notes = _parse_llm_response(wrapped, language="en")
    assert notes.title == "Q1 Planning Meeting"


def test_parse_llm_response_invalid_raises():
    with pytest.raises(Exception):
        _parse_llm_response("not valid json at all !!!!", language="en")


# ---------------------------------------------------------------------------
# Long transcripts: chunking, map-reduce, context window
# ---------------------------------------------------------------------------

import json  # noqa: E402

import requests  # noqa: E402

from meeting_notes import summarizer as summarizer_module  # noqa: E402
from meeting_notes.summarizer import (  # noqa: E402
    _batch_partials,
    _build_reduce_prompt,
    _chunk_transcript,
    _merge_plain,
)


@pytest.fixture(autouse=True)
def _no_ollama_catalogue_lookup(monkeypatch):
    """``resolve_model`` asks a real localhost:11434 which models exist. With no
    Ollama running that is a ~4 s refused connection on Windows, and the long
    tests make dozens of calls, so pin the name instead."""
    monkeypatch.setattr(Summarizer, "resolve_model", lambda self: "llama3")


def _ollama_reply(**fields):
    """A fake ``requests.post`` result carrying *fields* as the model's JSON."""
    base = {
        "title": "T",
        "summary": "S",
        "key_topics": [],
        "decisions": [],
        "action_items": [],
    }
    base.update(fields)
    response = MagicMock()
    response.json.return_value = {"response": json.dumps(base)}
    response.raise_for_status = MagicMock()
    return response


def _long_transcript(sentences: int = 600) -> str:
    return " ".join(f"Sentence number {i} is about topic {i}." for i in range(sentences))


def test_chunk_transcript_covers_all_text_within_limit():
    text = _long_transcript()
    chunks = _chunk_transcript(text, 1_000)
    assert len(chunks) > 1
    assert all(0 < len(c) <= 1_000 for c in chunks)
    # Nothing dropped: every sentence survives, in order.
    joined = " ".join(chunks)
    assert "Sentence number 0 " in joined
    assert "Sentence number 599 " in joined
    assert joined.index("number 10 ") < joined.index("number 500 ")


def test_chunk_transcript_prefers_line_boundaries():
    text = "\n".join(f"[00:{i:02d}] Person 1: line {i}" for i in range(60))
    chunks = _chunk_transcript(text, 200)
    # A cut never lands inside a line.
    assert all(line.startswith("[00:") for c in chunks for line in c.split("\n"))


def test_chunk_transcript_splits_unpunctuated_run_on_words():
    text = "word " * 5_000
    chunks = _chunk_transcript(text, 1_000)
    assert all(len(c) <= 1_000 for c in chunks)


def test_chunk_transcript_cuts_a_word_longer_than_the_limit():
    chunks = _chunk_transcript("x" * 2_500, 1_000)
    assert [len(c) for c in chunks] == [1_000, 1_000, 500]


def test_chunk_transcript_empty():
    assert _chunk_transcript("  \n ", 1_000) == []


def test_build_prompt_does_not_truncate():
    """The old prompt cut at 4,000 chars, so notes described only the opening."""
    text = "a" * 9_000 + " THE-END-MARKER"
    assert "THE-END-MARKER" in _build_prompt(text, "en")


def test_build_prompt_part_hint():
    prompt = _build_prompt("Hello.", "en", part=(2, 5))
    assert "part 2 of 5" in prompt


@patch("requests.post")
def test_ollama_request_sets_context_window(mock_post):
    mock_post.return_value = _ollama_reply()
    Summarizer(use_ollama=True).summarize("Short transcript.", language="en")
    options = mock_post.call_args.kwargs["json"]["options"]
    assert options["num_ctx"] == summarizer_module.OLLAMA_NUM_CTX


@patch("requests.post")
def test_short_transcript_is_one_call(mock_post):
    mock_post.return_value = _ollama_reply()
    Summarizer(use_ollama=True).summarize(SAMPLE_TRANSCRIPT, language="en")
    assert mock_post.call_count == 1


@patch("requests.post")
def test_long_transcript_covers_the_whole_meeting(mock_post):
    """Regression: text past 4,000 chars must reach the model."""
    prompts: list[str] = []

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        prompts.append(json["prompt"])
        if "NOTES BY PART" in json["prompt"]:
            return _ollama_reply(title="Whole meeting", summary="Merged.")
        return _ollama_reply(summary="part")

    mock_post.side_effect = fake_post

    text = _long_transcript(1_000) + " FINAL-DECISION-MARKER."
    assert len(text) > summarizer_module.SINGLE_PASS_CHARS

    notes = Summarizer(use_ollama=True).summarize(text, language="en")

    map_prompts = [p for p in prompts if "NOTES BY PART" not in p]
    assert len(map_prompts) > 1
    assert any("FINAL-DECISION-MARKER" in p for p in map_prompts)
    assert any("NOTES BY PART" in p for p in prompts)  # a merge happened
    assert notes.title == "Whole meeting"


@patch("requests.post")
def test_long_transcript_later_part_failure_degrades_that_part(mock_post):
    calls = {"n": 0}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        if "NOTES BY PART" in json["prompt"]:
            return _ollama_reply(title="Merged", summary="ok")
        calls["n"] += 1
        if calls["n"] == 2:
            raise requests.ReadTimeout("slow")
        return _ollama_reply(summary="part")

    mock_post.side_effect = fake_post
    notes = Summarizer(use_ollama=True).summarize(_long_transcript(1_000), language="en")
    assert notes.title == "Merged"


@patch("requests.post")
def test_long_transcript_unreachable_ollama_falls_back_quickly(mock_post):
    mock_post.side_effect = requests.ConnectionError("refused")
    notes = Summarizer(use_ollama=True).summarize(_long_transcript(1_000), language="en")
    assert isinstance(notes, MeetingNotes)
    assert notes.summary
    assert mock_post.call_count == 1  # gave up on the first part, did not grind on


@patch("requests.post")
def test_long_transcript_unparseable_part_does_not_fail_meeting(mock_post):
    calls = {"n": 0}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        if "NOTES BY PART" in json["prompt"]:
            return _ollama_reply(title="Merged")
        calls["n"] += 1
        if calls["n"] == 1:
            bad = MagicMock()
            bad.json.return_value = {"response": "not json at all"}
            bad.raise_for_status = MagicMock()
            return bad
        return _ollama_reply(summary="part")

    mock_post.side_effect = fake_post
    notes = Summarizer(use_ollama=True).summarize(_long_transcript(1_000), language="en")
    assert notes.title == "Merged"


@patch("requests.post")
def test_merge_failure_falls_back_to_plain_union(mock_post):
    def fake_post(url, json=None, timeout=None):  # noqa: A002
        if "NOTES BY PART" in json["prompt"]:
            raise requests.ReadTimeout("slow")
        return _ollama_reply(
            title="Part title",
            summary="part",
            decisions=["Ship it"],
            action_items=["Ana sends the build"],
        )

    mock_post.side_effect = fake_post
    notes = Summarizer(use_ollama=True).summarize(_long_transcript(1_000), language="en")
    assert notes.title == "Part title"
    assert notes.decisions == ["Ship it"]  # repeated across parts, listed once
    assert notes.action_items == ["Ana sends the build"]


@patch("requests.post")
def test_many_parts_merge_in_rounds_and_terminate(mock_post, monkeypatch):
    monkeypatch.setattr(summarizer_module, "CHUNK_CHARS", 300)
    monkeypatch.setattr(summarizer_module, "SINGLE_PASS_CHARS", 400)
    monkeypatch.setattr(summarizer_module, "REDUCE_CHARS", 200)

    merges = {"n": 0}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        if "NOTES BY PART" in json["prompt"]:
            merges["n"] += 1
            return _ollama_reply(title="M", summary="merged " + "x" * 150)
        return _ollama_reply(summary="part " + "y" * 150)

    mock_post.side_effect = fake_post
    notes = Summarizer(use_ollama=True).summarize(_long_transcript(200), language="en")
    assert notes.title == "M"
    assert merges["n"] > 1  # more than one round of merging


def test_batch_partials_always_groups_at_least_two():
    parts = [MeetingNotes(title="t", summary="s" * 500) for _ in range(5)]
    batches = _batch_partials(parts, max_chars=10)  # smaller than any one part
    assert all(len(b) >= 2 for b in batches[:-1])
    assert sum(len(b) for b in batches) == 5
    assert len(batches) < 5  # progress guaranteed


def test_merge_plain_dedupes_case_insensitively_and_keeps_order():
    a = MeetingNotes(title="A", summary="one", decisions=["Hire two"], key_topics=["Budget"])
    b = MeetingNotes(
        title="B", summary="two", decisions=["hire two ", "Cut scope"], key_topics=["Budget"]
    )
    merged = _merge_plain([a, b], "en")
    assert merged.title == "A"
    assert merged.decisions == ["Hire two", "Cut scope"]
    assert merged.key_topics == ["Budget"]
    assert merged.summary == "one two"


def test_reduce_prompt_lists_every_part_in_order():
    parts = [MeetingNotes(title=f"T{i}", summary=f"S{i}") for i in range(3)]
    prompt = _build_reduce_prompt(parts, "en")
    assert prompt.index("PART 1 of 3") < prompt.index("PART 2 of 3") < prompt.index("PART 3 of 3")
    assert "Do not favour the first or the last part" in prompt


# ---------------------------------------------------------------------------
# Provenance: what wrote the notes
# ---------------------------------------------------------------------------


def test_extractive_notes_say_so():
    notes = Summarizer(use_ollama=False).summarize(SAMPLE_TRANSCRIPT, language="en")
    assert notes.generator == "extractive"


@patch("requests.post")
def test_ollama_notes_name_the_model(mock_post):
    mock_post.return_value = _ollama_reply()
    notes = Summarizer(use_ollama=True).summarize(SAMPLE_TRANSCRIPT, language="en")
    assert notes.generator == "ollama:llama3"


@patch("requests.post")
def test_the_silent_fallback_is_visible_in_the_notes(mock_post):
    """Ollama being down used to be invisible; now the notes carry that they are the basic kind."""
    mock_post.side_effect = Exception("Connection refused")
    notes = Summarizer(use_ollama=True).summarize(SAMPLE_TRANSCRIPT, language="en")
    assert notes.generator == "extractive"


# ---------------------------------------------------------------------------
# Speaker-labelled text for the model, and the meeting guidance
# ---------------------------------------------------------------------------

from meeting_notes.backends.base import TranscriptSegment  # noqa: E402
from meeting_notes.summarizer import speaker_text  # noqa: E402


def _seg(speaker, text):
    return TranscriptSegment(start=0.0, end=1.0, text=text, speaker=speaker)


def test_speaker_text_labels_each_turn():
    text = speaker_text([_seg("Person 1", "Hello."), _seg("Person 2", "Hi there."), _seg("Person 1", "Ready?")])
    assert text.splitlines() == ["Person 1: Hello.", "Person 2: Hi there.", "Person 1: Ready?"]


def test_speaker_text_merges_consecutive_turns_by_one_speaker():
    text = speaker_text([_seg("Person 1", "One."), _seg("Person 1", "Two."), _seg("Person 2", "Three.")])
    assert text.splitlines() == ["Person 1: One. Two.", "Person 2: Three."]


def test_speaker_text_is_none_when_voices_cannot_be_told_apart():
    assert speaker_text([_seg(None, "a"), _seg(None, "b")]) is None          # diarization off or failed
    assert speaker_text([_seg("Person 1", "a"), _seg("Person 1", "b")]) is None   # a monologue
    assert speaker_text([]) is None


def test_speaker_text_skips_empty_segments():
    text = speaker_text([_seg("Person 1", "  "), _seg("Person 1", "Hi."), _seg("Person 2", "Yo.")])
    assert text.splitlines() == ["Person 1: Hi.", "Person 2: Yo."]


def test_the_model_reads_the_labelled_text_and_the_fallback_the_plain():
    seen = []

    class Spy:
        name = "llamacpp:test"

        def generate(self, prompt):
            seen.append(prompt)
            raise RuntimeError("force the fallback")

    plain = "We ship Friday. Ana sends the build."
    labelled = "Person 1: We ship Friday.\nPerson 2: Ana sends the build."
    notes = Summarizer(use_ollama=False, provider=Spy()).summarize(plain, speaker_text=labelled)

    assert "Person 1: We ship Friday." in seen[0]           # what the model was given
    assert notes.generator == "extractive"
    assert "Person 1:" not in notes.summary                  # the fallback worked from the plain text


def test_the_meeting_prompt_asks_for_english_and_forbids_inventing():
    prompt = _build_prompt("Person 1: Kumusta.", "tl", "meeting")
    assert "ENGLISH" in prompt
    assert "Do not invent" in prompt
    assert "Person 1, Person 2" in prompt


def test_a_long_labelled_transcript_is_cut_between_turns():
    turns = [f"Person {1 + i % 2}: " + "word " * 120 for i in range(200)]
    chunks = _chunk_transcript("\n".join(turns), 4_000)
    assert len(chunks) > 1
    assert all(line.startswith("Person ") for c in chunks for line in c.split("\n"))
