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
