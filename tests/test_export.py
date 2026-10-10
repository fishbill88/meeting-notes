"""Tests for in-memory export and the note templates."""

from __future__ import annotations

import pytest

from meeting_notes.document import (
    EXPORT_FORMATS,
    render_document,
    suggested_filename,
)
from meeting_notes.summarizer import (
    TEMPLATE_NAMES,
    TEMPLATES,
    MeetingNotes,
    Summarizer,
    _build_prompt,
)

NOTES = MeetingNotes(
    title="Chest pain, 3 days",
    summary="Patient reports intermittent chest pain on exertion.",
    key_topics=["chest pain", "shortness of breath"],
    decisions=["Likely musculoskeletal"],
    action_items=["Amoxicillin 500 mg twice a day", "ECG next visit"],
    medical_terms=["amoxicillin", "chest pain"],
    language="tl",
)

TRANSCRIPT = "[00:00 – 00:05] Nag-reklamo siya ng chest pain."


# ---------------------------------------------------------------------------
# render_document
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", sorted(EXPORT_FORMATS))
def test_every_declared_format_renders_bytes(fmt):
    payload = render_document(NOTES, TRANSCRIPT, fmt)
    assert isinstance(payload, bytes)
    assert len(payload) > 0


def test_markdown_contains_the_notes():
    text = render_document(NOTES, TRANSCRIPT, "markdown").decode("utf-8")
    assert "# Chest pain, 3 days" in text
    assert "Amoxicillin 500 mg twice a day" in text
    assert "Nag-reklamo" in text


def test_docx_is_a_zip_container():
    # A .docx is an OPC zip; "PK" is the signature. Cheaper than parsing it,
    # and enough to catch an empty or truncated write.
    payload = render_document(NOTES, TRANSCRIPT, "docx")
    assert payload[:2] == b"PK"
    assert len(payload) > 1000


def test_docx_round_trips_through_python_docx(tmp_path):
    docx = pytest.importorskip("docx")

    path = tmp_path / "notes.docx"
    path.write_bytes(render_document(NOTES, TRANSCRIPT, "docx"))

    text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    assert "Patient reports intermittent chest pain" in text
    assert "Amoxicillin 500 mg twice a day" in text


def test_text_format_strips_markdown_syntax():
    # This is the copy-to-EMR path; an EMR field renders no Markdown.
    text = render_document(NOTES, None, "text").decode("utf-8")
    assert "#" not in text
    assert "**" not in text
    assert "Chest pain, 3 days" in text
    assert "• Amoxicillin 500 mg twice a day" in text


def test_transcript_is_optional():
    text = render_document(NOTES, None, "markdown").decode("utf-8")
    assert "Nag-reklamo" not in text
    assert "Chest pain" in text


def test_invalid_format_raises():
    with pytest.raises(ValueError, match="Invalid output_format"):
        render_document(NOTES, None, "pdf")


def test_date_string_is_used_when_given():
    text = render_document(NOTES, None, "markdown", date_str="2026-07-30 09:15")
    assert "2026-07-30 09:15" in text.decode("utf-8")


# ---------------------------------------------------------------------------
# suggested_filename
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt,extension",
    [("markdown", ".md"), ("docx", ".docx"), ("text", ".txt")],
)
def test_suggested_filename_extension(fmt, extension):
    assert suggested_filename(NOTES, fmt).endswith(extension)


def test_suggested_filename_is_filesystem_safe():
    notes = MeetingNotes(title='Chest pain: "urgent" / review?', summary="")
    name = suggested_filename(notes, "markdown")
    assert not any(ch in name for ch in '\\/:*?"<>|')


def test_suggested_filename_handles_an_empty_title():
    assert suggested_filename(MeetingNotes(title="", summary=""), "markdown")


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def test_both_templates_are_registered():
    assert set(TEMPLATE_NAMES) == {"meeting", "consultation"}


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
def test_every_template_declares_the_required_keys(name):
    spec = TEMPLATES[name]
    for key in ("label", "empty_title", "role", "fields", "source_label"):
        assert spec[key], f"{name}.{key}"


def test_consultation_prompt_requests_clinical_structure():
    prompt = _build_prompt("...", "tl", "consultation")
    assert "chief complaint" in prompt.lower()
    assert "history of present illness" in prompt.lower()


def test_consultation_prompt_requires_english_output():
    # PH medical records are English even when the consult was not.
    prompt = _build_prompt("...", "tl", "consultation")
    assert "ENGLISH" in prompt


def test_consultation_prompt_preserves_patient_quotes():
    prompt = _build_prompt("...", "tl", "consultation")
    assert "direct quote" in prompt.lower()


def test_consultation_prompt_forbids_inventing_findings():
    prompt = _build_prompt("...", "en", "consultation")
    assert "never infer" in prompt.lower()


def test_meeting_prompt_stays_generic():
    prompt = _build_prompt("...", "en", "meeting")
    assert "meeting note taker" in prompt.lower()
    assert "chief complaint" not in prompt.lower()


def test_unknown_template_falls_back_in_prompt_builder():
    # Defensive: the validation lives in summarize(), but the builder should
    # not explode if it is ever called directly with junk.
    assert _build_prompt("...", "en", "nonsense")


def test_summarize_rejects_an_unknown_template():
    with pytest.raises(ValueError, match="Invalid template"):
        Summarizer(use_ollama=False).summarize("text", template="nonsense")


def test_empty_transcript_uses_the_template_title():
    notes = Summarizer(use_ollama=False).summarize("", template="consultation")
    assert notes.title == "Consultation Notes"


def test_consultation_template_works_with_the_extractive_fallback():
    # Ollama is usually absent on a clinic laptop; the template must not be a
    # hard dependency on it.
    notes = Summarizer(use_ollama=False).summarize(
        "Patient reports chest pain. Prescribed amoxicillin.",
        template="consultation",
    )
    assert notes.summary
    assert notes.title


def test_notes_carry_medical_terms():
    assert MeetingNotes(title="t", summary="s").medical_terms == []
    assert NOTES.medical_terms == ["amoxicillin", "chest pain"]
