"""Tests for the document generation module."""

from __future__ import annotations

import os
import re
import tempfile

import pytest

from meeting_notes.document import _safe_filename, generate_document
from meeting_notes.summarizer import MeetingNotes


SAMPLE_NOTES = MeetingNotes(
    title="Q1 Planning Meeting",
    summary="We discussed the Q1 roadmap and budget allocation.",
    key_topics=["Budget", "Hiring", "Product Roadmap"],
    action_items=["Review budget by Friday", "Post job listings by Monday"],
    decisions=["Hire 2 backend engineers", "Launch in March"],
    language="en",
)

SAMPLE_TRANSCRIPT = (
    "[00:00 – 00:30] Good morning everyone.\n"
    "[00:30 – 01:00] Today we discuss Q1 goals."
)


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------


def test_generate_markdown_file_created():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_document(
            SAMPLE_NOTES,
            transcript=SAMPLE_TRANSCRIPT,
            output_dir=tmpdir,
            output_format="markdown",
        )
        assert os.path.exists(path)
        assert path.endswith(".md")


def test_generate_markdown_content():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_document(
            SAMPLE_NOTES,
            transcript=SAMPLE_TRANSCRIPT,
            output_dir=tmpdir,
            output_format="markdown",
        )
        with open(path, encoding="utf-8") as f:
            content = f.read()

    assert "# Q1 Planning Meeting" in content
    assert "We discussed the Q1 roadmap" in content
    assert "- Budget" in content
    assert "- [ ] Review budget by Friday" in content
    assert "- Hire 2 backend engineers" in content
    assert "Full Transcript" in content
    assert "Good morning everyone." in content


def test_generate_markdown_custom_filename():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_document(
            SAMPLE_NOTES,
            output_dir=tmpdir,
            output_format="markdown",
            filename="my_meeting",
        )
        assert os.path.basename(path) == "my_meeting.md"


def test_generate_creates_output_dir_if_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, "sub", "meetings")
        path = generate_document(
            SAMPLE_NOTES,
            output_dir=new_dir,
            output_format="markdown",
        )
        assert os.path.exists(new_dir)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# DOCX generation
# ---------------------------------------------------------------------------


def test_generate_docx_file_created():
    pytest.importorskip("docx")  # skip if python-docx not installed
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_document(
            SAMPLE_NOTES,
            transcript=SAMPLE_TRANSCRIPT,
            output_dir=tmpdir,
            output_format="docx",
        )
        assert os.path.exists(path)
        assert path.endswith(".docx")


def test_generate_docx_contains_expected_text():
    docx = pytest.importorskip("docx")
    from docx import Document

    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_document(
            SAMPLE_NOTES,
            transcript=SAMPLE_TRANSCRIPT,
            output_dir=tmpdir,
            output_format="docx",
        )
        doc = Document(path)
        full_text = "\n".join(p.text for p in doc.paragraphs)

    assert "Q1 Planning Meeting" in full_text
    assert "We discussed the Q1 roadmap" in full_text
    assert "Budget" in full_text
    assert "Review budget by Friday" in full_text


# ---------------------------------------------------------------------------
# _safe_filename helper
# ---------------------------------------------------------------------------


def test_safe_filename_strips_special_chars():
    result = _safe_filename("Hello, World! Meeting.")
    assert re.match(r"^[\w_-]+$", result), f"Not safe: {result!r}"


def test_safe_filename_max_length():
    long_title = "A" * 100
    result = _safe_filename(long_title)
    assert len(result) <= 40


def test_safe_filename_empty_fallback():
    assert _safe_filename("!!!") == "meeting"


def test_safe_filename_spaces_become_underscores():
    result = _safe_filename("Q1 Planning Meeting")
    assert " " not in result
    assert "Q1" in result
