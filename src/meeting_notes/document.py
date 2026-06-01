"""Document generation module – saves meeting notes as Markdown or DOCX files."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from meeting_notes.summarizer import MeetingNotes


def generate_document(
    notes: MeetingNotes,
    transcript: Optional[str] = None,
    output_dir: str = ".",
    output_format: str = "markdown",
    filename: Optional[str] = None,
) -> str:
    """Write meeting notes to a file.

    Args:
        notes: Structured :class:`~meeting_notes.summarizer.MeetingNotes`.
        transcript: Optional full transcript text to append to the document.
        output_dir: Directory where the file will be saved.
        output_format: ``"markdown"`` (produces a ``.md`` file) or ``"docx"``
            (produces a ``.docx`` Word document).
        filename: Optional base filename (without extension).  A timestamped
            name is generated when not provided.

    Returns:
        Absolute path to the generated file.
    """
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if filename is None:
        safe_title = _safe_filename(notes.title)
        filename = f"{timestamp}_{safe_title}"

    if output_format == "docx":
        out_path = os.path.join(output_dir, f"{filename}.docx")
        _write_docx(notes, transcript, out_path, date_str)
    else:
        out_path = os.path.join(output_dir, f"{filename}.md")
        _write_markdown(notes, transcript, out_path, date_str)

    return str(Path(out_path).resolve())


# ------------------------------------------------------------------
# Markdown writer
# ------------------------------------------------------------------


def _write_markdown(
    notes: MeetingNotes,
    transcript: Optional[str],
    path: str,
    date_str: str,
) -> None:
    content = notes.to_markdown(transcript=transcript, date_str=date_str)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ------------------------------------------------------------------
# DOCX writer
# ------------------------------------------------------------------


def _write_docx(
    notes: MeetingNotes,
    transcript: Optional[str],
    path: str,
    date_str: str,
) -> None:
    try:
        from docx import Document  # noqa: PLC0415
        from docx.shared import Pt, RGBColor  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is required to generate DOCX files. "
            "Install it with: pip install python-docx"
        ) from exc

    doc = Document()

    # Title
    title_para = doc.add_heading(notes.title, level=0)

    # Metadata
    meta = doc.add_paragraph()
    meta.add_run("Date: ").bold = True
    meta.add_run(date_str)

    if notes.language and notes.language != "en":
        lang_para = doc.add_paragraph()
        lang_para.add_run("Language detected: ").bold = True
        lang_para.add_run(notes.language)

    doc.add_paragraph()  # spacer

    # Summary
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(notes.summary)

    # Key Topics
    if notes.key_topics:
        doc.add_heading("Key Topics", level=1)
        for topic in notes.key_topics:
            doc.add_paragraph(topic, style="List Bullet")

    # Decisions
    if notes.decisions:
        doc.add_heading("Decisions Made", level=1)
        for decision in notes.decisions:
            doc.add_paragraph(decision, style="List Bullet")

    # Action Items
    if notes.action_items:
        doc.add_heading("Action Items", level=1)
        for item in notes.action_items:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run("☐ ").bold = True
            p.add_run(item)

    # Full Transcript
    if transcript:
        doc.add_heading("Full Transcript", level=1)
        # Split into paragraphs to avoid a single giant paragraph
        for chunk in transcript.split("\n"):
            if chunk.strip():
                doc.add_paragraph(chunk.strip())

    doc.save(path)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _safe_filename(text: str, max_len: int = 40) -> str:
    """Convert arbitrary text to a filesystem-safe filename fragment."""
    import re
    safe = re.sub(r"[^\w\s-]", "", text)
    safe = re.sub(r"[\s-]+", "_", safe).strip("_")
    return safe[:max_len] if safe else "meeting"
