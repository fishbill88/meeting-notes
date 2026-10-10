---
name: add-export-format
description: "Guide for adding a new document export format (HTML, PDF, JSON, etc.) to the meeting-notes app. Use when asked to add a new output format."
---

# Add Export Format — Meeting Notes AI

## Overview

The app exports meeting notes via `src/meeting_notes/document.py`. Adding a new format (e.g. HTML, PDF, JSON) involves three files.

## Step-by-step

### 1. Update `config.py` (if needed)

Only needed for new default constants. Most format additions don't touch this.

### 2. Update `document.py`

Add a private writer function:

```python
def _write_html(
    notes: MeetingNotes,
    transcript: Optional[str],
    path: str,
    date_str: str,
) -> None:
    # Build HTML string from notes fields
    ...
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
```

Then update `generate_document` to dispatch to it:

```python
elif output_format == "html":
    out_path = os.path.join(output_dir, f"{filename}.html")
    _write_html(notes, transcript, out_path, date_str)
```

### 3. Update `main.py` (both commands)

Add the new format to the `click.Choice` in both `record` and `transcribe` commands:

```python
type=click.Choice(["markdown", "docx", "html"]),
```

### 4. Add tests in `tests/test_document.py`

Follow the existing pattern — mock `MeetingNotes` and assert the output file exists and contains expected content.

## For PDF output

Use `reportlab` or `weasyprint`. Add to `pyproject.toml` dependencies. Use lazy import inside `_write_pdf`.

## For JSON output

No extra dependencies — just `json.dumps` the `MeetingNotes` dataclass via `dataclasses.asdict`.

## For HTML output

Use Python's built-in string formatting or `jinja2` (lazy import). Reference the `to_markdown` method in `summarizer.py` as a template for field ordering.
