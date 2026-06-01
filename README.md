# Meeting Notes AI

An offline, multi-language AI meeting note taker.

## Features

| Feature | Detail |
|---|---|
| 🎙️ **Live recording** | Capture audio directly from your microphone |
| 📁 **File transcription** | Transcribe existing WAV, MP3, M4A, and other audio files |
| 🌍 **Multi-language** | 99+ languages including **Filipino / Tagalog** (`tl`) via OpenAI Whisper |
| ✈️ **Fully offline** | Whisper runs locally; Ollama provides a local LLM – no internet required after setup |
| 🤖 **AI-generated notes** | Structured notes with summary, key topics, decisions, and action items |
| 📄 **Document output** | Export as **Markdown** (`.md`) or **Word** (`.docx`) |

---

## Quick Start

### 1. Install

```bash
pip install -e .
```

> **System dependencies**
> - `portaudio` is required for microphone recording: `brew install portaudio` (macOS) or `sudo apt install portaudio19-dev` (Linux).
> - `ffmpeg` is required to load non-WAV audio files: `brew install ffmpeg` or `sudo apt install ffmpeg`.

### 2. (Recommended) Install Ollama for richer AI notes

```bash
# Install Ollama from https://ollama.com, then pull a model:
ollama pull llama3
```

If Ollama is not running, the app automatically falls back to a lightweight built-in extractive summarizer.

### 3. Record a meeting

```bash
meeting-notes record
# Press Ctrl+C when the meeting is done.
```

### 4. Or transcribe an existing audio file

```bash
meeting-notes transcribe meeting.wav
```

---

## Usage

### `record` – live microphone recording

```
meeting-notes record [OPTIONS]

Options:
  --model [tiny|base|small|medium|large]
                                  Whisper model size  [default: base]
  --language TEXT                 Language code, e.g. "tl" (Filipino).
                                  Auto-detected if omitted.
  --output-dir PATH               Directory to save the meeting document
                                  [default: ~/meeting-notes]
  --format [markdown|docx]        Output document format  [default: markdown]
  --no-ollama                     Use built-in extractive summarizer only
  --ollama-model TEXT             Ollama model  [default: llama3]
  --ollama-url TEXT               Ollama server URL  [default: http://localhost:11434]
```

### `transcribe` – process an audio file

```
meeting-notes transcribe AUDIO_FILE [OPTIONS]
```

All options are the same as `record`.

### `languages` – list supported languages

```
meeting-notes languages
```

---

## Language Support (sample)

| Code | Language |
|------|----------|
| `en` | English |
| `tl` | Filipino / Tagalog |
| `es` | Spanish |
| `fr` | French |
| `de` | German |
| `zh` | Chinese |
| `ja` | Japanese |
| `ko` | Korean |
| `ar` | Arabic |
| `hi` | Hindi |
| `id` | Indonesian |

Whisper supports **99+ languages** in total. Pass any ISO 639-1 code with `--language`.

---

## Example output (Markdown)

```markdown
# Q1 Planning Meeting

**Date:** 2024-01-15 10:30

## Summary

The team discussed the Q1 roadmap, budget allocation, and hiring plans.
A decision was made to launch the product in March.

## Key Topics

- Budget review
- Engineering hiring
- Product launch timeline

## Decisions Made

- Hire 2 backend engineers
- Launch product in March

## Action Items

- [ ] John to prepare financial report by Monday
- [ ] Sarah to follow up with vendors on contract

## Full Transcript

[00:00 – 00:30] Good morning everyone.
[00:30 – 01:00] Today we discuss the Q1 roadmap…
```

---

## Offline Operation

Everything runs locally:

| Component | Tool | Notes |
|-----------|------|-------|
| Speech-to-text | [OpenAI Whisper](https://github.com/openai/whisper) | Models downloaded once to `~/.cache/whisper` |
| LLM summarization | [Ollama](https://ollama.com) | Optional; falls back to extractive if unavailable |
| Document export | `python-docx` / built-in | No external service needed |

### Choosing a Whisper model

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | 39 MB | Fastest | Good |
| `base` | 74 MB | Fast | Better *(default)* |
| `small` | 244 MB | Moderate | Good+ |
| `medium` | 769 MB | Slow | High |
| `large` | 1.5 GB | Slowest | Best |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Project structure

```
src/meeting_notes/
├── __init__.py
├── main.py         # CLI entry point (click)
├── audio.py        # Microphone recording & audio file loading
├── transcriber.py  # Offline transcription via OpenAI Whisper
├── summarizer.py   # Meeting notes generation (Ollama LLM + extractive fallback)
├── document.py     # Markdown & DOCX document export
└── config.py       # Configuration & constants
tests/
├── test_audio.py
├── test_config.py
├── test_document.py
├── test_summarizer.py
└── test_transcriber.py
```
