---
name: run
description: "Launch and test the meeting-notes CLI app. Use when asked to run, start, demo, or test the app end-to-end. Handles record, transcribe, and languages commands."
---

# Run — Meeting Notes AI

## How to launch this project

This is a Python CLI app. There is no server to start.

### Verify the install

```bash
pip install -e ".[dev]"
meeting-notes --version
```

### List supported languages (fastest smoke-test)

```bash
meeting-notes languages
```

No audio hardware or Ollama required.

### Transcribe an existing audio file

```bash
meeting-notes transcribe path/to/audio.wav --language en --no-ollama
```

### Record from microphone

```bash
meeting-notes record --model tiny --no-ollama
# Press Ctrl+C to stop
```

## Confirming a change works

1. `meeting-notes languages` — verifies CLI boots correctly
2. For summarizer changes: run `pytest tests/test_summarizer.py -v`
3. For document changes: run `pytest tests/test_document.py -v`
4. For transcriber changes: run `pytest tests/test_transcriber.py -v`
5. Check output in `~/meeting-notes/` (default output dir)

## Common flags

| Flag | Purpose |
|------|---------|
| `--model tiny` | Fastest Whisper model for testing |
| `--no-ollama` | Skip Ollama, use extractive fallback |
| `--format docx` | Output Word document instead of Markdown |
| `--language tl` | Force Filipino/Tagalog |
| `--output-dir ./output` | Save to local dir instead of home |
