---
name: test
description: "Run the pytest test suite for this project. Use when asked to run tests, check test coverage, or verify nothing is broken."
---

# Test — Meeting Notes AI

## Run all tests

```bash
pytest -v
```

## Run a single module's tests

```bash
pytest tests/test_summarizer.py -v
pytest tests/test_transcriber.py -v
pytest tests/test_document.py -v
pytest tests/test_audio.py -v
pytest tests/test_config.py -v
```

## Run with coverage

```bash
pytest --cov=meeting_notes tests/ --cov-report=term-missing
```

## Run a specific test by name

```bash
pytest -k "test_extractive" -v
pytest -k "test_ollama" -v
```

## Test structure

| Test file | What it covers |
|-----------|---------------|
| `tests/test_audio.py` | `AudioRecorder`, `load_audio_file` |
| `tests/test_transcriber.py` | `Transcriber.transcribe_file`, `transcribe_array`, model loading |
| `tests/test_summarizer.py` | `Summarizer.summarize`, Ollama path, extractive fallback, `MeetingNotes.to_markdown` |
| `tests/test_document.py` | `generate_document` for both markdown and docx |
| `tests/test_config.py` | `AppConfig.validate`, constants |

## Notes for this project

- Whisper (`openai-whisper`) is mocked — tests do not download models
- `sounddevice` is mocked — tests do not need a microphone
- `requests` is mocked — tests do not need Ollama running
- Tests run offline and fast
