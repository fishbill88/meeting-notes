---
name: improve-summarizer
description: "Guide for improving meeting notes quality — tuning the Ollama LLM prompt, extractive fallback, and language-specific keywords. Use when asked to improve notes quality, better action items, better summaries, or add language support."
---

# Improve Summarizer — Meeting Notes AI

## Architecture

`src/meeting_notes/summarizer.py` has two backends:

1. **Ollama (LLM)** — `_summarize_ollama` → `_build_prompt` → Ollama `/api/generate` → `_parse_llm_response`
2. **Extractive fallback** — `_summarize_extractive` — pure Python, no LLM

## Improving the Ollama prompt

Edit `_build_prompt()` in `summarizer.py`. Current approach:
- Sends full transcript (up to 4000 chars) to the model
- Asks for JSON with: `title`, `summary`, `key_topics`, `decisions`, `action_items`

Tips for better results:
- Add **few-shot examples** inside the prompt
- For multilingual transcripts, include language-specific instruction: `"Respond in the same language as the transcript"`
- Increase the `[:4000]` slice if summaries are truncated (check `requests` timeout too)
- Try a system prompt pattern if the Ollama model supports it (use `messages` format instead of `prompt`)

## Improving the extractive fallback

The fallback in `_summarize_extractive` uses TF-based sentence scoring.

### Adding language support

1. Add action keywords for the new language to `action_keywords` dict:

```python
action_keywords = {
    "en": ["will", "should", ...],
    "tl": ["kailangang", "gagawin", ...],
    "es": ["deberá", "necesita", "acción", ...],  # add here
}
```

2. Add stopwords to `_score_sentences`:

```python
stopwords = {
    # English
    "the", "a", ...
    # Filipino
    "ang", "ng", ...
    # Spanish — add here
    "el", "la", "los", "las", "un", "una", ...
}
```

### Improving title derivation

`_derive_title` takes the first 6 words of the transcript. To improve:
- Look for meeting title patterns (e.g. lines starting with "Meeting:", "Subject:", "Re:")
- Use the first topic from `key_topics` if transcript start is generic

## Testing summarizer changes

```bash
pytest tests/test_summarizer.py -v
```

Test the Ollama path manually with Ollama running:
```bash
meeting-notes transcribe test.wav --ollama-model llama3
```

Test extractive fallback:
```bash
meeting-notes transcribe test.wav --no-ollama
```
