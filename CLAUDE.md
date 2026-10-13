# Meeting Notes AI — CLAUDE.md

## Current scope: corporate meetings

**The product is corporate meeting notes** — project syncs, standups, stakeholder
reviews. Default every prompt, fixture, sample transcript and UI decision to that
frame. The desktop app (`desktop/`) is the active surface; mobile is on hold until
desktop is settled.

The doctor–patient/clinical use case is **parked, not cancelled** — it returns in a
later phase. Do not delete `medicalTerms.ts`, the `consultation` template, or the
`medical_terms` field, and do not introduce clinical framing into anything new.
Highlighting and `medical_terms` are gated on the `consultation` template; a meeting
carries neither. See `enrich()` in `desktop/src/App.tsx` for where that gate lives.

## Project Overview

Python CLI app for offline, multi-language meeting transcription and AI-generated structured notes.

**Stack:** Python 3.9+, OpenAI Whisper (speech-to-text), Ollama (local LLM), Click (CLI), Rich (terminal UI), python-docx (Word export)

**Core capability:** Fully offline after initial setup. Whisper runs locally; Ollama provides a local LLM with extractive fallback when Ollama is unavailable.

## Architecture

```
CLI command (main.py)
  → AudioRecorder (audio.py)         — mic capture → WAV file
  → Transcriber (transcriber.py)     — Whisper offline STT
  → Summarizer (summarizer.py)       — Ollama LLM or extractive fallback
  → generate_document (document.py)  — .md or .docx output
```

## Module Reference

| File | Purpose |
|------|---------|
| `src/meeting_notes/main.py` | Click CLI: `record`, `transcribe`, `languages` commands + shared `_process_audio` pipeline |
| `src/meeting_notes/audio.py` | `AudioRecorder` (sounddevice background thread) + `load_audio_file` helper |
| `src/meeting_notes/transcriber.py` | `Transcriber` wraps Whisper; lazy model loading; returns `TranscriptionResult` with timed segments |
| `src/meeting_notes/diarize.py` | Speaker diarization (sherpa-onnx pyannote + CAM++, same models as mobile); labels segments "Person N"; re-checks each short phrase against every speaker's voice to undo turn-edge flips (desktop only); models auto-download to `models_dir()/diarization` (~34 MB); degrades to no-speakers on any failure |
| `src/meeting_notes/backends/omnilingual.py` | `OmnilingualBackend` — Meta Omnilingual ASR 300M CTC via sherpa-onnx; the `ceb` (Cebuano/Bisaya) route; ~292 MB auto-download to `models_dir()/omnilingual`; language-agnostic, stamps the requested code |
| `src/meeting_notes/backends/router.py` | Language mode → engine: `en`/`auto` → faster-whisper, `tl` → NeMo Tagalog, `ceb` → Omnilingual; each dedicated engine falls back to Whisper (Cebuano as `tl`) with `backend` recorded for the UI |
| `src/meeting_notes/llm.py` | The bundled language model: a PINNED catalog (llama.cpp build tag + sha256; each GGUF with sha256 and a Hugging Face revision in its URL), a resumable verified `Installer` (download, hash, zip-slip-safe unpack, start the model once and ask it a question BEFORE writing `installed.json`), and `Runtime` which runs `llama-server` as a child of the engine under a Windows kill-on-close job object, idle-stops after 5 min, and asks for JSON via the server's grammar. `BundledProvider` is what `Summarizer(provider=...)` calls; `LlmUnavailable` means "could not be reached" (give up at the first part), not "answered badly". Routes `GET /llm`, `POST /llm/install?model=`, `DELETE /llm/install`, `DELETE /llm` |
| `src/meeting_notes/jobs.py` | Background notes jobs for hosts that cannot hold a request open (Weave): `JobQueue` runs ONE job at a time on a worker thread, in memory (a job does not survive a restart; the saved meeting does). Routes `POST /jobs`, `GET /jobs`, `GET /jobs/{id}`, `DELETE /jobs/{id}` in `server.py`. The uploaded audio copy lives under `data/jobs/` and is deleted when the job ends, however it ends; anything left there at start-up is swept |
| `src/meeting_notes/summarizer.py` | `Summarizer` calls Ollama (JSON prompt) or extractive fallback; returns `MeetingNotes` dataclass. Transcripts over `SINGLE_PASS_CHARS` are summarised in parts (`_summarize_long`) and merged (`_reduce`); `num_ctx` is set explicitly |
| `src/meeting_notes/document.py` | `generate_document` dispatches to `_write_markdown` or `_write_docx` |
| `src/meeting_notes/config.py` | `WHISPER_MODELS`, `DEFAULT_*`, `SUPPORTED_LANGUAGES`, `AppConfig` dataclass |

## Data Flow

```
Audio file/mic
  ↓
TranscriptionResult(text, language, language_name, segments[])
  ↓
MeetingNotes(title, summary, key_topics, decisions, action_items, language)
  ↓
<timestamp>_<title>.md  or  .docx
```

## Development Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=meeting_notes tests/

# Run the CLI
meeting-notes record
meeting-notes transcribe path/to/audio.wav --language tl
meeting-notes languages
```

## Common Development Tasks

### Adding a new export format (e.g., HTML, PDF)

1. Add format option to `output_format` choices in `main.py` (both `record` and `transcribe` commands)
2. Add `_write_<format>(notes, transcript, path, date_str)` in `document.py`
3. Update `generate_document` to dispatch to the new writer
4. Add tests in `tests/test_document.py`

### Adding a new CLI command

1. Define `@cli.command()` in `main.py`
2. Use lazy imports inside the function body (project pattern for heavy deps)

### Improving the LLM summarization prompt

- Edit `_build_prompt()` in `summarizer.py`
- Prompt must instruct Ollama to return **only** valid JSON with these fields: `title`, `summary`, `key_topics`, `decisions`, `action_items`
- `_parse_llm_response` handles markdown code-fence stripping and brace extraction

### Adding language-specific keywords (extractive fallback)

- Add a new language entry to `action_keywords` dict in `Summarizer._summarize_extractive`
- Add stopwords for that language to the `stopwords` set in `_score_sentences`

### Adding a new Whisper model tier

- Append to `WHISPER_MODELS` list in `config.py` — Click choices auto-update

## Key Design Decisions

- **Lazy imports:** Heavy dependencies (`whisper`, `sounddevice`, `requests`, `docx`) are imported inside functions — keeps startup fast
- **Ollama fallback:** Unreachable Ollama silently falls back to extractive; no user error unless both methods fail
- **WAV-native recording:** Internal mic capture always saves to 16-bit PCM WAV; non-WAV inputs delegate to Whisper's own ffmpeg-backed loader
- **fp16=False:** Forces FP32 for CPU compatibility — most users won't have CUDA
- **Lazy model loading:** Whisper model downloads/loads on first transcription call, not at import time

## Testing

Tests use `pytest` + `pytest-mock`. Whisper, sounddevice, and requests are mocked — no audio hardware or model downloads needed.

```bash
pytest -v                          # all tests, verbose
pytest tests/test_summarizer.py    # single module
pytest -k "test_extractive"        # by name pattern
```

## Desktop App (`desktop/`) — the active surface

Tauri (Rust shell) + React/TypeScript webview, with the Python code above running as a
**sidecar** rather than being reimplemented. Capture happens in the webview (Web Audio →
16 kHz mono WAV); everything else goes over loopback HTTP to the sidecar.

```
App.tsx                        — record → transcribe → correct → notes → saved
  → lib/recorder.ts            — Web Audio capture; mic or system audio (getDisplayMedia)
  → lib/api.ts                 — sidecar client: handshake, /health, process, summarize, export
  → components/RecordPanel     — capture controls, level meter, language picker
  → components/TranscriptEditor— segment-level editing; notes regenerate from corrections
  → components/NotesView       — structured notes + engine provenance
  → components/HistoryPanel    — saved meetings: search, open, delete
```

Key points:

- **Note template is pinned to `meeting`** (`App.tsx`) and the picker is hidden — meetings
  are the only surface today. `consultation` still exists end-to-end (sidecar, `enrich()`,
  types); restoring the picker in `RecordPanel` is a UI-only change. The template gates
  `enrich()`, which is the only place the medical lexicon runs.
- **`highlightSpans` highlights only the terms it is given** — it does not consult the
  lexicon itself. A meeting carries no `medical_terms`, so it renders unhighlighted.
- **`API_VERSION` (`meeting_notes/server.py`, reported at `/health` as `api`) is the contract with whatever hosts the engine** (this Tauri shell, or Weave, which downloads the engine separately and refuses one whose number it does not know). Bump it when a route a host depends on changes shape or goes away; NOT on every release. `scripts/publish-notes-engine.ps1` in the Weave repo refuses to publish an exe whose `/health` disagrees with this file, so rebuild the sidecar after changing it.
- **Sidecar handshake:** the sidecar binds a random loopback port + bearer token and writes
  one JSON line to stdout. Never assume a fixed port. See `desktop/README.md`.
- `lib/types.ts` and `lib/medicalTerms.ts` are deliberate copies of the mobile versions,
  not a shared package — a drift should surface as a type error here.

Conventions: TS strict; `npx tsc --noEmit` and `npx vitest run` in `desktop/` must pass.

## Mobile App (`mobile/`) — on hold

> Paused while desktop is brought up to the corporate-meeting standard. Mobile is
> still hardcoded to the medical framing (`llm.ts` makes `medical_terms` a required
> schema field; highlighting runs unconditionally) — that gets the same template gate
> desktop now has, in a later pass. Do not treat its current behaviour as the target.

Expo (React Native) app, **fully on-device**: whisper.rn (whisper.cpp) for STT, llama.rn
(llama.cpp GGUF) for notes. No server needed. Android-first (custom native module).

```
Record screen (app/index.tsx)
  → wavRecorder.ts + WavRecorderModule.kt   — 16kHz mono WAV streamed to disk; optional PCM chunk events
  → live.ts                                 — live captions (RealtimeTranscriber fed by PCM chunks)
  → vad.ts                                  — Silero VAD speech regions; transcription skips silence
  → whisper.ts                              — full-quality pass per speech region; model + language from settings.ts
  → diarize.ts + DiarizationModule.kt       — speakers: sherpa-onnx (pyannote + CAM++ embeddings) native;
                                              falls back to pause/pitch clustering when module/models absent
  → llm.ts                                  — notes JSON (grammar-enforced via json_schema); map-reduce > 6k chars
  → medicalTerms.ts                         — EN/TL medical lexicon (parked; still applied unconditionally here)
  → export.ts                               — HTML → PDF (expo-print), email (expo-mail-composer), share sheet
```

| File | Purpose |
|------|---------|
| `mobile/lib/settings.ts` | Persisted app settings: language preset (auto/en/tl), live captions, auto-delete audio, model keys |
| `mobile/lib/models.ts` | Model catalogues + download manager; active models resolved from settings |
| `mobile/components/ConversationView.tsx` | Chat-style speaker turns, tap-to-rename speakers |
| `mobile/components/HighlightedText.tsx` | Renders text with medical terms highlighted (unconditional — unlike desktop) |
| `mobile/app/history.tsx` | Meeting list + search + detail view with email/share |

Conventions: TS strict; `npx tsc --noEmit` in `mobile/` must pass. tsconfig has `paths`
workarounds for whisper.rn's exports map. Native changes or new Expo modules require a
dev-client rebuild (`npx expo run:android` / `start-build.bat`) — Metro reload is not enough.

The Python CLI above still works standalone; `server.py` (FastAPI) exists but the mobile
app no longer depends on it.

## UI/UX Skill (installed)

The `ui-ux-pro-max` skill is installed at `.claude/skills/ui-ux-pro-max/`. Use it when designing any web UI, TUI improvements, or terminal output formatting. Run:

```bash
python .claude/skills/ui-ux-pro-max/scripts/search.py "meeting productivity SaaS tool" --design-system -p "Meeting Notes AI"
```
