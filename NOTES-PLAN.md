# Notes — plan

Turns the desktop app's **Record / History** into one **Notes** surface, Granola-style:
a notepad you type into while the app records, transcribes and diarizes in the background,
then an **Enhance** step that expands your own bullets from the transcript. Adds call
detection, long-recording chunking, automatic titles, and bug capture with screenshots for
the bug-fixing project.

Status: **decisions settled** (2026-10-10, see §12). **Target changed 2026-10-12: this lives in Weave's Recordings tile, not in this repo's Tauri app (§0).** Built so far: the long-meeting summarizer fix in `src/meeting_notes/summarizer.py` (map-reduce, `num_ctx`), which the engine keeps. A React Notes UI was built in the Tauri app and then reverted. Build order: proven stack first (Whisper, sherpa-onnx, Ollama/`llama3`), then try the newer models as an enhancement pass.

---

## 0. Where this lives: Weave's Recordings tile (decided 2026-10-12)

**Target.** The flyout's disabled "Recordings — coming soon" tile
(`FlyoutWindow.xaml`, `RecordingsBtn`) becomes **Notes**. Weave is a .NET WPF app; this repo's
Tauri/React desktop app is not where the feature ships. Everything below describes the *behaviour*;
this section says what runs where.

**Architecture (option 1): Weave is the shell, this repo's Python is the engine.**

| Piece | Where | Notes |
|---|---|---|
| Notes window, Live panel, prompt window, tray, call detection, screen capture, hotkey | **Weave (C#/WPF)** | Replaces the "Rust" parts of this plan. The decision to move capture out of the webview now means *into Weave*, written in C# (WASAPI mic + loopback). |
| Transcribe, diarize, link speakers, summarize, enhance, bug detection, encrypted storage | **Python engine** (this repo's sidecar) | Unchanged. Weave starts it as a local process on a random loopback port with a bearer token: the same handshake the Tauri shell uses today (`sidecar.py`). |
| LLM | **Ollama** for now; bundled llama.cpp later (see model shortlist) | Weave users do not have Ollama. Until a bundled runner exists, Enhance falls back to the extractive summarizer with a clear "install the notes AI" prompt. |

**Engine delivery: downloaded on first use, not shipped inside Weave. BUILT 2026-10-12** (Weave repo:
`Services/NotesEngineService.cs`, `publish-notes-engine.bat`). The bundled engine is ~235 MB
(PyInstaller). Putting that in the Velopack package would add it to every employee's install and every
update, including people who never open Notes, and a Weave release is all-or-nothing for the whole
fleet. Instead the Notes window offers **Set up Notes**, which downloads the engine to
`%LocalAppData%\ClaudeMeter\notes-engine`. Two things changed from the first draft of this plan:

- **Hosting is static files under the dashboard's `/downloads/notes-engine/`** (the TaskTopFight
  installer's precedent), not a new key-guarded route: no dashboard code, no deploy ordering.
- **The manifest is signed and the PUBLIC KEY is compiled in, not a hash.** A compiled-in hash would
  force a Weave release for every engine update. A signed manifest (ECDSA P-256) lets the engine update
  on its own while staying unforgeable, and carries the zip's SHA-256, its size, the unpacked size
  (the unpack ceiling) and the engine's API number. The private key is `.local\notes-engine-signing-key.txt`
  in the Weave repo and must be backed up: lose it and every installed Weave refuses every future engine.

The engine reports `API_VERSION` at `/health` (`meeting_notes.server`). Bump it when a route a host
depends on changes shape, not on every release. Weave supports a range and treats anything outside it as
"needs a newer Weave". Models (Whisper, diarization, Omnilingual) download on top, by the engine itself.

**What Weave already has that this plan was going to build:**
- tray, autostart and the keep-alive watchdog;
- **its own calls**: Weave knows when a call starts, the group name, and every participant's real
  name, so Weave calls need no mic-in-use guessing and no "Person N". The mic-in-use watcher (§5)
  stays for calls in other apps (Teams, Zoom, Meet);
- a recording file already written during a call (`CallWebHost`, `Documents\Claude Meter\Recordings`),
  so **the library already has content**: every existing recording gets a **Make notes** action;
- WebView2 / Windows Graphics Capture in the repo (the chat wallpaper), for the screen sampler.

**What does not carry over from this repo:** the React UI (`desktop/src`). The Notes list/detail,
Live panel and prompt are rebuilt in WPF following Weave's conventions (`ThemeService` slots,
`GlassBackdrop.Attach`, `Motion`, `AppIcon.Install`, `NotificationCount` badge rules). A Notes tile
badge follows the surveys rule: the server-side pending state, not "unseen locally".

**What changes in the phases (§10):** Phase 0 becomes *Weave: Recordings library + Notes window* (list
existing recordings, open one, empty state), with the engine download and **Make notes** next. Phases
1–6 keep their order. "Rust capture" reads as "C# capture in Weave".

**Cross-repo cautions:** the Weave tray release is all-or-nothing and the Mac port is separate, so
Windows first, behind the engine download, tested on a dev build (`run-dev.bat`; dev runs never
auto-update or push telemetry). Nothing here should add an opt-out-less data flow: audio and
transcripts stay on the machine, and the only outbound content is a user-sent bug report (§7.4).

---

## 0b. The bundled model: decided and built (2026-10-12)

Notes need a language model, and the only one the engine could use was Ollama, which no Weave user has.
The engine now brings its own, downloaded on demand (`meeting_notes/llm.py`; Weave's "Write better notes"
panel). **Default: Qwen3.5 4B, Q4_K_M, via llama.cpp b11429 (CPU).** Apache-2.0, 2.55 GiB, plus a 19 MB
runtime. Both files are pinned by SHA-256 and the model by Hugging Face revision.

**How it was chosen** (the three `Recordings/` fixtures, transcribed by the engine's own pipeline, the
engine's own prompt, JSON-constrained output, 8 threads on an i9-13900HX). This is a small, early
comparison on ONE meeting said three ways, not a leaderboard; it is enough to pick a default, not to
claim a ranking.

| | Qwen3.5 4B (2.55 GiB) | Gemma 4 E4B (4.64 GiB) | Qwen3.5 9B (5.29 GiB) |
|---|---|---|---|
| English | complete, faithful; 4 action items | terser; 2 decisions, 2 actions; invented a team name | most faithful; copies ASR garble ("Staging Pincess") |
| Taglish | good; **named the wrong person** for an action | good, but thinner; avoided naming | **named the wrong person** |
| Bisaya (garbled ASR) | English; invented a plausible decision | English; vague but mostly grounded | **wrote Bisaya notes, copying the garble into its decisions** |
| Valid JSON | 4/4 | 4/4 | 4/4 |

Speed on this machine ranged ~10-60 prompt tokens/s and ~3.5-9 generated tokens/s, **and varied up to 3x
between runs of the same model** (a hybrid-core CPU, with Defender scanning fresh multi-GB files). Treat
those as an order of magnitude, an upper bound for an employee laptop, not as a comparison between models.
The decisive differences were size and behaviour on noisy input, not speed. The 9B costs twice the disk and
RAM for no better result on this material; Gemma E4B is nearly twice the size for terser notes.

**The prompt mattered more than the model.** Feeding the 4B `Person N:` turns plus three rules (English
notes; use only the transcript and leave garbled parts out; name someone only when the transcript makes it
clear) removed the wrong attribution (it now works out "Marielle runs the cleanup script" from context, and
says "Person 2" when it cannot), stopped invented decisions on the Bisaya input, and cost ~150 prompt tokens.
Adopted in `TEMPLATES["meeting"]` and `summarizer.speaker_text`.

**Not decided by this:** Tagalog/Bisaya quality beyond three short clips; long meetings (the map-reduce path
is tested with a fake model, and a real one-hour run is untested); a larger model as an optional download for
fast machines; Gemma 4 12B (needs ~16 GB RAM, not benchmarked); Qwen3.6, which exists but whose small sizes
were not checked. **Re-run this comparison before changing `DEFAULT_MODEL`.**

---

## 0c. Long recordings: chunked, measured (2026-10-13)

Built (`chunking.py`, `diarize.SpeakerLinker`, `pipeline.py`) and run on a 21-minute file made of the fixtures
repeated ten times (3 real voices; repetition k must be labelled like repetition 0).

| | Single pass | Chunked (5 parts) |
|---|---|---|
| Total time | 1045 s | 1106 s (+6%) |
| Speakers found (3 real) | 7, counts scrambled | 8 with the first linker; **3** with the current one |
| Label agreement with repetition 0 | n/a | 96% (201/210) |

What went wrong first: the linker matched one-to-one within a part, so a diarizer's over-split fragments
(short interjections) could never rejoin their speaker and each became a new "person". Fingerprints separate
cleanly (same person 0.81-0.99, different people at most 0.59), so near-identical clusters (>= `MERGE_THRESHOLD`
0.78) now merge, biggest first, compared against the speakers as they sounded BEFORE the part (folding
matches in first drags a centroid towards them). Decision: keep chunking; it is no slower in practice, bounds
memory, and gives progress and cancel. Offline tuning: capture each part's fingerprints once, replay the linker.

## 0d. Notepad and Enhance, built in Weave (2026-10-13)

Built as designed in section 8 with these decisions: the notepad is keyed by an opaque id and stored encrypted in the
engine (`PUT/GET/DELETE /notepads/{key}`); Enhance is a background job (`POST /jobs/enhance`) that saves `my_notes`
and `enhanced` onto the meeting; the output tree is assembled from the person's lines so the "your structure wins"
rule cannot be broken by the model; added lines survive only if their cited transcript time is real and shares words
with what was said there; decisions and action items come from the existing notes. Additive routes are advertised in
`/health` `features`, not by bumping the API number. Not yet: the Live panel inside the call window, click-to-play
audio, Word export of the enhanced notes.

## 1. Vocabulary

| Term | Means |
|---|---|
| **Note** | One session: audio chunks + transcript + speakers + *my notes* + *enhanced notes* + bugs. Replaces "meeting"/"recording" in the UI. The storage table keeps its `meetings` name, so no data migration is needed for the rename. |
| **My notes** | What the person typed during the session. Timestamped per line. **Never rewritten by the AI.** |
| **Enhanced notes** | Derived from *my notes* + transcript. Can always be regenerated. |
| **Chunk** | A slice of audio (about 5 min, cut at a pause) that is transcribed and diarized on its own. |
| **Live panel** | The small window that floats beside the call while a Note is recording. |
| **Notes panel** | The main window: the list of Notes plus the selected Note's detail. |
| **Bug** | An issue raised in the session, with its transcript excerpt, speaker, time and screenshots. |

---

## 2. What exists today, and what has to change

| Today | Problem for this feature | Change |
|---|---|---|
| `recorder.ts` captures in the **webview**, either the mic **or** system audio (`getDisplayMedia`), holding everything in memory until Stop | (a) Can't start from a notification: `getDisplayMedia` opens a picker every time. (b) Can't record both sides of a call. (c) An hour of audio sits in RAM, and a webview crash loses all of it. | Move capture to **Rust** (`cpal`, WASAPI): mic plus **loopback** of the default output device, written to disk as chunk files while recording. **Decided.** This reverses the "capture in the webview" decision at the top of `recorder.ts`, so update that comment and `CLAUDE.md` when it lands. |
| `/process` takes one whole file, transcribes, diarizes and summarizes in one request | Long recordings: one Whisper pass and one diarization over an hour of audio is slow and the clustering degrades. Nothing is visible until all of it finishes. | Sidecar **job queue**: each chunk is processed as soon as it closes. The UI shows progress per chunk. |
| `_build_prompt` keeps `transcript[:4000]`, and Ollama is called without `num_ctx` (the default context is 2–4k tokens) | **Notes for anything longer than about 5 minutes describe only the opening.** This is a bug today, independent of this feature. | Map-reduce summarization (§6.3). Set `num_ctx` explicitly. |
| `storage.py`: "Audio is never persisted" | Background processing needs chunks on disk until they are processed, and bug screenshots are new persisted content. | Chunks and screenshots are encrypted at rest with the existing `Vault` (add `encrypt_bytes`/`decrypt_bytes`) and **kept until the user deletes them** (§9). Rewrite the `storage.py` docstring and `PRIVACY.md` to say so. |
| Title comes from the LLM's `title` field, after everything else | Fine, but the list needs a name from the moment a Note starts | Provisional title from the call app's window title ("Sprint sync \| Microsoft Teams" becomes "Sprint sync"), else "Teams call · 10:30". Replaced by the topic once the notes are ready, **unless the user renamed it** (a `title_locked` flag). |
| Speakers are "Person N" within one file | Chunks diarize independently, so chunk 2's "Person 1" isn't chunk 1's | Cross-chunk speaker linking (§6.2). With two-track capture the mic track is always **You**. |

---

## 3. User flows

### 3.1 In a call (automatic prompt)
1. The app runs in the **tray** and starts with Windows (Tauri `tray-icon` + `tauri-plugin-autostart`). Closing the window hides it to the tray.
2. The call watcher (§5) sees another app start using the mic: Teams, Zoom, Meet in a browser, Slack, Weave.
3. A small **prompt window** appears bottom-right: *"Looks like you're in a Teams call. Take notes?"* with **Take notes**, **Not now** and **Never for Teams**. It is our own window, not a Windows toast, because toasts from unpackaged apps can't reliably carry buttons and get swallowed by Focus Assist. It never steals focus and goes away by itself after 30 s or when the call ends.
4. **Take notes** starts recording **mic + system audio** and opens the **Live panel**.
5. The call ends (mic released by the call app for 20 s), or the user presses Stop. Recording stops, the last chunk closes, and processing finishes in the background. A tray balloon says *"Notes ready: Sprint sync"* when done.

### 3.2 No call (manual)
- Tray menu or Notes panel: **New note**. With no call detected it records the **mic only** (the room around you). Everything else is identical.
- Import a file stays: it goes through the same chunked pipeline.

### 3.3 Live panel (during recording)
A compact, always-on-top, resizable window that docks beside the call:
- Timer, two level meters (You / Others), Pause, Stop.
- **Notepad**: plain text with Markdown-ish bullets. Every line remembers when it was typed (seconds from session start).
- **Flag bug** button and global hotkey (`Ctrl+Shift+B`): marks "a bug is being discussed or shown now" and grabs a screenshot immediately (§7).
- **Live transcript** (§8a): a collapsible strip under the notepad showing what's being said a few seconds behind, with speakers. Collapsed by default because it competes with typing. Clicking a live line drops it into the notepad as a quote with its timestamp.

### 3.4 Notes panel (main window)
- **Left: list of Notes**, newest first, grouped by day. Each row shows title (topic), app icon (Teams/Zoom/Room), duration, and a status chip: `Recording`, `Processing 3/7`, `Ready`, `Failed – retry`. Search covers title, my notes, enhanced notes and transcript (the existing decrypt-and-match search).
- **Right: the selected Note**, in tabs:
  - **Notes**: enhanced notes. Your own lines look different from AI-added lines (Granola's black-vs-grey idea), so you can always see what you said versus what was inferred. **Enhance** / **Re-enhance** button.
  - **My notes**: the raw notepad, editable.
  - **Transcript**: the existing `TranscriptEditor` (speaker turns, rename speakers, edit segments), available per chunk **as soon as each chunk is done**, before the whole Note is ready.
  - **Bugs**: the bug cards (§7) with an **Export** action.
- A Note that is still processing is openable. Finished parts show, unfinished parts say so.

---

## 4. Architecture

```
┌──────────── Tauri (Rust) ─────────────┐        ┌──────────── Python sidecar ─────────────┐
│ call_watch   mic-in-use poll (2 s)    │        │ SessionQueue (one worker thread)        │
│ capture      cpal: mic + loopback     │ chunk  │   per chunk: transcribe → diarize        │
│              → VAD-cut chunk WAVs ────┼──────► │   per session: link speakers → title     │
│ screen       frame sampler + flag hot │ files  │                → map-reduce notes        │
│              key → encrypted PNGs     │        │                → enhance → bug detect    │
│ tray, prompt window, live window      │        │ MeetingStore v2 (encrypted)              │
└──────────────┬────────────────────────┘        └──────────────▲──────────────────────────┘
               │ events (level, chunk-closed, call-started)      │ loopback HTTP + bearer token
               ▼                                                 │ (existing handshake)
        React webview: Notes panel / Live panel / Prompt ────────┘
```

- **Rust owns devices** (audio, screen, hotkey, tray, call detection). **Python owns models and storage.** The webview only displays and edits. This follows the existing split: storage stays on the Python side so one process owns the data end to end.
- Rust writes each chunk to `data_dir/sessions/<id>/chunk-NNN-{mic,sys}.wav` and calls `POST /sessions/{id}/chunks`. The sidecar encrypts it in place once it has read it. *(Alternative: Rust streams PCM to the sidecar over HTTP. Rejected because files survive a sidecar restart and are trivially resumable.)*
- **Crash-safe by construction:** a crash loses at most the open chunk. On startup the sidecar re-queues any chunk without a result, and the UI shows those Notes as `Processing`.

### 4.1 New sidecar routes

| Route | Purpose |
|---|---|
| `POST /sessions` | Create a Note (source: `call`/`room`/`import`, app name, provisional title, started_at). Returns id. |
| `POST /sessions/{id}/chunks` | Register a closed chunk (index, start/end s, paths). Queues it. |
| `POST /sessions/{id}/stop` | No more chunks. Queues the session-level steps. |
| `PUT /sessions/{id}/my-notes` | Save the notepad (lines with `t` seconds). Autosaved every few seconds from the Live panel. |
| `POST /sessions/{id}/enhance` | (Re)run Enhance. |
| `POST /sessions/{id}/bugs/flag` | A manual flag (t, screenshot ids). |
| `GET /sessions/{id}` | Full Note, with per-chunk status. |
| `GET /sessions/{id}/events` | SSE progress stream (chunk done, stage, errors). Polling `GET` is the fallback. |
| `GET /sessions/{id}/screenshots/{sid}` | Decrypted PNG for display. |
| `POST /sessions/{id}/bugs/{bid}/send` | Deliver one bug report to the bug-fixing app's API (§7.4). Returns the issue URL. |
| `POST /sessions/{id}/bugs/{bid}/export` | Save the same report as a JSON + PNG folder. |
| `GET /sessions/{id}/live` | SSE stream of live transcript lines (§8a). |
| `GET /sessions/{id}/audio?from=&to=` | Decrypted audio slice for click-to-play. |
| `DELETE /sessions/{id}/audio`, `…/screenshots?unattached=1` | User-initiated deletes (§9). |

The existing `/process`, `/summarize` and `/meetings*` routes stay for the CLI and old records.

### 4.2 Storage — schema v2 (`_MIGRATIONS` entry 2)

```sql
ALTER TABLE meetings ADD COLUMN status      TEXT NOT NULL DEFAULT 'ready';  -- recording|processing|ready|failed
ALTER TABLE meetings ADD COLUMN source      TEXT NOT NULL DEFAULT 'import'; -- call|room|import
ALTER TABLE meetings ADD COLUMN app         TEXT NOT NULL DEFAULT '';       -- "Teams", "Zoom", ""
CREATE TABLE chunks (
  session_id TEXT NOT NULL, idx INTEGER NOT NULL,
  start_s REAL NOT NULL, end_s REAL NOT NULL,
  status TEXT NOT NULL,              -- queued|done|failed
  result BLOB,                       -- encrypted: segments + per-speaker embeddings
  PRIMARY KEY (session_id, idx));
CREATE TABLE screenshots (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
  t REAL NOT NULL, path TEXT NOT NULL,    -- encrypted file on disk
  kept INTEGER NOT NULL DEFAULT 0);       -- 1 = attached to a bug
```

The encrypted `payload` gains `my_notes`, `enhanced`, `bugs`, `title_locked`. The app name stays in clear metadata like `language`, because the list sorts and filters on it. It doesn't identify anyone.

---

## 5. Call detection (Windows)

Windows already tracks which app is using the mic. That's what drives the mic icon in the taskbar:

```
HKCU\Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone
  \<PackageFamilyName>              ← packaged apps (new Teams is MSIX)
  \NonPackaged\<exe path, \ → #>    ← Zoom.exe, chrome.exe, ClaudeMeter.exe …
     LastUsedTimeStart / LastUsedTimeStop   (in use ⇔ Stop == 0 or Stop < Start)
```

- Poll every 2 s from Rust (`winreg`). Ignore our own exe.
- Map exe or package name to a friendly name: Teams, Zoom, Slack, Webex, Weave (`ClaudeMeter.exe`), Discord. A browser shows as *"a call in Chrome/Edge"*. A browser using the mic is usually Meet or Teams-web, but could be dictation. That's acceptable because the prompt only asks.
- **Provisional title:** read the call app's foreground window title (`EnumWindows` for that PID) and strip the app suffix.
- **Call ended:** mic released by that app for more than 20 s. That absorbs mute/unmute, since some apps release the device on mute.
- Debounce: one prompt per call; *Not now* stays quiet until that app releases the mic; *Never for X* is a setting.
- macOS later: there is no equivalent registry. It would need the Core Audio "device is running somewhere" property and gets no per-app name. Out of scope until desktop-Windows is settled.

---

## 6. Long recordings: chunk, transcribe, diarize, link, summarize

### 6.1 Chunking
- Target **5 min** per chunk. Within ±30 s of the target, cut at the **longest pause** found by Silero VAD (already in the stack). Never cut mid-word. A hard cap at 6 min handles nonstop talk.
- Both tracks (mic, sys) are cut at the **same** timestamps, so a chunk is a pair.
- Each chunk's segments carry **absolute** session time (`start + chunk.start_s`).

### 6.2 Diarization and cross-chunk speaker linking
- **Mic track = You.** Transcribe it but don't diarize it. Label it with the user's name from settings.
  - With speakers rather than a headset, the mic also hears the far end. Drop mic segments whose time overlaps a louder sys segment carrying the same words (simple energy and text-overlap check), so lines aren't duplicated.
- **Sys track** gets diarized per chunk with the existing `assign_speakers` (pyannote + CAM++ + `_refine_clusters`).
- **Linking:** per chunk, compute each local speaker's **centroid embedding** from segments of at least `ANCHOR_SECONDS`. `segment_voices` already embeds them. Store centroids in the chunk result. At session end, run agglomerative clustering over all chunk centroids (cosine, threshold tuned like `CLUSTER_THRESHOLD`, starting at 0.6). This gives global "Person N". Re-label every segment. Also do it **incrementally** after each chunk, so live captions and the Transcript tab keep stable names during the meeting.
- Speaker names the user typed are keyed to global labels, as today (`speaker_names`).
- **Room mode** (no call) has only a mic track, so it gets diarized like sys.

### 6.3 Summarization (map-reduce) — fixes the 4,000-char bug
1. **Map**, per chunk as it finishes: a short JSON of points, decisions, action items and candidate bugs, each with timestamps.
2. **Reduce**, at stop: merge chunk JSONs into the final `title / summary / key_topics / decisions / action_items`. Dedupe across chunks and keep owners ("Ana: send the build").
3. Set `num_ctx` in the Ollama payload, and size chunks so map prompts fit.
4. The **title** comes from the reduce step: *"the topic of the meeting, 3–7 words, no date"*. It replaces the provisional title unless `title_locked`.
5. The extractive fallback runs per chunk and concatenates, so a missing Ollama still yields something for long Notes.

### 6.4 Throughput
One worker thread, because models are resident and CPU-bound. Chunk processing must stay faster than real time, or the queue grows during long meetings. Measure on this machine with `meeting-notes bench`. If it falls behind, use a smaller Whisper tier for live chunks and an optional "re-transcribe at full quality" afterwards.

---

## 7. Bug capture (for the bug-fixing project)

**The hard constraint:** you can't screenshot the past. By the time the transcript says "see, it throws here", that moment is minutes old. So screen capture has to run **while recording**, and the bug step **picks** frames afterwards.

### 7.1 Screen frames
- When a Note starts from a call, also sample the screen the call window is on: Windows Graphics Capture through Rust (`windows-capture` crate), or DXGI duplication as fallback.
- Every 2 s, downscale and compare to the last kept frame with a perceptual hash. **Keep only frames that changed.** A static slide is one frame, not 900. Store kept frames encrypted, full resolution.
- Frames stay on disk with the Note. Frames attached to a bug are marked `kept = 1`, so the user can delete "unattached frames only" to reclaim space without losing bug evidence.
- **On by default for calls** (decided). It can be turned off per Note (Live panel toggle) and globally in Settings. Room notes (no call) don't sample the screen. The Live panel shows a visible "Screen capture on" marker whenever it runs, so it's never silent.
- Because frames are kept (§9), a bug can also be **added after the fact**: select transcript lines → *Add as bug*, and the card pulls frames from that time just as detection would.

### 7.2 Two ways a bug is found
1. **Manual flag** (Live panel button or `Ctrl+Shift+B`): highest confidence. Grabs a full-resolution screenshot *right now* plus the last 3 changed frames.
2. **Automatic detection**, per chunk in the map step: the LLM marks transcript spans that describe a defect ("it crashes when…", "the total is wrong", "nag-error siya", "dili mo-load"), returning `{start, end, summary, confidence}`. Prompt examples should include Taglish/Bisaya phrasing, since that's how the team talks. Candidates below the confidence threshold show as *"Possible bug, keep or dismiss?"* rather than being dropped or exported.
3. Spans that overlap a manual flag merge into it.

### 7.3 Building the bug card
For each bug:
- **Frames:** the kept frames within `[start − 30 s, end + 10 s]` of the span, plus any manual-flag shots.
- **Screenshot details:**
  - **OCR**, offline and built in: `Windows.Media.Ocr` (WinRT, via Rust). Error dialogs, stack traces, URLs and field values become searchable text.
  - **Vision description** (decided: offered). A local multimodal model in Ollama (e.g. `gemma3` or `qwen2.5vl`) answers *"What application and screen is this? What error or wrong state is visible?"* It stays offline. The several-GB `ollama pull` is offered once, the first time a bug card is built, with its size stated, and can be started from Settings. Until it's installed, cards carry OCR only and say so. Absent is a state, not an error. The model is run with OCR text in the prompt, so it describes the error rather than guessing at blurry text.
- **Issue details** (LLM, from the span ± context): title, description, **steps to reproduce as described**, expected vs actual, environment mentioned (build, browser, module), **reported by** (speaker), and who said they'd fix it if anyone did.
- User edits everything on the Bugs tab before export. Nothing is exported automatically.

### 7.4 Hand-off: an API on the bug-fixing app (decided)
The coworker's app creates the GitHub issue, so this app's job ends at **delivering one bug report
to that app's API**. This app never talks to GitHub and never holds a GitHub token.

**The bug report:** one per bug, the canonical form (also what *Save as file* writes, for when the
API is unreachable or not set up):

```jsonc
{
  "schema": "meeting-notes.bug/1",
  "note": { "id": "…", "title": "Sprint sync", "started_at": "2026-10-10T02:30:00Z", "app": "Teams" },
  "bug": {
    "id": "8f3c…",            // stable per bug: also the idempotency key
    "title": "Invoice total ignores discount",
    "description": "…",
    "steps": ["Open an invoice with a 10% discount", "Click Recompute"],
    "expected": "Total minus 10%",
    "actual": "Full total",
    "environment": "staging build 4.2",
    "reported_by": "Ana",
    "assignee_mentioned": "Marco",
    "source": "manual|detected",
    "confidence": 0.86,
    "transcript": [{ "t": 754.2, "speaker": "Ana", "text": "…" }],
    "screenshots": [{ "file": "01.png", "t": 751.0, "ocr": "…", "description": "…" }]
  }
}
```

**The call** (proposed contract; confirm with your coworker):

```
POST {bugApiBaseUrl}/api/bug-reports
Authorization: Bearer {bugApiKey}
Idempotency-Key: {bug.id}
Content-Type: multipart/form-data
  report      = the JSON above (application/json)
  01.png …    = screenshots, part names matching screenshots[].file

201 { "issueUrl": "https://github.com/org/repo/issues/123", "issueNumber": 123 }
200 { same }   when this Idempotency-Key was already received (no second issue)
4xx { "error": "…" }  shown on the card verbatim
```

- **Send is per bug, by the user, after review** on the Bugs tab. A *Send all reviewed* button
  covers a meeting with many. Nothing is sent automatically. This is the app's first outbound call
  that carries meeting content, and it should stay an explicit act.
- The returned `issueUrl` is stored on the card. The card then shows *Sent → #123* and opens the
  issue on click. Re-sending is safe because of the idempotency key. An edited card after sending
  is offered as *Send update*, which needs a `PUT /api/bug-reports/{id}` on their side
  (**ask your coworker**).
- **Settings:** `bugApiBaseUrl` + `bugApiKey`. The key is stored encrypted by the `Vault`, never in
  plain settings. A *Test connection* button calls `GET /api/bug-reports/health`.
- **Failure is visible, never silent:** a failed send leaves the card in `Not sent – retry` with the
  server's message. Offline or no API configured → *Save as file* writes the JSON + PNGs folder.
- Multipart rather than base64-in-JSON: a few full-resolution screenshots base64'd are megabytes
  of JSON, and the receiver needs the files as files to attach them to the issue anyway.

**Questions for your coworker:** the URL and auth scheme; whether their side already has a report
shape (adopt theirs if so; this one is a proposal); whether they want updates (`PUT`); how screenshots
reach GitHub (their app uploads them, since GitHub issues can't take attachments through the
REST API directly, it typically commits them or hosts them itself).

---

## 8. Enhance (the Granola part)

**Input:** *my notes* (lines with `t`), the transcript with speakers, and the reduce-step JSON.

**Rules the prompt enforces, and the parser checks:**
1. **Your structure wins.** Every heading and bullet you wrote appears, in your order and your wording.
2. Under each of your lines, add detail from the transcript **near that line's `t`** (±3 min window first, then the whole Note). That's why lines are timestamped: "pricing?" typed at 12:40 means the pricing discussion at 12:40, not the one at 50:00.
3. Then append a section for what you didn't write down: decisions, action items with owners, open questions.
4. Every AI-added line carries a `source_t` (transcript timestamp). Clicking it jumps to the Transcript tab there. That makes the notes checkable rather than trusted.
5. Output is structured JSON (`[{kind: "mine"|"added", text, source_t?, children}]`), not free Markdown, so the UI can style provenance and re-render it. Markdown/DOCX export renders from it.

With an empty notepad, Enhance degrades to today's notes, from the reduce step. Re-enhance after editing *my notes* or the transcript, as `onRegenerate` does now.

---

## 8a. Live transcription (good to have)

The chunk pipeline (§6) produces the **accurate** transcript but runs minutes behind. Live
transcription is a **second, fast pass** whose only job is to show what's being said now. When a
chunk's full-quality result lands, it **replaces** the live lines for that time range. The live text
is never stored as the record.

- **Where:** a second sidecar worker thread with its own small model, separate from the chunk
  worker, so captions don't wait behind a 5-minute chunk and chunks don't wait behind captions.
- **How:** Rust sends the sidecar short speech windows as they happen: Silero VAD utterances,
  cut at 8 s max. Mic and system tracks are captioned separately, so "You" vs "Others" is known
  live, without diarization. Others are split into Person N only when the chunk pass finishes.
- **Engine options, in order of preference:**
  1. **faster-whisper `base`/`small` int8 on short windows.** Already a dependency. About 1–3 s
     latency per utterance on CPU, and handles Taglish no worse than today's Whisper path.
  2. **sherpa-onnx streaming zipformer** (true streaming, words appear as spoken). sherpa-onnx is
     already a dependency for diarization and Omnilingual. But the streaming models are mostly
     English/Chinese, so Tagalog/Bisaya live captions would be poor. Option for English-only notes.
  - Language follows the Note's language setting. `ceb` uses the Omnilingual CTC model, which
    is non-streaming but fast enough on short windows.
- **CPU budget:** both workers on a laptop CPU during a call is the main risk. The live worker
  **yields**: if the chunk queue is more than one chunk behind, live captions pause with a visible
  *"Live captions paused to keep up"* rather than let the accurate pipeline fall behind. Live is the
  part that can be sacrificed.
- **UI:** the Live panel strip (§3.3), and the Transcript tab of a Note still recording, which
  shows finished chunks plus a greyed live tail.
- **Uses beyond display:** live lines feed bug detection early. A "crash/error/nag-error" phrase
  in a live line prompts *"Sounds like a bug, flag it?"* in the Live panel, which grabs the screen
  *now*, while it's still showing. That makes the manual flag easier to remember to use.

---

## 9. Privacy and consent

The app stays offline: no new network calls except the one-time model downloads that exist today, plus the optional vision model.

- **Recording other people in a call.** RA 10173 applies (see `PRIVACY.md`). The prompt window and the Live panel say *"Let the others know you're taking notes."* Optional: a one-click *copy notice to clipboard* to paste into the call chat. That's Granola's "Heads up" idea without needing their calendar.
- **Screenshots capture whatever is on screen**, including other people's shared screens and your own chat. Encrypted at rest, and visible on the Bugs tab before anything leaves the machine. Only a reviewed, user-sent bug ever leaves it (§7.4).
- **Retention (decided): nothing is deleted automatically.** Audio chunks and screen frames stay
  on this machine, encrypted, until the user deletes them. Per Note: *Delete audio*, *Delete
  unattached screenshots*, or *Delete note* (everything). In bulk: Settings → Storage shows the
  space used per Note and total, with the same three actions.
  - **Disk cost:** 16 kHz mono 16-bit is ~1.9 MB/min per track, so a 1-hour call with two tracks
    is ~230 MB as WAV. Store chunks as **FLAC** once processed: lossless, so it doesn't break the
    "no lossy compression before ASR" rule in `recorder.ts`, and roughly half the size. Frames add
    tens to low hundreds of MB for an hour of screen share.
  - What keeping audio buys: **click-to-play** on any transcript line (Phase 2), and
    **re-transcribe** a Note later with a better model or a corrected language, which matters
    for Bisaya as `RESEARCH.md`'s engines improve.
  - This reverses "Audio is never persisted" in `storage.py` and `PRIVACY.md`. Both need
    rewriting, and an existing user should see the change explained once, not discover it.

---

## 10. Phases

Each phase ships on its own and leaves the app working.

| # | Phase | Contents | Done when |
|---|---|---|---|
| **0** | **Rename + long-notes fix** | Record/History → **Notes** (list + detail layout, tabs). Map-reduce summarizer, `num_ctx`. | A 30-min import produces notes covering the whole meeting. All existing records open in the new panel. |
| **1** | **Background chunked pipeline** | Rust capture (mic, mic+loopback) writing VAD-cut chunks. Sidecar session queue and routes. Schema v2. Per-chunk transcribe/diarize. Cross-chunk speaker linking. Status chips. Resume after crash. Provisional title, then topic title. | A 60-min synthetic meeting (concatenated `Recordings/` fixtures, known speakers) records, processes chunk by chunk, keeps speakers consistent across chunks, and survives killing the app mid-recording. |
| **2** | **Live panel + Enhance** | Floating notepad with timestamped lines, autosave. Enhance with provenance and `source_t` jump. Click-to-play audio on transcript lines. Storage settings + deletes. | Bullets typed during the fixture meeting come back expanded, in the user's order, every added line linked to the transcript. |
| **3** | **Call detection + tray** | Tray, autostart, close-to-tray. Mic-in-use watcher. Prompt window. Auto-stop on call end. "Notes ready" balloon. | Joining a Teams/Zoom/Meet test call raises the prompt within ~5 s, *Take notes* records both sides, hanging up finishes the Note. |
| **4** | **Bug capture + hand-off** | Screen sampler (on by default for calls). Flag button + hotkey. Detection in map step. *Add as bug* from transcript. OCR. Vision description with one-time model download. Bugs tab. **Send to the bug-fixing app's API**, save-as-file fallback. | A scripted call where someone screen-shares an error yields a bug card with the right screenshot, OCR'd error text, steps and reporter. *Send* creates exactly one GitHub issue through the coworker's app even when pressed twice. |
| **5** | **Live transcription** (good to have) | Second sidecar worker, short-window captions, replace-on-chunk, yield under load, "sounds like a bug" prompt. | Captions appear within ~3 s on the fixture meeting, and the chunk queue never falls more than one chunk behind with captions on. |
| **6** | **Polish** | Re-transcribe a Note. Outlook/ICS calendar for titles and attendee names. Weave integration (§11). Mobile parity, later, per `CLAUDE.md`. | — |

**Tests**, following project conventions (`pytest`, `vitest`, no hardware or models in unit tests):
- VAD cut-point selection, chunk time offsets, cross-chunk linking (synthetic embeddings with known identities), mic/sys duplicate suppression.
- Map/reduce/enhance/bug JSON parsing, including malformed LLM output (the existing `_parse_llm_response` tolerance).
- Enhance invariant: every *my notes* line is present, in order, unaltered.
- Schema v2 migration from a v1 database.
- Benchmark fixture: a long concatenation of `Recordings/` (English/Taglish/Bisaya) with ground-truth speakers, for linking accuracy and real-time factor.

---

## 11. Suggestions beyond the brief

1. **Two-track capture is the biggest single win.** "You" becomes certain rather than inferred. Diarization only has to separate the far end. Action items like "I'll send it" get the right owner.
2. **The manual Flag bug button matters more than detection.** Detection finds what was *said*. The flag catches what was *shown*, which is what a bug fixer needs. Detection is the safety net.
3. **Provenance on every AI line** (`source_t`). Notes you can verify in one click get trusted, and notes you can't get re-checked by hand, which defeats the purpose.
4. **Weave calls can skip detection entirely.** The tray app in the other repo already knows when a call starts, its group name and every participant's real name. A tiny local hand-off (Weave → `http://127.0.0.1` on our handshake port, or a named pipe) would give this app the exact title and real speaker names instead of "Person N". Worth doing after Phase 3. It needs a change in both repos.
5. **Agree the bug-report contract with your coworker before Phase 4**, and if they can, build a stub of their endpoint first. The idempotency key is the one part not to skip: a retry after a timeout must not open a second GitHub issue.
6. **Fix the 4,000-char truncation now (Phase 0)** even if nothing else here proceeds. Every long meeting today gets notes about its first few minutes.
7. **Live transcription can be pulled ahead of Phase 5.** The short-window worker depends only on Phase 1's Rust capture, not on Enhance or bugs, if a visible "it's listening" is wanted early.

---

## 12. Decisions (2026-10-10)

| # | Question | Decision |
|---|---|---|
| 1 | Bug hand-off | **API** on the coworker's app, which creates the GitHub issue. Contract proposed in §7.4. Their URL/auth/report shape still to confirm. |
| 2 | Audio retention | **Keep locally; the user deletes when they want.** Applied to screen frames too (§9). |
| 3 | Capture in Rust | **Yes.** |
| 4 | Screen sampling during calls | **On by default**, visible marker, toggle per Note and globally. |
| 5 | Vision model | **Yes**, offered as a one-time download, OCR-only until installed. |
| + | Live transcription | **Good to have**: Phase 5, design in §8a. |

Still open, for the coworker: endpoint URL and auth, whether their side already defines a report
shape, whether updates (`PUT`) are wanted, and how screenshots get into the GitHub issue.
