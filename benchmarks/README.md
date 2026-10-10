# Benchmarks

Engine decisions here are measured, not assumed. Published vendor numbers are a
starting hypothesis; this directory holds what we actually reproduced.

```bash
# See what's available
meeting-notes bench --list

# Smoke test (10 utterances, fast)
meeting-notes bench --dataset fleurs-fil --limit 10

# The real Tagalog comparison
meeting-notes bench --dataset fleurs-fil \
    --engine nemo-tagalog \
    --engine faster-whisper:large-v3-turbo \
    --engine faster-whisper:medium

# Your own audio
meeting-notes bench --manifest taglish/manifest.jsonl --language tl \
    --terms taglish/terms.txt
```

Public datasets need the `bench` extra: `pip install -e ".[bench]"`.
Local manifests do not.

### Download sizes — plan for these

| Dataset | First-run download |
|---|---|
| `fleurs-fil` | **~1.05 GB** |
| `fleurs-en` | ~1.2 GB |
| `fleurs-ceb` | ~510 MB (Cebuano; pair with `--engine omnilingual-ctc`) |
| `common-voice-*` | Gated — accept terms on the Hub, set `HF_TOKEN` |

FLEURS embeds audio in a single parquet file per split, so **`--limit` does not
reduce the download** — it only shortens the scoring loop. Budget for the full
fetch once; it is cached afterwards under `~/.cache/huggingface`.

Set `HF_TOKEN` even for ungated datasets: unauthenticated Hub requests are
rate-limited and noticeably slower.

## What the metrics mean

| Metric | Reading it |
|---|---|
| **WER** | The standard. Compare against published figures. |
| **CER** | Only for comparing against papers that report CER (Qwen3-ASR does). **Not interchangeable with WER** — CER is normally the more forgiving of the two, so a 24% CER is *worse* than a 9% WER. |
| **Term recall** | Of the must-not-miss terms actually said, how many survived. Decides clinical usability, and routinely disagrees with WER — an engine can score well overall while dropping every drug name. |
| **RTF** | Compute-seconds per audio-second, measured after model warm-up. Below 1.0 is faster than real time. Live captions need comfortably below 1.0, since this excludes UI and capture overhead. |

Normalisation is fixed and documented in every report: lowercase, punctuation
stripped, whitespace collapsed, NFKC. **Word-internal hyphens and apostrophes
are preserved** so Tagalog affixed forms (`nag-meeting`, `i-refer`) score as
single tokens — splitting them would score `nag meeting` as correct and hide
exactly the failure mode Taglish support has to fix.

## Manifest format

JSONL, one object per line. Relative `audio` paths resolve against the
manifest's own directory.

```json
{"audio": "clips/001.wav", "text": "Nag-reklamo siya ng chest pain."}
{"audio": "clips/002.wav", "text": "Ubo at lagnat po.", "id": "002", "duration": 3.4}
```

`id` and `duration` are optional. Supplying `duration` avoids decoding the file
twice and makes RTF exact.

## Directory layout

```
benchmarks/
  README.md          — this file
  results/           — dated markdown reports (committed; they are the audit trail)
  taglish/           — the Taglish clinical test set (see below)
```

## `taglish/` — not populated yet

Deferred by decision, scaffolded so it can be dropped in later. When clinical
recordings exist, three tiers of roughly 40 minutes each:

| Tier | Content | Why |
|---|---|---|
| **A** | Tagalog-matrix with English medical insertions | The common case |
| **B** | English-matrix with Tagalog insertions | Specialist consults |
| **C** | Heavy switching, patient-side, informal register | The hard case |

Alongside the manifest:

- `terms.txt` — every drug name and condition that must not be missed
- `hybrids.txt` — every affixed form appearing in the audio (`nag-meeting`,
  `i-refer`, `na-diagnose`), so hybrid-form accuracy can be scored separately

⚠️ **What public datasets cannot tell you.** FLEURS is clean read speech;
Common Voice is read and slightly more varied. Neither is conversational, and
neither contains Taglish or clinical vocabulary. Treat public numbers as a
*ranking of engines*, not a prediction of clinic accuracy — expect real
consultations to be materially worse.
