# Model Reference — Meeting Notes AI

> Two products, two runtimes. **Desktop** (Python, faster-whisper + sherpa-onnx)
> is documented immediately below. **Mobile** (Expo, whisper.rn + llama.rn)
> starts at [How model selection works now](#how-model-selection-works-now).
>
> This file records **what ships and why**. For the full evaluation behind those
> choices — engines rejected, benchmarks, datasets, and how to refresh the
> research — see [`RESEARCH.md`](RESEARCH.md).

---

# Desktop

## Engine routing

Language mode selects the engine — see `backends/router.py`.

| Mode | Engine | Model |
|---|---|---|
| `en` | faster-whisper (CTranslate2) | `large-v3-turbo`, int8 on CPU |
| `tl` | NeMo FastConformer via sherpa-onnx | `NCSpeech/stt_tl_fastconformer_hybrid_large` |
| `ceb` | **Meta Omnilingual ASR CTC via sherpa-onnx** | `omniASR_CTC_300M` int8 (sherpa-onnx `…-300M-ctc-int8-2025-11-12`), Apache-2.0 |
| `auto` | faster-whisper | `large-v3-turbo` |
| `taglish` | *not implemented* | Two-pass merge of the above — Phase 7 |

`tl` falls back to faster-whisper with a visible warning when the NeMo model
has not been exported. `ceb` falls back to faster-whisper decoding **as
Tagalog** (Whisper has no Cebuano token) when sherpa-onnx is missing.
`TranscriptionResult.backend` records which engine actually ran.

**Omnilingual (`ceb`)** — added 2026-09-07. ~292 MB download on first use,
auto-fetched to `models_dir()/omnilingual/` (plus Silero VAD, 0.6 MB). No
language token: one model decodes Bisaya, Tagalog and English in the same
utterance, and the `language` on the result is the *requested* tag, not a
detection. Meta publishes per-language numbers only for the 7B model; no
Cebuano WER for this one yet. Run `meeting-notes bench --dataset fleurs-ceb
--engine omnilingual-ctc` to get the first figure. See RESEARCH.md §6a, §9b.

**[MEASURED] 2026-09-07, this machine, `Recordings/Bisaya.mp4` (70 s, three
speakers, Bisaya–English project sync):**

| What | Result |
|---|---|
| Decode speed | 30 s of audio in 9.3 s on 4 CPU threads (RTF ≈ 0.3) |
| Bisaya | Intelligible: *"para dili maguba ang layout sa gagmay nga monitors"*, *"basta limpio lang ang data migration makasugod na ta sa UAT ugma"* |
| English jargon | Mangled phonetically: "custom web service" → *coston webe service*, "user acceptance testing" → *usor acceptane tusting* |
| Script drift | Short/noisy segments came out in Arabic script; now stripped by `clean_latin_text`, so those segments drop rather than pollute the notes |
| VAD | Must be fed in 512-sample blocks — pushing the whole file yielded one 0.3 s region and an empty transcript. Fixed in `drain_vad`, also applied to the Tagalog backend |

Good enough to *test* Bisaya meetings today; not good enough to ship without
the domain fine-tune described in RESEARCH.md §9c.

## Why these

**faster-whisper over openai-whisper.** Same weights, same accuracy, ~4× faster
on CPU at int8 and roughly half the memory (~4 GB vs ~10 GB for large-v3).
Dropping PyTorch also cuts the eventual desktop installer from ~1.5 GB to a few
hundred MB.

**`large-v3-turbo` over `large-v3`.** large-v3 runs at well under real time on
CPU — a published figure has it around 2.5 s of compute per second of audio on
an i9-12900K, which would make a 1-hour consult take 2.5 hours. Turbo keeps the
full large-v3 encoder and prunes the decoder from 32 layers to 4: roughly
large-v2 accuracy, several times faster. It is the only large-class option that
is viable CPU-only.

**NeMo FastConformer for Tagalog.** 115M params, ~520h of Tagalog training
data, CC-BY-4.0. Published 9.34% WER on FLEURS `fil_ph` and 16.10% on Magic
Data Tech conversational — the best Tagalog figures available, by a wide
margin, from any model we found.

| Model | FLEURS Filipino | Params | CPU viability | License |
|---|---|---|---|---|
| NCSpeech Tagalog FastConformer | 9.34% **WER** | 115M | Very fast | CC-BY-4.0 |
| Qwen3-ASR-1.7B | 24.29 **CER** | 1.7B | 0.49× realtime — too slow | Apache-2.0 |
| Qwen3-ASR-0.6B | 36.10 **CER** | 0.6B | ~2.5× realtime | Apache-2.0 |
| Whisper large-v3 | not reported | 1.55B | ~0.4× realtime | MIT |
| Whisper large-v3-turbo | not reported | 809M | ~7× faster than v3 | MIT |

⚠️ These are **vendor/paper-reported and not directly comparable**: Qwen reports
CER, NCSpeech reports WER. CER is normally the more forgiving metric, so Qwen's
Filipino numbers are further behind 9.34% WER than the table makes them look.
NVIDIA's own models (Parakeet, Canary, Nemotron) are excluded because none of
them support Filipino in any tier.

## Measured results

Reproductions live in `benchmarks/results/`, dated. Run them with
`meeting-notes bench` — see `benchmarks/README.md`.

**Accuracy: not yet confirmed on our own hardware.** Until a dated report exists
in `benchmarks/results/`, every WER number above is someone else's claim.

**Packaging, measured 2026-07-30** (Windows 11, x86_64, Rust 1.97.1):

| Artifact | Size | Notes |
|---|---|---|
| Sidecar (`meeting-notes-server.exe`) | **222 MB** | PyInstaller one-file, torch excluded |
| App binary (`meeting-notes-desktop.exe`) | **4.5 MB** | Tauri release, LTO + stripped |
| MSI installer | **223.1 MB** | |
| NSIS installer | **222.9 MB** | |
| Cold start to first HTTP response | **5.4 s** | Model load is additional and lazy |

⚠️ Installers are **unsigned**. SmartScreen will warn and hospital IT will
likely block them; code signing is the outstanding packaging step.

**End-to-end run, measured 2026-07-30** — 35 s of synthesized English meeting
speech through the packaged sidecar at shipping defaults:

| Model | Wall clock (transcribe + notes) | Transcript quality |
|---|---|---|
| `large-v3-turbo` | 27 s | Verbatim |
| `tiny` | 4 s (transcription only) | Verbatim on this clip |

Notes came from Ollama (`llama3.2`), not the extractive fallback. Both models
transcribed clean synthetic English perfectly, which says nothing about noisy
rooms or Taglish — treat it as proof the pipeline works, not as an accuracy
measurement.

⚠️ **First-run model download is the real risk.** `large-v3-turbo` is ~1.55 GB
and downloads on first transcription. On a slow connection that is a long,
silent wait. Pre-warm before any demo:

```bash
meeting-notes transcribe some-short-clip.wav --language en
```

`tiny` (75 MB) is a reasonable fallback if bandwidth is tight — it handled
clean English verbatim here.

222 MB is almost entirely CTranslate2, onnxruntime and their numeric
dependencies. Keeping torch out is what holds it there — a bundle that pulls
torch in lands around 1.5 GB. `scripts/build_sidecar.py` excludes it explicitly;
if the size jumps, something re-imported the NeMo export path at runtime.

The 5.4 s cold start is why the desktop shell has a startup gate rather than an
empty window: silence during load reads as a hang.

## Setup

```bash
pip install -e ".[dev,server,tagalog]"   # runtime
pip install -e ".[bench]"                 # public eval datasets
python scripts/export_nemo_to_onnx.py     # one-time Tagalog ONNX export
meeting-notes doctor                      # verify both engines report ✓
```

The `nemo` extra (PyTorch + NeMo toolkit) is needed **only** for the export —
install it in a throwaway venv and discard it. Runtime needs just sherpa-onnx.

---

# Mobile

## How model selection works now

Both models are **selectable inside the app**: Settings → Transcription Model / AI Notes Model.
Selection is stored on-device; picking a model that isn't downloaded starts the download
automatically. The old `ACTIVE_*` constants in `mobile/lib/models.ts` were replaced by
`getActiveWhisperModel()` / `getActiveLlmModel()`, which read the user's choice.

**Defaults (first install):** Whisper base + TinyLlama — chosen to match previously
downloaded files so existing installs keep working. For real use, switch to:

| Role | Recommended | Size | Why |
|------|-------------|------|-----|
| **Transcription** | Whisper large-v3-turbo q5 | ~547 MB | Near large-v3 accuracy at several times the speed — best Taglish (re-verified July 2026: still no better multilingual option) |
| **AI Notes** | Qwen3.5 2B Q4_K_M | ~1.2 GB | 201 languages, much stronger structured/JSON output than Qwen3-era models; Qwen3.5 4B for max quality |

---

## Whisper — Models offered in Settings

Multilingual models support 99+ languages including Filipino/Tagalog (`tl`).
English-only (`.en`) variants exist in the catalogue but are not offered in the picker —
Taglish needs multilingual.

| Key | Model | Size | Speed | Quality |
|-----|-------|------|-------|---------|
| `TINY` | ggml-tiny | ~77 MB | Fastest | Basic (also used for live captions) |
| `BASE` (default) | ggml-base | ~142 MB | Fast | Good for clear English |
| `SMALL` | ggml-small | ~466 MB | Medium | Good Taglish + medical terms |
| `LARGE_V3_TURBO_Q5` (recommended) | ggml-large-v3-turbo-q5_0 | ~547 MB | Medium | Near large-v3 — best Taglish |
| `LARGE_V3_TURBO_Q8` | ggml-large-v3-turbo-q8_0 | ~833 MB | Medium | Turbo at higher fidelity |

> large-v3-turbo keeps the full large-v3 encoder but prunes the decoder from 32 to
> 4 layers — several times faster than large-v3 with nearly the same accuracy.
> It replaces `MEDIUM`/`LARGE_V3` for practical use (both remain in the catalogue).

> All models from: `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/<filename>`

### Silero VAD (hidden asset)

`ggml-silero-v6.2.0.bin` (~0.9 MB) is auto-downloaded on first transcription and never
shown in the picker. Transcription detects speech regions first and skips silence, so
Whisper time scales with **speech** duration, not recording duration (meetings are
typically 20–40% silence). If the VAD model can't be fetched (offline first run), the
app silently falls back to full-file transcription.

### Speaker diarization (hidden assets)

Real diarization via **sherpa-onnx** (`DiarizationModule.kt`, official AAR v1.13.4 in
`android/app/libs/`) replaces pitch-only clustering as the primary speaker detector.
Auto-downloaded on first use (~34 MB total):

| Model | File | Size | Role |
|-------|------|------|------|
| pyannote segmentation-3.0 (MIT) | `pyannote-segmentation-3-0.onnx` | ~5.7 MB | Finds speaker change points acoustically — works with zero pause between speakers |
| 3D-Speaker CAM++ (zh-en advanced) | `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | ~27 MB | Voice embeddings — tells similar voices apart, keeps labels consistent. Replaced the VoxCeleb-EN CAM++ 2026-08-28: benchmarked on a labelled 3-voice fixture with a background bed, the old model never found the right speaker count at any clustering threshold; this one is exact at threshold 0.6 |

Desktop (`diarize.py`) uses the same two models plus a GTCRN denoise pre-pass
(`gtcrn_simple.onnx`, ~0.5 MB) that strips background beds before embedding —
raises purity from 94% to 97.5% on the benchmark fixture. Not on mobile yet
(needs a native module addition + dev-client rebuild).

Both rehosted ungated by the sherpa-onnx project. Language-agnostic (fine for Taglish).
Fallback chain when unavailable: VAD-refined pause turns + pitch clustering → pause
alternation (the previous behavior).

---

## LLM — Models offered in Settings

All models are GGUF quantized, run by `llama.rn` (llama.cpp) with `n_ctx` 4096.
Transcripts longer than ~6k characters are summarized in chunks (map-reduce), so long
meetings work on every model — bigger models just do it better.

| Key | Model | Size | Speed on phone | Quality |
|-----|-------|------|----------------|---------|
| `TINYLLAMA` (default) | TinyLlama 1.1B Q4_K_M | ~600 MB | ~15 sec/pass | Basic |
| `QWEN3_5_2B` (recommended) | Qwen3.5 2B Q4_K_M | ~1.2 GB | ~25 sec/pass | Strong multilingual (201 languages), best structured output in class |
| `QWEN3_5_4B` | Qwen3.5 4B Q4_K_M | ~2.6 GB | ~60 sec/pass | Best notes quality — flagship phones |
| `GEMMA3_4B` | Gemma 3 4B Instruct Q4_K_M | ~2.4 GB | ~60 sec/pass | Strongest verified Filipino/Tagalog |

> **Qwen3.5 small series** (Mar 2026, Apache 2.0) replaced the Qwen3 picks in
> July 2026: Qwen3.5-4B decisively beats Gemma 4 E4B on structured-output
> benchmarks (TAU2 79.9 vs 42.2) — exactly the JSON-notes workload. Caveat:
> its Tagalog quality has no published FilBench score yet; A/B against
> `QWEN3_1_7B`/`GEMMA3_4B` on a real Taglish transcript before dropping them.
> Older entries (`QWEN3_1_7B`, `QWEN3_4B`, `QWEN2_5_1_5B`, `GEMMA2_2B`,
> `PHI3_MINI`) remain in the catalogue but are no longer offered in the picker.

> **JSON is now grammar-enforced**: notes generation passes a `json_schema` to
> llama.rn, which converts it to a GBNF grammar — the sampler cannot emit
> invalid JSON on any model. Prompt-level "respond only with JSON" remains as
> belt-and-braces.

---

## Adding a new model to the catalogue

1. Add an entry to `WHISPER_MODELS` or `LLM_MODELS` in `mobile/lib/models.ts`
2. Add its key to `WHISPER_CHOICES` / `LLM_CHOICES` to surface it in the Settings picker

## Notes

- Models are downloaded **once** and stored in the app's document directory.
- Switching models in Settings keeps both files; delete unused ones from the picker rows.
- Whisper runs on CPU (Android) and Neural Engine (iOS) via CoreML.
- LLM runs on CPU with NEON/SIMD acceleration; Metal GPU on iOS is optional.
- Filipino/Tagalog requires a **multilingual** Whisper model (not `.en` variants).
- Live captions reuse the same Whisper model via `whisper.rn`'s `RealtimeTranscriber`.
