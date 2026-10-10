# ASR & Model Research — Meeting Notes AI

Consolidated research log for speech recognition and note-generation options.
Written so it can be picked up cold months later, and so specific claims can be
re-checked online without redoing the whole investigation.

**Last updated:** 2026-09-07 — Cebuano/Bisaya pass and training playbook (§6a
update, §9, §9b, §9c). Earlier passes: 2026-07-28, 2026-07-30.
**Scope:** desktop app (Windows), Philippine languages, **corporate meetings
first** (medical parked — see `CLAUDE.md`). Older sections still say "clinic";
read that as "meeting".

---

## How to read this document

Every claim carries an evidence tag. This matters more than usual here, because
several vendor numbers in this space turned out to be misleading.

| Tag | Means |
|---|---|
| **[MEASURED]** | We ran it ourselves on this project's hardware. Trust it. |
| **[SOURCE]** | Extracted from the vendor's own repo/API, not their marketing. Trust the fact, question the framing. |
| **[PAPER]** | Published research figure. Check the test set before comparing. |
| **[VENDOR]** | Vendor's own claim, unverified. Treat as a hypothesis. |
| **[STALE-RISK]** | Fast-moving; re-check before relying on it. |

**Fit ratings** for this project:

| Rating | Means |
|---|---|
| 🟢 **In use** | Shipping in the app today |
| 🟡 **Queued** | Decided on, not yet integrated |
| 🔵 **Watch** | Promising, blocked on something specific |
| ⚪ **Parked** | Real but doesn't fit our constraints |
| 🔴 **Ruled out** | Investigated and rejected, with reason |

---

## 1. Current state of the project

🟢 **What ships today** (see `MODELS.md` for detail):

| Role | Choice | Why |
|---|---|---|
| English / auto ASR | **faster-whisper `large-v3-turbo`** (CTranslate2, int8) | Only large-class model viable CPU-only |
| Tagalog ASR | **NCSpeech Tagalog FastConformer** via sherpa-onnx | Best published Tagalog figure found |
| Taglish | *Not implemented* | Scaffolded in `backends/router.py::merge_taglish` |
| Notes | **Ollama**, model auto-detected | Falls back to extractive if Ollama is down |

**[MEASURED] 2026-07-30, this machine** — 35 s of synthesized English meeting
speech through the packaged app:

| Metric | Value |
|---|---|
| `large-v3-turbo` end-to-end (transcribe + notes) | 27 s |
| `tiny` transcription only | 4 s |
| Transcript accuracy (clean synthetic English) | Verbatim, both models |
| Sidecar bundle | 222 MB (torch excluded) |
| Sidecar cold start | 5.4 s |
| MSI / NSIS installer | 223.1 MB / 222.9 MB |

⚠️ Clean synthetic English proves the pipeline works. It says **nothing** about
noisy rooms, accents, or Taglish.

---

## 2. The central problem

Three distinct problems that get conflated. Any engine evaluation has to say
which one it addresses.

1. **Tagalog** — a low-resource language. Solvable with the right model.
2. **Taglish** — *intra-sentential* code-switching (`Nag-reklamo siya ng chest
   pain`). Structurally different: language-conditioned models make this
   worse, not better.
3. **Regional languages** — Cebuano, Ilocano, Hiligaynon, Waray, Bikol. Until
   Nov 2025 essentially unserved; **now covered by Meta** (§6).

Taglish morphology is the hard part: Tagalog affixes bolt onto English roots
while the root keeps English spelling — `nag-meeting`, `i-refer`,
`na-diagnose`, `magda-drive`. An English-trained model drops the prefix; a
Tagalog-trained one mis-segments the root. **[PAPER]**

---

## 3. NVIDIA NeMo family — 🔴 Ruled out

Investigated because it was the original proposal. **No NVIDIA ASR model
supports Filipino in any tier.** **[SOURCE]** — verified against model cards.

| Model | Languages | Filipino? |
|---|---|---|
| `nemotron-3.5-asr-streaming-0.6b` | 40 locales, 3 tiers | ❌ not in any tier |
| `parakeet-tdt-0.6b-v3` | 25 European | ❌ |
| `canary-1b-v2` | 25 European | ❌ |
| Granary corpus (1M hrs) | EU-focused | ❌ |

Also: `stt_en_fastconformer_hybrid_large_streaming_multi` (the specific model
originally recommended) is **superseded** by Nemotron 3.5 ASR Streaming.

**Why ruled out:** choosing NeMo means bootstrapping a language NVIDIA has
never touched — new tokenizer, new decoder, hundreds of hours of labelled
Taglish clinical audio that does not exist. The `change_vocabulary()` /
encoder-freeze workflow is real and documented, but it is domain adaptation,
not language creation.

**Still worth borrowing:** their **KenLM / N-gram LM fusion** approach for
biasing decoding toward drug names. That idea generalises to every engine.

---

## 4. Whisper family — 🟢 In use

| Model | Params | CPU viability | Notes |
|---|---|---|---|
| `large-v3` | 1.55B | ~0.4× realtime **[PAPER]** | Not shippable CPU-only |
| **`large-v3-turbo`** | 809M | ~7× faster than v3 **[PAPER]** | 32→4 decoder layers, ≈large-v2 accuracy |
| `medium` / `small` / `tiny` | — | Fast | `tiny` handled clean English verbatim **[MEASURED]** |

Tagalog: ~78–85% accuracy on clean formal audio, 70–80% casual, worse on
Taglish. **[VENDOR]** No reliable published Tagalog WER.

**Runtime:** `faster-whisper` (CTranslate2) — same weights, ~4× faster on CPU
at int8, ~4 GB vs ~10 GB memory. This is what makes turbo viable and keeps
torch out of the bundle.

⚠️ **[MEASURED]** `large-v3-turbo` is ~1.55 GB and downloads on first
transcription. Pre-warm before any demo.

---

## 5. NCSpeech Tagalog FastConformer — 🟡 Queued

`NCSpeech/stt_tl_fastconformer_hybrid_large` — 115M params, ~520 h Tagalog,
CC-BY-4.0.

| Benchmark | Score |
|---|---|
| FLEURS `fil_ph` | **9.34% WER** **[VENDOR]** |
| Magic Data Tech (conversational) | 16.10% WER **[VENDOR]** |

Best Tagalog figure found anywhere, by a wide margin. Someone already did the
NeMo tokenizer-swap + fine-tune that §3 describes.

**Caveats:** vendor-reported, ~156 downloads/month (community-unvalidated),
Tagalog-only with no documented English or code-switching handling.

**Status:** wired into `backends/nemo.py` and routed by `--language tl`, but
**not yet exported to ONNX**, so it currently falls back to Whisper with a
visible warning. Run `scripts/export_nemo_to_onnx.py` to activate.

---

## 6. Meta models — 🔵 Watch (the big finding)

> **Correction to earlier research.** An earlier round of this investigation
> concluded that no engine supported Cebuano, Ilocano, Hiligaynon, Waray or
> Bikol. That was wrong — it checked NVIDIA, OpenAI and Alibaba but not Meta.
> Both Meta models cover all of them.

### 6a. Omnilingual ASR (Nov 2025, Apache 2.0)

**[SOURCE]** — language list pulled from `lang_ids.py` via the GitHub API
(1,334 codes), not from marketing.

All confirmed present: `tgl_Latn`, `ceb_Latn`, `ilo_Latn`, `hil_Latn`,
`war_Latn`, `bcl_Latn`, `pam_Latn`, `pag_Latn`, `krj_Latn`, plus `ifb_Latn`
and `ifk_Latn` (Ifugao).

**[SOURCE]** Published CER, from their `per_language_results_table_7B_llm_asr.csv`:

| Language | Training hrs | CER |
|---|---|---|
| Hiligaynon | 116.2 | 0.2 |
| Ilocano | 189.2 | 0.3 |
| Bikol | 32.8 | 0.4 |
| Waray | 28.6 | 0.5 |
| Pangasinan | 123.2 | 0.5 |
| Kinaray-a | 22.4 | 0.5 |
| Kapampangan | 20.4 | 0.8 |
| Cebuano | 196.8 | 2.0 |
| **Tagalog** | — | **no result published** |
| *English (reference)* | *63,591* | *3.0* |

⚠️ **Do not read these as accuracy rankings.** Hiligaynon at 0.2 on 116 hours
vs English at 3.0 on 63,591 hours is a statement about *test sets*, not
languages: low-resource entries are scripted read speech, English is scored on
hard conversational benchmarks. **CER is also far more forgiving than WER.**
Median CER across all 1,683 scored languages is 2.4.

**Blockers for this project**, in order of severity:

1. **Language-conditioned, not code-switching.** You declare the language up
   front — structurally the opposite of what Taglish needs.
2. **Requires `torch` + `torchaudio`** — the exact dependency excluded to keep
   the installer at 222 MB. Would push it to ~1.5–2 GB.
3. **Python 3.10–3.12 only** (`requires_python: <=3.12,>=3.10` **[SOURCE]**).
   This machine runs 3.14.5; install fails today. **[MEASURED]**
4. **40 s audio limit** on standard models; `_Unlimited` variants chunk at 30 s
   with 5 s overlap. Not built for live captioning.
5. **Published speed is GPU-only** — CTC-300M 96× realtime / ~2 GiB VRAM,
   7B 16× / ~15 GiB. No CPU figures.

**Model families:** W2V (SSL), CTC (no language conditioning), LLM (with
conditioning), Zero-Shot. Sizes 300M / 1B / 3B / 7B.

> **Update 2026-09-07 — blockers 2, 3 and 5 no longer apply to the CTC family.**
> sherpa-onnx, already our diarization runtime, now ships Omnilingual **CTC**
> models as ONNX **[SOURCE]** (k2-fsa docs, "Omnilingual ASR" section):
>
> | sherpa-onnx package (`…-2025-11-12`) | Size | Notes |
> |---|---|---|
> | `omnilingual-asr-1600-languages-300M-ctc-int8` | **348 MB** | RTF **0.225** on 2 threads, greedy **[VENDOR]** |
> | `…-300M-ctc` (fp32) | 581 MB | |
> | `…-1B-ctc-int8`, `…-1B-ctc` | not stated on the docs page | |
>
> Python entry point: `sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(model=…, tokens=…)`
> (repo example `python-api-examples/offline-omnilingual-asr-ctc-decode-files.py`).
> Windows wheels exist for CPython 3.10–3.13 in `sherpa-onnx` 1.13.7 **[SOURCE]**;
> this machine has 1.13.4 installed **[MEASURED]**. No torch, no fairseq2, no
> Python ceiling, and it fits the 222 MB-class bundle story (+348 MB).
>
> CTC models carry **no language token at all** — they emit characters for
> whatever they hear — so blocker 1 does not apply either. Whether that makes
> them *good* at Bisaya–English switching is untested.
>
> ⚠️ The 7B per-language CER table above does **not** describe the 300M CTC —
> Meta publishes per-language results only for the 7B LLM. On the paper's own
> Whisper-overlap benchmarks **[PAPER]** (Table 5, average CER across languages)
> the 300M CTC beats Whisper `large-v3` on three of four sets (e.g. 11.8 vs
> 22.6) and loses on one European-heavy set (4.1 vs 2.0); the 7B LLM is roughly
> 2× better than the 300M CTC everywhere. Expect "usable", not "2.0 CER".
>
> The paper's language inventory tags **Cebuano as "H" (high-resource)** and
> Tagalog as "L" **[SOURCE]** — Cebuano has more Omnilingual training data
> (196.8 h) than Tagalog, the reverse of every other resource we have found.

### 6b. MMS — Massively Multilingual Speech (2023)

**[SOURCE]** — `facebook/mms-1b-all` adapter list queried via the HF API:
**1,198 adapters**, including all nine Philippine codes checked (`tgl`, `ceb`,
`ilo`, `hil`, `war`, `bcl`, `pam`, `pag`, `krj`).

Architecturally the friendlier of the two: **wav2vec2-CTC**, plain
`transformers`, no fairseq2, and **ONNX-exportable** — meaning it could run in
sherpa-onnx beside the existing engines with no torch at runtime.

Less accurate than Omnilingual, and trained largely on religious readings, so
weak on conversational speech. But it is the one that could ship in a 222 MB
bundle.

**Where Meta genuinely unlocks something:** *monolingual regional
consultations.* A Cebuano-only consult in Cebu, or Ilocano-only in Ilocos, went
from impossible to plausible. That is a real market expansion — but it is not
the Taglish problem.

---

## 7. Qwen3-ASR (Alibaba, Jan 2026, Apache 2.0) — ⚪ Parked

The only *other* open model with first-class Filipino, and it does
**in-file language switching** — closer to Taglish than anything else found.

| Model | FLEURS Filipino | CPU speed |
|---|---|---|
| Qwen3-ASR-1.7B | 24.29 **CER** **[PAPER]** | 0.49× realtime — below realtime **[PAPER]** |
| Qwen3-ASR-0.6B | 36.10 **CER** **[PAPER]** | ~2.5× realtime **[PAPER]** |

⚠️ These are **CER**, while NCSpeech's 9.34% is **WER**. CER is normally the
more forgiving metric, so Qwen's Filipino numbers are *further* behind than the
table suggests.

**Parked because:** 1.7B is too slow CPU-only; 0.6B's Filipino is weak. Worth
re-checking if a better small variant appears.

Related: **[PAPER]** *Polyglot-Lion* fine-tuned Qwen3-ASR for Singapore
code-switching — the closest published precedent to a Taglish fine-tune.

Also: **[antirez/qwen-asr](https://github.com/antirez/qwen-asr)** — pure C, MIT,
BLAS-only, ~8× realtime for 0.6B on M3 Max, streaming + prompt biasing. **No
Windows support listed.** If that lands, the Python sidecar could disappear.
🔵 **Watch.**

---

## 8. Cloud APIs — ⚪ Parked

All 🔵 **[VENDOR]**, none benchmarked on Taglish by anyone:

| Provider | Claim |
|---|---|
| Gladia Solaria-1 | 100+ languages incl. Tagalog, native code-switching |
| ElevenLabs Scribe | 3.1% WER on FLEURS Filipino |
| Speechmatics | 96% word accuracy, Tagalog |

**Parked because:** for patient audio, RA 10173 + NPC/DOH rules make cloud a
compliance project (data residency, processor agreements, NPC registration,
penalties to ₱5M). On-device is the cheaper route to compliance. Useful as an
*upper-bound baseline* in benchmarks, not as a shipping target.

For non-clinical client meetings the compliance argument is weaker — but the
app already works offline, so there's no reason to add the dependency.

---

## 9. Datasets

| Dataset | Size | Fit |
|---|---|---|
| **FLEURS `fil_ph`** | ~400 test utts, **1.05 GB** download | 🟡 Benchmark. Wired into `bench`. Clean read speech. |
| **FLEURS `ceb_ph`** | 3,261 / 225 / 541 utts; **2.78 GB / 202 MB / 510 MB** train/val/test **[SOURCE]** | 🟡 Not yet wired into `bench` — same loader, config `ceb_ph`. |
| **Common Voice** | cv-corpus-26.0 (June 2026), 170 locales; spontaneous-speech corpus 4.0, 78 locales | 🔴 **No Philippine language in either** — not `tl`, `fil`, `ceb`, `hil`, `ilo`, `war`, `bcl`, `pam` **[SOURCE]** (checked `common-voice/cv-dataset` metadata). The earlier "Common Voice Tagalog" row was wrong. |
| **Philippine Languages Database (PLD)** | **448 h / 10 languages / 980 speakers**, incl. **58.5 h Cebuano**; news, **medical**, education, tourism prompts | 🟡 **On the Hub** as `sapinsapin/pld`, ungated parquet. **CC-BY-NC, research-only** — details §9b. |
| HF `Filipino-Tagalog-Speech-Dataset` | ~75 h | 🔵 |
| **TweetTaglish** (LREC 2022) | Text only | 🔵 Useful for lexicon/LM, **not** acoustic training |
| Omnilingual ASR Corpus | 350 underserved languages, CC-BY | 🔴 No Philippine languages **[SOURCE]** |

⚠️ **[MEASURED]** `load_dataset("google/fleurs", "fil_ph")` **hangs** — the repo
has no loading script and ~1,000 files to resolve. Must address the parquet
export directly. Already fixed in `bench/datasets.py`.

**Nothing public is Taglish (or Bisaya–English) meeting conversation.** That gap
is the binding constraint on any fine-tune, whichever engine is chosen.

---

## 9b. Cebuano / Bisaya — what exists (re-researched 2026-09-07)

> **Correction to the 2026-07-30 pass.** It concluded "FLEURS `ceb_ph` is the
> only free Cebuano speech" and "PLD access unknown". Both are out of date: the
> full PLD (58.5 h Cebuano) is on the Hub, a 3.5 h *spontaneous* Cebuano set
> exists, and there are five Cebuano Whisper fine-tunes to benchmark against.
> The 2026-07 claims that HF has no Cebuano *speech* still holds for
> conversational, meeting-domain audio — nothing public is Bisaya–English
> meeting talk.

### The language, as it matters for ASR

- **Names.** Cebuano = Sugbuanon = Bisaya/Binisaya in everyday use. Speakers
  say "Bisaya"; corpora and model tags say `ceb`. ISO 639-3 `ceb`, Omnilingual
  `ceb_Latn`. Whisper has **no `ceb` token** — every Cebuano Whisper fine-tune
  found reuses the Tagalog (`tl`) slot **[SOURCE]**, as `finetune_asr.py` in
  `sapinsapin/halohalo` does explicitly ("the other Philippine languages are
  effectively repurposing `tagalog` as the language slot").
- **Dialects.** Cebu (the reference variety), Bohol (Boholano), Leyte /
  Southern Leyte, Negros Oriental, and the Mindanao varieties (Davao, Cagayan
  de Oro, Zamboanga peninsula). Wolff (1972) calls Cebuano "remarkably uniform
  in pronunciation" across regions **[PAPER]**; the documented phonetic
  outliers are central-Bohol only — *y* → [dʒ] (`iya` → `ija`), `ako` → `aho`,
  intervocalic *l* → *w* after *u/o* (`kulang` → `kuwang`). Coastal Bohol incl.
  Tagbilaran is indistinguishable from Cebu. **Davao Bisaya** differs mainly by
  heavy Tagalog/English mixing, not phonetics.
  **Implication:** one acoustic model covers all of it. Dialect coverage is a
  *data-collection* checklist (a few Bohol and Davao speakers), not separate
  models. Lexical variation is the notes-LLM's problem, not the ASR's.
- **Orthography.** The KWF-funded *Giya sa Ortograpiyang Sinugbuanong Binisaya*
  (2022; USC, CNU, CTU) is the closest thing to a standard **[SOURCE]**:
  phonemic spelling, three vowels /a i u/ (o/e appear in practice and in
  loanwords), glottal stop written as a hyphen, contractions with an
  apostrophe. Real-world text swaps *o/u* and *e/i* freely (`unsa`/`onsa`).
  **Implication:** adopt the Giya for transcription; for **scoring**, fold
  o→u and e→i on both sides, or select on CER as `sapinsapin` did ("Taglish
  orthography varies at the word level"). Our bench normaliser already keeps
  word-internal hyphens, which the glottal-stop hyphen needs.

### Ready-made Cebuano models

All **[SOURCE]** from model cards / HF API on 2026-09-07. Downloads are
lifetime.

| Model | Base | Reported score | Data | Licence | Verdict |
|---|---|---|---|---|---|
| **Omnilingual 300M CTC** (sherpa-onnx) | wav2vec2-CTC | none published at this size (§6a). **[MEASURED 2026-09-07]** on the 70 s Bisaya fixture: intelligible Bisaya, English jargon mangled, RTF ≈ 0.3 on 4 threads — see `MODELS.md` | 196.8 h Cebuano in pretraining | Apache-2.0 | 🟢 **Wired in** as the `ceb` route (`backends/omnilingual.py`). WER still unmeasured — run `bench --dataset fleurs-ceb` |
| **`sapinsapin/whisper-small-pld-ceb`** (2026-08-11) | whisper-small, `tl` token | **WER 11.14 / CER 4.51**, PLD held-out, lowercased | 10k PLD clips, 2000 steps | Apache-2.0 (weights) — but trained on NC data | 🟡 Bench. ⚠️ read prompts + speaker overlap between splits ⇒ optimistic |
| `eemberda/phcodeswitch-ceb-dvo` (2026-08-01) | whisper-small, `tl` token | **WER 20.86** on own test | "small community-contributed" **Davao Cebuano + English code-switching** | **CC-BY-NC 4.0** | 🔵 Best *evidence* that Bisaya–English CS fine-tunes work; cannot ship (NC) |
| `vonrayn/xls-r-cebuano-asr` (2026-09-02) | XLS-R wav2vec2 | WER 20.8 / CER 5.75 | "unknown dataset" | none stated | ⚪ Unusable without provenance |
| `arrow2026/whisper-turbo-cebuano-epoch1-ct2` (2026-07-12) | large-v3-turbo, **already CTranslate2** | none | none | none stated | 🔵 Drops straight into faster-whisper — worth one bench run, nothing more |
| `troxyz1268/whisper-small-bisaya` (2026-08-25) | whisper-small | WER 44.5 after 60 steps | private | Apache-2.0 | 🔴 Toy |
| `sil-ai/wav2vec2-bloom-speech-ceb` (2022) | wav2vec2 | gated | Bloom Speech (children's books) | other | ⚪ |
| `kylegregory/wav2vec2-large-xlsr-53-ceb`, `Jerich/whisper-cebuano`, `donnamae/whisper-finetuned-cebuano-accent` | — | none | none | none | 🔴 Empty or undocumented |

Academic, not downloadable **[PAPER]**: Pascual & Azcarraga, *Filipino and
Bisaya Speech Corpus and Baseline Acoustic Models for Healthcare Chatbot ASR*
(2023) — 16 female speakers (7 Bisaya), HMM/trigram baseline **7.16% WER**;
and a 2025 IEEE follow-up on Filipino/Cebuano children's speech. Useful as
precedent that a small clean Bisaya corpus gets to single-digit WER; not
useful as data.

### Cebuano speech data

| Source | Cebuano speech | Style | Access / licence | Fit |
|---|---|---|---|---|
| **PLD — `sapinsapin/pld`** | **58.5 h, 56,928 utts** (paper: 152 speakers, 86 F / 66 M, 63 h 32 m, avg 3.7 s) **[SOURCE]** | Read sentences (news, **medical**, education, tourism, literature), isolated words, digits; 32 h *spontaneous* across all 10 languages but **text is the elicitation question, not a transcript** (`text_is_prompt`) | Ungated parquet on the Hub. **CC-BY-NC 4.0**; UP Diliman DSP Lab, DOST-funded (ISIP, 2011–2014); research/academic "letter of pledge" to **dsp@eee.upd.edu.ph**; "contact UP-DSP for uses beyond research" | 🟡 **Best training base by far.** Fieldwork-recorded in the Visayas. Has `speaker_dialect`, `mother_dialect`, `father_dialect` columns |
| **FLEURS `ceb_ph`** | ~10 h; 3,261 / 225 / 541 utts | Read Wikipedia/news sentences | CC-BY 4.0; 2.78 GB / 202 MB / 510 MB | 🟡 The only **commercially clean** public set. Wire `ceb_ph` into `bench` |
| **`SilencioNetwork/cebuano-speech`** | **3.48 h, 90 clips, 15 speakers**, mean clip 139 s, word-level alignment, human-validated | **Spontaneous, long-form, own devices** — the closest thing to meeting audio | CC-BY-NC 4.0; single `test` split | 🟡 **Use as the conversational eval set.** ⚠️ 11 of 15 speakers self-report Manila/Tagalog origin — check `dialect` per row |
| `Speech-data/Cebuano-Speech-Dataset` | Card claims "108 h, 807 files" in **135 MB** — not physically possible; the Hub repo is a teaser (`n<1K`) | Read; Cebu, Mindanao, Bohol, Leyte speakers claimed | CC-BY-NC-ND on the sample; commercial vendor (speech-data.ai) | ⚪ Ask for hours, price, and a real sample before believing the card |
| Pangeanic "Cebuano datasets" | Not disclosed | Conversational Bisaya, **Bisaya–English code-switching**, call-centre | Commercial, contact form | ⚪ Only vendor advertising code-switched Bisaya; no sizes, no prices |
| IARPA Babel Cebuano (LDC2018S07) | ~30 h | Telephone conversation | LDC, paid | ⚪ Conversational but telephone-band |
| Bloom Speech (SIL) | unknown, small | Children's-book readings | Gated | ⚪ |
| Common Voice | **none** | — | — | 🔴 No `ceb` locale exists (§9) |
| Text only: `jfernandez/cebuano-filipino-sentences` (100k–1M), `saillab/alpaca_cebuano_*`, `eemberda/english-ceb-bible`, Cebuano Wikipedia | — | — | mixed | 🔵 LM / lexicon / TTS prompts. ⚠️ Cebuano Wikipedia is overwhelmingly bot-generated (Lsjbot) — poor LM text |

**What this means:** the 2026-07 "there is no Cebuano speech" problem is solved
for *read* speech (~70 h between PLD and FLEURS). It is **not** solved for
*conversational* speech: 3.5 h (Silencio) is an eval set, not a training set,
and it is NC-licensed. The gap that remains is exactly the one only we can
fill — consented Bisaya / Bisaya–English meeting audio.

---

## 9c. Training playbook — adding a local language or dialect

Written to answer "how do we train this, and how do we get training data". Every
engine we use can be fine-tuned; the tooling is not the constraint. Data and
licences are.

### 1. Decide which of three things you mean by "train"

| Tier | What it is | When | Cost |
|---|---|---|---|
| **A. Adopt** | Use a model that already covers the language (Omnilingual CTC, `whisper-small-pld-ceb`) | Always try first | An afternoon |
| **B. Adapt** | Fine-tune an existing multilingual model on 5–50 h of your language/domain | When A's error rate is measured and too high | Days + one GPU |
| **C. Create** | Train a language from scratch | Never for us — needs hundreds of hours | — |

Everything below is Tier B. Tier A is §9b.

### 2. How much data, and what to expect **[PAPER]/[SOURCE]**

| Evidence | Setup | Data | Result |
|---|---|---|---|
| Omnilingual paper §5.7.5 | Fine-tune the **300M/1B/3B CTC** from the OmniASR seed, lr 1e-5, 5K steps | **5–10 h** per language, 11 languages | Best CER **1.6–7.4** for 9 of 11 (two African languages 15–30). Converges well before 5K steps. ~1 h on 32 GPUs ≈ **32 GPU-hours** |
| Whisper-for-Pashto (arXiv 2604.06507, 2026) | whisper base / small / large-v3-turbo, full FT vs LoRA vs frozen encoder | 113 h Common Voice | From >100% to **21–25% WER**. **LoRA (r=64) was 33 pp worse than full FT**, frozen encoder 15 pp worse, transfer from Urdu 45 pp worse. whisper-**small** was the practical optimum |
| HF MMS-adapter blog | Train only the ~2.5 M adapter weights of `mms-1b-all` + new vocab head, lr 1e-3 | **4 h** Turkish | WER 22% after 400 steps, **< 30 min** on one GPU |
| `sapinsapin/whisper-small-pld-ceb` | whisper-small full FT, batch 8×2, lr 1e-5 | 10k PLD clips (~10 h read), 2000 steps | WER 11.1 / CER 4.5 on read Cebuano (optimistic split) |
| Scottish Gaelic practitioner's guide | LoRA + 8-bit on large-v3-turbo | 288 h | Works, but that is 30× our budget |

**Working rules from this:**

1. **5–10 h of clean, transcribed, in-domain audio is the target.** Below ~3 h,
   fine-tune an adapter (MMS) or accept Tier A; above ~20 h, full fine-tune of
   a small model beats every parameter-efficient trick.
2. **Full fine-tune of whisper-small beats LoRA on a bigger model** at our data
   scale. Do not start with LoRA on large-v3-turbo.
3. **Replay 10–20 % English (and Tagalog) in every batch** or the model forgets
   them — and Bisaya–English switching is the point. The Davao model above
   trained on mixed audio and reports 20.9 % WER on code-switched test data.
4. **Read speech trains a reader.** PLD/FLEURS give a strong *base*; only meeting
   audio fixes meeting audio. Plan the fine-tune as PLD + own data, not either.
5. **Speaker-disjoint eval or the number is fiction.** PLD's own split overlaps
   speakers; re-split on `speaker_id`.

### 3. Building the dataset (the part nobody else can do for us)

**Target:** 5–10 h train + 1–2 h speaker-disjoint eval, 20–40 speakers, both
sexes, spread across Cebu / Bohol / Davao (or whichever dialects you serve),
laptop mic *and* room mic, real meeting content.

1. **Consent and notice.** Written consent covering recording, transcription,
   model training, and (if you may ever ship it) commercial use. RA 10173 applies
   to voice as personal data even outside health — see `PRIVACY.md`. Store
   under the same encryption as user meetings.
2. **Record.** 16 kHz mono WAV (what `desktop/src/lib/recorder.ts` already
   produces — reuse the app). Sessions of 10–20 min; role-play from the
   corporate fixtures in `Recordings/` (standup, project sync, stakeholder
   review) translated into Bisaya, plus genuinely spontaneous discussion.
   Let people code-switch; do not coach "pure" Cebuano.
3. **Write a one-page transcription convention before anyone transcribes.**
   Giya orthography; English words in English spelling (`i-forward`,
   `nag-deploy`); glottal stop as hyphen; numbers as digits; fillers kept
   (`kuan`, `ah`); `[unintelligible]` tag → drop the segment from training;
   no speaker labels in the text (diarization is separate).
4. **Pseudo-label, then correct — don't transcribe from scratch.** Run the best
   available model over everything first (Omnilingual **7B LLM** on a rented
   GPU with `ceb_Latn` conditioning, or the 300M CTC locally; for a first
   Cebuano pass `whisper-small-pld-ceb` is fine), then have native speakers
   *correct* in a segment editor. The app's own `TranscriptEditor` is exactly
   this UI. Correction is several times faster than blank transcription and
   produces the same training pairs.
5. **Segment.** Silero VAD (already in the stack) to cut on silence; keep
   segments ≤ 30 s for Whisper (its window) and ≤ 40 s for Omnilingual. For
   long corrected transcripts, forced-align with `torchaudio`'s `MMS_FA` or
   `ctc-forced-aligner` to recover segment boundaries.
6. **QC.** Round-trip: transcribe each finished segment with a *different*
   model and flag CER > 30 % for a second look (this is what `halohalo`'s
   livestream pipeline does). Reject clipped or overlapped-speech segments.
7. **Store as the bench manifest.** JSONL, one row per segment — `audio`,
   `text`, `speaker`, `language`, `dialect`, `duration` — exactly the format
   `benchmarks/README.md` already defines, so the same file is the training
   set, the eval set, and the `--manifest` input to `meeting-notes bench`.

**Where else audio can come from** (all still need transcripts, but step 4
makes that cheap): Bisaya radio (DYHP, DYAB, Bombo Radyo Cebu/Davao), Cebuano
news programmes (*TV Patrol Central Visayas*, *Superbalita*), Sunday sermons,
municipal council livestreams, Cebuano podcasts and YouTube vlogs. **Check
rights before training on any of it** — broadcast audio is copyrighted; for a
shipped model, consented recordings are the only clean source.

### 4. Recipes

**A. Whisper → faster-whisper (fits the shipping app today).** Linux/WSL with an
NVIDIA GPU, or a rented one; `whisper-small` fits an 8 GB card at fp16.

```bash
# Reproduce sapinsapin's Cebuano model as the starting point
git clone https://github.com/sapinsapin/halohalo && cd halohalo && pip install -r requirements.txt
python finetune_asr.py --dataset pld --language ceb --model openai/whisper-small \
    --max-samples 25000 --max-steps 5000
# then swap --dataset for a loader over your own manifest (the script's adapter
# normalises every corpus to (audio@16k, text, speaker_id); add one for JSONL)

# Convert for the app — ct2-transformers-converter ships with faster-whisper
ct2-transformers-converter --model finetune_runs/asr_pld_ceb \
    --output_dir models/whisper-small-ceb-ct2 --quantization int8 \
    --copy_files tokenizer.json preprocessor_config.json
```

Then add the folder name to `WHISPER_MODELS` in `config.py` (Click choices
follow it) and route `--language ceb` to it with `--language tl` passed to the
decoder — the fine-tune lives in the Tagalog slot. `backends/whisper.py` already
takes any string faster-whisper accepts, including a directory.

**B. Omnilingual CTC → sherpa-onnx (the language-agnostic path).** Fine-tune
Meta's 300M CTC on your data with their fairseq2 recipe, then export to ONNX
the way sherpa-onnx did for the published models. Linux, Python 3.10–3.12, one
GPU (the recipe's "32 GPUs" is their batch; gradient accumulation covers it).

```bash
pip install omnilingual-asr           # Python 3.10–3.12 only
# manifest rows: {"audio": ..., "text": ..., "lang": "ceb_Latn"}; 16 kHz
python -m workflows.recipes.wav2vec2.asr $OUT_DIR \
    --config-file workflows/recipes/wav2vec2/asr/configs/ctc-finetune.yaml
# export: k2-fsa/sherpa-onnx documents the ONNX export it used for the
# 2025-11-12 omnilingual models; the result runs with
# sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(...)
```

Why bother when A exists: no language token means one model for Bisaya, Tagalog
and English *in the same utterance*, and sherpa-onnx is already in the sidecar.
This is the path if the untrained 300M CTC benchmarks close to Whisper.

**C. MMS adapter (cheapest experiment, ~30 min).** Train ~2.5 M adapter weights
on top of `facebook/mms-1b-all` for a new vocabulary. Good for a 2–4 h pilot
set to learn whether the *data* is any good before spending on A or B. Runtime
is torch and ~3.9 GB fp32, so this is a research tool, not a shipping engine.

```python
from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all",
    vocab_size=len(vocab), ignore_mismatched_sizes=True)
model.init_adapter_layers()
model.freeze_base_model()
for p in model._get_adapters().values():
    p.requires_grad = True
# Trainer: lr 1e-3, batch 32, ~4 epochs, eval every 100 steps (HF blog recipe)
```

**D. NeMo FastConformer (streaming path).** `change_vocabulary()` on the
NCSpeech Tagalog model, fine-tune decoder with frozen encoder, re-export to
ONNX. Only worth it once the Tagalog export in §5 is actually done.

### 5. Compute and cost — **[ESTIMATE]**, not measured

| Recipe | Hardware | Time | Rented cost (≈ $0.5–1.5 / GPU-h) |
|---|---|---|---|
| A. whisper-small, 5K steps | one 8–24 GB NVIDIA GPU | a few hours | **< $20** |
| B. Omnilingual 300M CTC, 5K steps | one 24 GB GPU | ~1–2 days (paper: 32 GPU-h) | **$30–60** |
| C. MMS adapter | any GPU | < 1 h | **< $5** |
| Pseudo-labelling 10 h with 7B LLM | 24 GB GPU (~15 GiB VRAM) | ~1 h at 16× realtime | **< $5** |

None of this runs on this Windows box: no NVIDIA GPU, and Python 3.14 blocks
`omnilingual-asr`. Use WSL + a rented GPU or Colab/Kaggle for A and C.

### 6. Licence gates before anything ships

| Asset | Licence | Bench / R&D | Ship a model trained on it |
|---|---|---|---|
| PLD (`sapinsapin/pld`) | CC-BY-NC 4.0; UP-DSP research pledge | ✅ | ❌ **until UP-DSP grants terms in writing** (dsp@eee.upd.edu.ph) |
| `sapinsapin/whisper-small-pld-ceb` | Apache-2.0 on weights, but PLD-derived | ✅ | ⚠️ Same question as PLD — treat as research |
| `eemberda/phcodeswitch-ceb-dvo` | CC-BY-NC 4.0 | ✅ | ❌ |
| `SilencioNetwork/cebuano-speech` | CC-BY-NC 4.0 | ✅ eval | ❌ don't train on it |
| FLEURS `ceb_ph` | CC-BY 4.0 | ✅ | ✅ |
| Omnilingual weights | Apache-2.0 | ✅ | ✅ |
| `facebook/mms-1b-all` | CC-BY-NC 4.0 **[SOURCE]** (HF API tag, 2026-09-07) | ✅ | ❌ |
| Own consented recordings | your consent form | ✅ | ✅ if the form says so |

The practical consequence: **the only Cebuano model we can ship without a
negotiation is one trained on FLEURS + our own recordings** — or an untrained
Omnilingual CTC. That is a strong argument for recording our own 5–10 h early.

### 7. Recommended order for Cebuano

1. **Measure Tier A this week, zero training.** Add `ceb_ph` to
   `bench/datasets.py` (same loader as `fil_ph`; test parquet is 510 MB) and a
   ~50-line sherpa-onnx Omnilingual backend beside `backends/nemo.py`. Run:
   Omnilingual 300M CTC int8 · `large-v3-turbo --language tl` (baseline) ·
   `arrow2026/…-ct2` (drop-in) · `whisper-small-pld-ceb` after a CT2 convert.
2. **Add the conversational eval.** Load `SilencioNetwork/cebuano-speech` as a
   second dataset. FLEURS tells you about read Cebuano; Silencio tells you about
   people talking. Expect the ranking to change.
3. **Start recording** the 5–10 h meeting set (§3) in parallel — it gates
   everything else and has the longest lead time.
4. **Fine-tune only after 1–2 say you must**, with recipe A on PLD-Cebuano +
   own data, or recipe B if the untrained CTC was already close.
5. **Email UP-DSP** for commercial terms the moment a PLD-trained model wins a
   benchmark. Lead time is unknown; don't let it gate the release.

### Sources for §9b–§9c (checked 2026-09-07)

- k2-fsa sherpa-onnx docs — Omnilingual ASR models: <https://k2-fsa.github.io/sherpa/onnx/omnilingual-asr/models.html>
- sherpa-onnx Python example: <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-omnilingual-asr-ctc-decode-files.py>
- Omnilingual ASR paper (Table 5, §5.7.5, language inventory): <https://arxiv.org/abs/2511.09690>; fine-tune recipe: <https://github.com/facebookresearch/omnilingual-asr/blob/main/workflows/recipes/wav2vec2/asr/README.md>
- PLD paper (§3.4 Availability and Licensing): <https://aclanthology.org/2024.sigul-1.32.pdf>; Hub repack: <https://huggingface.co/datasets/sapinsapin/pld>; pipeline: <https://github.com/sapinsapin/halohalo>
- Cebuano models: <https://huggingface.co/sapinsapin/whisper-small-pld-ceb>, <https://huggingface.co/eemberda/phcodeswitch-ceb-dvo>, <https://huggingface.co/vonrayn/xls-r-cebuano-asr>, <https://huggingface.co/arrow2026/whisper-turbo-cebuano-epoch1-ct2>, <https://huggingface.co/troxyz1268/whisper-small-bisaya>
- Cebuano data: <https://huggingface.co/datasets/SilencioNetwork/cebuano-speech>, <https://huggingface.co/datasets/Speech-data/Cebuano-Speech-Dataset>, <https://pangeanic.com/cebuano-datasets-for-ai-training>, FLEURS `parquet-data/ceb_ph/` on <https://huggingface.co/datasets/google/fleurs>
- Common Voice release metadata: <https://github.com/common-voice/cv-dataset/tree/main/datasets>
- Fine-tuning evidence: Pashto <https://arxiv.org/abs/2604.06507>; MMS adapters <https://huggingface.co/blog/mms_adapters>; Whisper new-language discussion <https://github.com/openai/whisper/discussions/13>; Scottish Gaelic guide <https://arxiv.org/abs/2506.04915>
- Cebuano dialectology / orthography: <https://en.wikipedia.org/wiki/Boholano_dialect>, <https://en.wikipedia.org/wiki/Cebuano_language>, LDC Babel Cebuano LSP <https://catalog.ldc.upenn.edu/docs/LDC2018S07/LSP_301_final.pdf>, KWF *Giya sa Ortograpiyang Sinugbuanong Binisaya* (2022)
- Academic Bisaya ASR: Pascual & Azcarraga 2023 <https://www.semanticscholar.org/paper/18a01068e933bc8773d11eabdb9b5813f04e8819>; IEEE 2025 children's Filipino/Cebuano ASR <https://ieeexplore.ieee.org/document/11100316/>

---

## 10. Runtimes & deployment

| Runtime | Fit | Notes |
|---|---|---|
| **CTranslate2 / faster-whisper** | 🟢 In use | ~4× faster than PyTorch Whisper on CPU |
| **sherpa-onnx** | 🟢 In use | Runs Whisper, Parakeet, Canary, Moonshine, Zipformer, **Omnilingual CTC (since 2025-11-12)** + VAD + diarization. Windows, Python/C#/Rust/Node. The engine-agnostic layer. |
| PyInstaller | 🟢 In use | 222 MB bundle. ⚠️ Runs the real program as a *child of a bootloader* — see §12. |
| fairseq2 (Omnilingual) | 🔵 Watch | Windows supported per docs, but Python ≤3.12 |
| ONNX Runtime | 🟢 Indirect | CPU EP reportedly 4× faster than WebGPU for Qwen3-ASR-0.6B |

**[PAPER]** On-device streaming benchmark (arXiv 2604.14493), CPU-only,
English: Nemotron-0.6B int4 = 7.28% WER, 2.46× realtime, <1 GB.
Qwen3-ASR-1.7B = 0.49× realtime. **The RTF column transfers to our case even
though the WER column does not.**

---

## 11. Regulatory — Philippines

- **RA 10173 (Data Privacy Act 2012)** + NPC issuances + DOH health-information
  guidance. Health data is **sensitive personal information**.
- On-device processing means **no third-party processor**, **no cross-border
  transfer**, no data-residency question.
- Registration thresholds still attach to the **clinic**, not the software.
- Consent and notice remain the clinician's obligation.

Full data-flow write-up: `PRIVACY.md`.

---

## 12. Hard-won practical findings

Things that cost time and would cost it again. **[MEASURED]** unless noted.

| Finding | Detail |
|---|---|
| **PyInstaller orphans** | One-file bundles run the real program as a child of a bootloader. Killing "the sidecar" leaves the child alive holding a port and a model. Fixed via a parent-PID watchdog. |
| **Windows text mode corrupts keys** | `os.open` defaults to text mode; any `0x0A` in a random salt became `0x0D 0x0A`, ~6% of the time. Needs `O_BINARY`. |
| **Ollama name matching is exact** | Configured `llama3` vs installed `llama3.2:latest` → 404 → **silent** fallback to extractive notes. Fixed with auto-detection. |
| **CORS hides custom headers** | Cross-origin `fetch` can't read `X-Suggested-Filename` without `expose_headers`. |
| **`winget install Rustlang.Rustup` fails** | Its silent flags break `rustup-init`. Run `rustup-init.exe -y` then `rustup default stable`. |
| **FLEURS is 1.05 GB** | `--limit` does not shrink it; audio is embedded in one parquet per split. |
| **Nothing may run during `tauri build`** | An open app locks the exe; bundling fails with "Access is denied". |

---

## 13. Decision summary

| Option | Fit | One-line verdict |
|---|---|---|
| faster-whisper `large-v3-turbo` | 🟢 **In use** | Right choice for English and the default path |
| NCSpeech Tagalog FastConformer | 🟡 **Queued** | Best Tagalog number; needs the one-time ONNX export |
| Taglish two-pass merge | 🟡 **Queued** | Scaffolded; blocked on a Taglish test set |
| **Omnilingual 300M CTC via sherpa-onnx** | 🟢 **In use** (`ceb`) | 348 MB, CPU, no torch, no language token; wired 2026-09-07. Bisaya intelligible on the fixture; WER not yet benchmarked (§9b) |
| Meta Omnilingual 7B LLM | 🔵 **Watch** | Best accuracy; GPU-only — use as the *pseudo-labeller* on a rented GPU, not as a runtime |
| Meta MMS | ⚪ **Parked** | Superseded by Omnilingual-in-sherpa-onnx; adapter fine-tune still the cheapest training recipe (§9c) |
| `sapinsapin/whisper-small-pld-ceb` | 🟡 **Queued** | Only Cebuano Whisper with a reported score (WER 11.1 on PLD held-out); CT2-convert and bench |
| Philippine Languages Database (`sapinsapin/pld`) | 🟡 **Queued** | 58.5 h Cebuano, downloadable now; **CC-BY-NC** — research/bench only until UP-DSP grants terms |
| Cebuano meeting-domain fine-tune | 🔵 **Watch** | Recipe ready (§9c); blocked on 5–10 h of consented Bisaya meeting audio |
| `antirez/qwen-asr` (C runtime) | 🔵 **Watch** | Would remove the Python sidecar entirely — if Windows lands |
| Qwen3-ASR | ⚪ **Parked** | Too slow (1.7B) or too weak (0.6B) |
| Cloud APIs | ⚪ **Parked** | Compliance cost; keep as benchmark baseline only |
| Common Voice | 🔴 **Ruled out** | No Philippine locale exists, scripted or spontaneous (June 2026) |
| NVIDIA NeMo family | 🔴 **Ruled out** | No Filipino in any model, in any tier |

---

## 14. Open questions

Unanswered, in priority order:

1. **Does NCSpeech's 9.34% hold?** Nothing in `benchmarks/results/` yet. One
   FLEURS run answers it.
2. **What is real Taglish clinical WER?** No public data exists. Needs ~2 h of
   consented recordings, three tiers (see `benchmarks/README.md`).
3. **What does the 300M CTC actually score on Cebuano?** Vendor RTF is 0.225
   on 2 threads; accuracy on FLEURS `ceb_ph` and on the Silencio spontaneous set
   (§9b) is unmeasured. This is the first bench run to do.
4. **Does a language-token-free CTC model cope with Bisaya–English switching?**
   Structurally it should; nobody has measured it.
5. **Does Tagalog work in Omnilingual at all?** In `supported_langs`, tagged
   low-resource ("L") in the paper, still no published result.
6. **Will UP-DSP license PLD-derived models for commercial use?** Paper says
   research pledge + CC-BY-NC; nothing published about commercial terms.

*Answered since 2026-07-30:* "Can Omnilingual run on CPU?" — yes, via
sherpa-onnx (§6a). "Is PLD downloadable?" — yes, `sapinsapin/pld` (§9b). "Is MMS
good enough?" — moot; Omnilingual CTC replaces it as the regional option.

---

## 15. Re-checking this research

Things move fast here. To refresh, ask for an update on:

**Engines**
- "Has NVIDIA added Filipino/Tagalog to any NeMo ASR model?" — was a hard no
- "New Omnilingual ASR releases — smaller models, ONNX export, or CPU
  benchmarks?"
- "Is there a Qwen3-ASR successor with better Filipino, or Windows support in
  `antirez/qwen-asr`?"
- "Any new Tagalog or Taglish fine-tune on HuggingFace beating 9.34% FLEURS?"
- "Has anyone published a Taglish or Philippine code-switching ASR benchmark?"

**Runtimes**
- "Has sherpa-onnx added Omnilingual 1B/3B CTC or any LLM-ASR variant?" — 300M
  and 1B CTC as of 2025-11-12 **[SOURCE]**
- "Did sherpa-onnx publish CPU accuracy/RTF for the 1B int8?"

**Data**
- "Has UP-DSP published commercial terms for PLD, or has `sapinsapin/pld`'s
  licence changed?"
- "Did Common Voice add any Philippine locale?" — none as of cv-corpus-26.0
- "Any new Cebuano/Bisaya conversational speech corpus, or a bigger Silencio
  pack?"
- "New Cebuano ASR models?" — query
  `https://huggingface.co/api/models?filter=ceb&pipeline_tag=automatic-speech-recognition`

**Direct sources worth re-reading**
- `gh api repos/facebookresearch/omnilingual-asr/contents/per_language_results_table_7B_llm_asr.csv`
- `https://k2-fsa.github.io/sherpa/onnx/omnilingual-asr/models.html`
- `https://huggingface.co/datasets/sapinsapin/pld` (README → "License, source and citation")
- `https://raw.githubusercontent.com/common-voice/cv-dataset/main/datasets/scripted-speech/` (latest `cv-corpus-*.json`, check `locales`)
- `https://huggingface.co/api/models/facebook/mms-1b-all` (adapter list)
- HuggingFace Open ASR Leaderboard
- `pypi.org/pypi/omnilingual-asr/json` (version constraints)

⚠️ When re-checking, **re-verify from source, not from blog posts.** Every
significant correction in this document came from reading a repo file or an API
response that contradicted the marketing.

---

## Related documents

| File | Contents |
|---|---|
| `MODELS.md` | What ships, why, and measured packaging numbers |
| `PRIVACY.md` | Data flow, encryption, RA 10173 position |
| `benchmarks/README.md` | How to run evaluations; Taglish test-set spec |
| `desktop/README.md` | Build, run, and demo checklist |
