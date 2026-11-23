"""Run Cebuano/Bisaya candidates over your own recordings and put the results side by side.

    python scripts/compare_ceb.py Recordings/Bisaya.mp4 Recordings/Taglish.mp4
    python scripts/compare_ceb.py Recordings/Bisaya.mp4 --engine omnilingual-ctc --engine wav2vec2-ctc \
        --engine wav2vec2-ctc:some-org/other-model --out benchmarks/results/ceb-compare.json

Engines are bench specs (``omnilingual-ctc``, ``wav2vec2-ctc``, ``wav2vec2-ctc:<hf repo>``,
``faster-whisper:large-v3-turbo``). A local model folder (a gated repo you downloaded) goes through the
``MEETING_NOTES_WAV2VEC2_MODEL`` environment variable instead, because a Windows path has a colon in it.

WHAT YOU GET WITHOUT A REFERENCE. Each engine's transcript, how long it took against the audio length, and
how much the engines DISAGREE with each other (word error rate of one against another). Disagreement is not
accuracy, but a big one tells you where to look, and reading the text next to each other is the quickest
honest comparison of two models on audio they were not trained for.

WITH A REFERENCE. Put the correct transcript in ``<audio name>.ref.txt`` beside the file (plain text, one
speaker turn per line is fine; timestamps and speaker names are not part of the score, so leave them out).
Then each engine also gets a WER and CER on that file, scored with the project's own normaliser. A reference
only needs to cover what you want judged: a 30-second excerpt of the audio and its text is enough.

Nothing here downloads a model on its own except what the engines themselves fetch on first use
(Omnilingual ~290 MB; the wav2vec2 repo ~1.2 GB, and it needs ``pip install torch transformers``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from meeting_notes.backends.base import BackendUnavailableError  # noqa: E402
from meeting_notes.backends.router import resolve_backend  # noqa: E402
from meeting_notes.bench.metrics import character_error_rate, word_error_rate  # noqa: E402
from meeting_notes.config import DEFAULT_WHISPER_MODEL, AppConfig, LanguageMode  # noqa: E402

DEFAULT_ENGINES = ["omnilingual-ctc", "wav2vec2-ctc"]


def build(spec: str, language: str):
    name, _, model = spec.partition(":")
    config = AppConfig(
        whisper_model=model or DEFAULT_WHISPER_MODEL,
        language_mode=LanguageMode.from_value(language),
        backend=name,
    )
    config.validate()
    return resolve_backend(config)


def audio_seconds(path: str) -> float:
    from meeting_notes.audio import load_audio_file

    audio, rate = load_audio_file(path)
    return len(audio) / rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="+", help="recordings to run")
    parser.add_argument("--engine", action="append", dest="engines", help="a bench engine spec; repeat for several")
    parser.add_argument("--language", default="ceb", help="language mode the engines are built for (default ceb)")
    parser.add_argument("--out", help="also write everything as JSON here")
    args = parser.parse_args()
    engines = args.engines or DEFAULT_ENGINES

    report: dict = {"engines": engines, "files": []}
    backends: dict = {}
    for spec in engines:
        try:
            backends[spec] = build(spec, args.language)
        except (ValueError, NotImplementedError, BackendUnavailableError) as exc:
            print(f"skipping {spec}: {exc}")

    if not backends:
        print("nothing to run")
        return 1

    for path in args.audio:
        seconds = audio_seconds(path)
        ref_file = Path(path).with_suffix(".ref.txt")
        reference = ref_file.read_text(encoding="utf-8") if ref_file.exists() else None
        print(f"\n{'=' * 78}\n{path}  ({seconds:.0f} s)" + ("  [reference found]" if reference else "  [no reference]"))
        row: dict = {"file": path, "seconds": seconds, "engines": {}}
        texts: dict[str, str] = {}

        for spec, backend in backends.items():
            try:
                backend.warmup()                       # model load is not part of the timing
                started = time.perf_counter()
                result = backend.transcribe_file(path)
                took = time.perf_counter() - started
            except BackendUnavailableError as exc:
                print(f"\n--- {spec}: unavailable: {exc}")
                continue
            texts[spec] = result.text
            entry = {"text": result.text, "seconds": round(took, 1), "rtf": round(took / seconds, 2)}
            if reference:
                w = word_error_rate(reference, result.text)
                c = character_error_rate(reference, result.text)
                entry["wer"], entry["cer"] = round(w.rate, 4), round(c.rate, 4)
            row["engines"][spec] = entry
            score = f"  WER {entry['wer'] * 100:.1f}%  CER {entry['cer'] * 100:.1f}%" if reference else ""
            print(f"\n--- {spec}   {took:.0f} s (RTF {took / seconds:.2f}){score}\n{result.text}")

        names = list(texts)
        if len(names) >= 2:
            print("\n--- disagreement between engines (word error rate of the second against the first; not accuracy)")
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    print(f"    {b}  vs  {a}: {word_error_rate(texts[a], texts[b]).rate * 100:.0f}%")
        if reference:
            print(f"\n--- reference ({ref_file.name})\n{reference.strip()}")
        report["files"].append(row)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
