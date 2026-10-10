"""One-time export of the Tagalog FastConformer to ONNX for sherpa-onnx.

Downloads ``NCSpeech/stt_tl_fastconformer_hybrid_large`` (115M params, ~520h of
Tagalog, CC-BY-4.0) and writes the ONNX files the Tagalog backend loads.

Run this once per machine::

    pip install -e ".[nemo,tagalog]"
    python scripts/export_nemo_to_onnx.py

The ``nemo`` extra pulls in PyTorch and the NeMo toolkit, which are needed only
for this export — uninstall them afterwards if you care about disk. Runtime
needs just ``sherpa-onnx``.

The checkpoint is *hybrid*: it carries both a CTC and an RNN-T (transducer)
head.

* **CTC** (default) — one ``model.onnx``, a single documented export call,
  slightly higher WER, no hotword biasing.
* **Transducer** (``--transducer``) — three ONNX files, marginally better
  accuracy, and the only head sherpa-onnx can apply hotword biasing to. The
  export is more sensitive to your installed NeMo version; if it fails, use
  CTC and open an issue with the traceback.

Verify the result before trusting it::

    meeting-notes doctor
    meeting-notes transcribe sample_tagalog.wav --language tl
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

# Import from the installed package so the export lands where the app looks.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from meeting_notes.backends.nemo import MODEL_SUBDIR, VAD_FILENAME  # noqa: E402
from meeting_notes.config import models_dir  # noqa: E402

HF_MODEL_ID = "NCSpeech/stt_tl_fastconformer_hybrid_large"

SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "silero_vad.onnx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transducer",
        action="store_true",
        help="Export the RNN-T head instead of CTC (enables hotword biasing).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the ONNX files. Defaults to the app models dir.",
    )
    parser.add_argument(
        "--skip-vad",
        action="store_true",
        help="Do not download the Silero VAD model.",
    )
    return parser.parse_args()


def load_model():
    """Download and load the NeMo checkpoint."""
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        sys.exit(
            "NeMo is not installed. Install the export dependencies:\n"
            "    pip install -e \".[nemo]\""
        )

    print(f"Downloading {HF_MODEL_ID} …")
    return nemo_asr.models.ASRModel.from_pretrained(model_name=HF_MODEL_ID)


def write_tokens(model, out_dir: Path, blank_last: bool) -> None:
    """Write ``tokens.txt`` in the layout sherpa-onnx expects.

    NeMo CTC models put the blank symbol *last*; transducers put it first.
    Getting this wrong produces confident nonsense rather than an error, so it
    is worth eyeballing the first and last lines of the generated file.
    """
    vocab = list(model.decoder.vocabulary)
    tokens_path = out_dir / "tokens.txt"

    with tokens_path.open("w", encoding="utf-8") as f:
        if blank_last:
            for i, token in enumerate(vocab):
                f.write(f"{token} {i}\n")
            f.write(f"<blk> {len(vocab)}\n")
        else:
            f.write("<blk> 0\n")
            for i, token in enumerate(vocab):
                f.write(f"{token} {i + 1}\n")

    print(f"  wrote {tokens_path} ({len(vocab) + 1} tokens)")


def export_ctc(model, out_dir: Path) -> None:
    """Export the CTC head as a single ``model.onnx``."""
    print("Exporting the CTC head …")
    model.set_export_config({"decoder_type": "ctc"})
    model.export(str(out_dir / "model.onnx"))
    write_tokens(model, out_dir, blank_last=True)


def export_transducer(model, out_dir: Path) -> None:
    """Export the RNN-T head as encoder/decoder/joiner ONNX files."""
    print("Exporting the transducer head …")
    model.set_export_config({"decoder_type": "rnnt"})

    # NeMo emits `<stem>_encoder.onnx` and `<stem>_decoder_joint.onnx`; sherpa
    # -onnx wants encoder/decoder/joiner. Newer NeMo splits the joint network
    # when `export_joint_separately` is set.
    model.export(str(out_dir / "model.onnx"))

    renames = {
        "model_encoder.onnx": "encoder.onnx",
        "model_decoder.onnx": "decoder.onnx",
        "model_joiner.onnx": "joiner.onnx",
        "model_decoder_joint.onnx": "decoder_joint.onnx",
    }
    for src_name, dst_name in renames.items():
        src = out_dir / src_name
        if src.exists():
            shutil.move(str(src), str(out_dir / dst_name))
            print(f"  {src_name} → {dst_name}")

    write_tokens(model, out_dir, blank_last=False)

    if not (out_dir / "joiner.onnx").exists():
        print(
            "\n  WARNING: no joiner.onnx was produced. Your NeMo version "
            "exported a fused decoder+joint network, which sherpa-onnx cannot "
            "load. Re-run without --transducer to use the CTC head.",
            file=sys.stderr,
        )


def download_vad(out_dir: Path) -> None:
    """Fetch Silero VAD so the backend can segment by utterance."""
    dest = out_dir / VAD_FILENAME
    if dest.exists():
        print(f"  {VAD_FILENAME} already present")
        return
    print(f"Downloading {VAD_FILENAME} …")
    try:
        urllib.request.urlretrieve(SILERO_VAD_URL, dest)
        print(f"  wrote {dest}")
    except Exception as exc:  # noqa: BLE001
        print(
            f"  WARNING: VAD download failed ({exc}). The backend still works "
            "but decodes each file as one segment, losing timestamps.",
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else models_dir() / MODEL_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    model = load_model()

    if args.transducer:
        export_transducer(model, out_dir)
    else:
        export_ctc(model, out_dir)

    if not args.skip_vad:
        download_vad(out_dir)

    print("\nDone. Verify with:")
    print("    meeting-notes doctor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
