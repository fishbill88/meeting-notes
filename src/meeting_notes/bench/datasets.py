"""Evaluation datasets.

Three sources, in increasing order of how much they tell you:

* **FLEURS** (``fil_ph``, ``en_us``) — ~400 test utterances per language of
  clean read speech. Reproduces published figures, which is its whole value.
  It will flatter every engine relative to a real consultation.
* **Common Voice** — read but more spontaneous and more varied in recording
  quality. A better proxy, still not conversational.
* **Local JSONL manifest** — your own audio. This is the format the Taglish
  clinical test set will use, so the loader exists before the data does.

Manifest format, one JSON object per line::

    {"audio": "clips/001.wav", "text": "Nag-reklamo siya ng chest pain."}
    {"audio": "clips/002.wav", "text": "Ubo at lagnat po.", "id": "002"}

Relative ``audio`` paths resolve against the manifest's own directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from meeting_notes.config import DEFAULT_SAMPLE_RATE

_DATASETS_HINT = (
    "The Hugging Face `datasets` package is required for FLEURS and Common "
    "Voice.\nInstall it with:\n"
    "    pip install -e \".[bench]\"\n"
    "Or benchmark your own audio with:\n"
    "    meeting-notes bench --manifest path/to/manifest.jsonl"
)


@dataclass
class Utterance:
    """One scored item: audio plus its reference transcript."""

    id: str
    reference: str
    audio_path: Optional[str] = None
    audio_array: Optional[Any] = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    duration: Optional[float] = None

    def resolved_duration(self) -> Optional[float]:
        """Duration in seconds, derived from the array when not supplied."""
        if self.duration is not None:
            return self.duration
        if self.audio_array is not None and self.sample_rate:
            return len(self.audio_array) / self.sample_rate
        return None


@dataclass
class Dataset:
    """A named, lazily-iterated collection of utterances."""

    name: str
    language: str
    loader: Callable[[], Iterator[Utterance]]
    description: str = ""
    #: Approximate first-run download in MB, or None for local data.
    download_mb: Optional[int] = None

    def __iter__(self) -> Iterator[Utterance]:
        return self.loader()


# ---------------------------------------------------------------------------
# Hugging Face loaders
# ---------------------------------------------------------------------------


def _require_datasets():
    try:
        import datasets  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(_DATASETS_HINT) from exc
    return datasets


def _iter_rows(ds, text_field: str, limit: Optional[int]) -> Iterator[Utterance]:
    """Convert dataset rows into :class:`Utterance` objects."""
    for index, row in enumerate(ds):
        if limit is not None and index >= limit:
            return
        audio = row["audio"]
        yield Utterance(
            id=str(row.get("id", index)),
            reference=row[text_field],
            audio_array=audio["array"],
            sample_rate=audio["sampling_rate"],
        )


def _hf_parquet_utterances(
    parquet_url: str,
    text_field: str,
    limit: Optional[int] = None,
) -> Iterator[Utterance]:
    """Stream utterances from a single parquet file on the Hub.

    Addressing the parquet export directly rather than going through
    ``load_dataset(repo, config)`` is not an optimisation — it is required.
    ``google/fleurs`` carries no loading script and has a non-standard layout,
    so the config-based path tries to resolve all ~1000 files in the repo and
    effectively hangs.
    """
    datasets = _require_datasets()

    ds = datasets.load_dataset(
        "parquet", data_files=parquet_url, split="train", streaming=True
    )
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=DEFAULT_SAMPLE_RATE))
    return _iter_rows(ds, text_field, limit)


def _hf_config_utterances(
    repo: str,
    config: str,
    split: str,
    text_field: str,
    limit: Optional[int] = None,
) -> Iterator[Utterance]:
    """Stream from a conventionally-laid-out Hub dataset."""
    datasets = _require_datasets()

    ds = datasets.load_dataset(repo, config, split=split, streaming=True)
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=DEFAULT_SAMPLE_RATE))
    return _iter_rows(ds, text_field, limit)


#: Approximate download size per FLEURS split, in MB. Reported before a run so
#: a 1 GB fetch is a choice rather than a surprise.
FLEURS_TEST_SIZE_MB = {"fil_ph": 1051, "en_us": 1200, "ceb_ph": 510}


def fleurs_parquet_url(config: str, split: str = "test") -> str:
    return (
        f"hf://datasets/google/fleurs/parquet-data/{config}/"
        f"{split}-00000-of-00001.parquet"
    )


def load_fleurs(config: str, split: str = "test", limit: Optional[int] = None):
    """FLEURS loader. ``config`` is e.g. ``"fil_ph"`` or ``"en_us"``.

    Note that ``limit`` does not reduce the download meaningfully: audio is
    embedded in one large parquet file per split, so streaming still pulls
    substantial row groups. Budget for the full download on first run; it is
    cached afterwards.
    """
    return lambda: _hf_parquet_utterances(
        fleurs_parquet_url(config, split), "transcription", limit
    )


def load_common_voice(
    config: str,
    split: str = "test",
    limit: Optional[int] = None,
    version: str = "mozilla-foundation/common_voice_17_0",
):
    """Common Voice loader.

    Gated: accept the dataset terms on the Hub and set ``HF_TOKEN`` first, or
    this fails with a 401.
    """
    return lambda: _hf_config_utterances(version, config, split, "sentence", limit)


# ---------------------------------------------------------------------------
# Local manifest loader
# ---------------------------------------------------------------------------


def load_manifest(path: str, limit: Optional[int] = None):
    """Loader for a local JSONL manifest — the bring-your-own-audio path."""

    def _load() -> Iterator[Utterance]:
        manifest = Path(path).resolve()
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")

        base = manifest.parent
        with manifest.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if limit is not None and index >= limit:
                    return

                row = json.loads(line)
                audio = Path(row["audio"])
                if not audio.is_absolute():
                    audio = base / audio

                yield Utterance(
                    id=str(row.get("id", index)),
                    reference=row["text"],
                    audio_path=str(audio),
                    duration=row.get("duration"),
                )

    return _load


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Datasets addressable by name from ``meeting-notes bench --dataset``.
DATASET_SPECS = {
    "fleurs-fil": {
        "language": "tl",
        "factory": lambda limit: load_fleurs("fil_ph", limit=limit),
        "description": "FLEURS Filipino test split (~400 utts, clean read speech)",
        "download_mb": FLEURS_TEST_SIZE_MB["fil_ph"],
    },
    "fleurs-en": {
        "language": "en",
        "factory": lambda limit: load_fleurs("en_us", limit=limit),
        "description": "FLEURS US English test split",
        "download_mb": FLEURS_TEST_SIZE_MB["en_us"],
    },
    "fleurs-ceb": {
        "language": "ceb",
        "factory": lambda limit: load_fleurs("ceb_ph", limit=limit),
        "description": "FLEURS Cebuano test split (541 utts, clean read speech)",
        "download_mb": FLEURS_TEST_SIZE_MB["ceb_ph"],
    },
    "common-voice-tl": {
        "language": "tl",
        "factory": lambda limit: load_common_voice("tl", limit=limit),
        "description": "Common Voice Tagalog test split (gated — needs HF_TOKEN)",
        "download_mb": None,
    },
    "common-voice-en": {
        "language": "en",
        "factory": lambda limit: load_common_voice("en", limit=limit),
        "description": "Common Voice English test split (gated — needs HF_TOKEN)",
        "download_mb": None,
    },
}

DATASET_NAMES = tuple(DATASET_SPECS)


def get_dataset(name: str, limit: Optional[int] = None) -> Dataset:
    """Build a named dataset from the registry."""
    spec = DATASET_SPECS.get(name)
    if spec is None:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {', '.join(DATASET_NAMES)}"
        )
    return Dataset(
        name=name,
        language=spec["language"],
        loader=spec["factory"](limit),
        description=spec["description"],
        download_mb=spec.get("download_mb"),
    )


def manifest_dataset(
    path: str,
    language: str = "tl",
    limit: Optional[int] = None,
) -> Dataset:
    """Build a dataset from a local JSONL manifest."""
    return Dataset(
        name=Path(path).stem,
        language=language,
        loader=load_manifest(path, limit),
        description=f"Local manifest: {path}",
    )
