"""Configuration for the meeting-notes app."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# Whisper model sizes, as understood by faster-whisper (CTranslate2).
#
# tiny            ~75 MB   – fast, low accuracy
# base            ~145 MB  – quick smoke tests
# small           ~484 MB  – usable
# medium          ~1.5 GB  – good, slow on CPU
# large-v3        ~3.1 GB  – best accuracy, well under real time on CPU
# large-v3-turbo  ~1.6 GB  – large-v2-class accuracy, ~7x faster than large-v3
#
# large-v3-turbo is the default: it is the only large-class model that keeps
# up on clinic hardware without a GPU. See MODELS.md.
WHISPER_MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "turbo",
)

DEFAULT_WHISPER_MODEL = "large-v3-turbo"


class LanguageMode(str, Enum):
    """Which language the app expects to hear.

    This replaces a bare ISO code because Taglish is not a language code — it
    is a decoding strategy over two engines. Modelling it as a mode keeps that
    distinction explicit instead of hiding it behind ``--language``.
    """

    ENGLISH = "en"
    TAGALOG = "tl"
    #: Cebuano / Bisaya. Routes to the Omnilingual CTC engine (sherpa-onnx);
    #: Whisper has no Cebuano token, so its fallback decodes with ``tl``.
    CEBUANO = "ceb"
    AUTO = "auto"

    #: Mixed English/Tagalog. Scaffolded but not implemented — see
    #: ``backends.router.merge_taglish``. ``AppConfig.validate`` rejects it.
    TAGLISH = "taglish"

    @classmethod
    def from_value(cls, value: Optional[str]) -> "LanguageMode":
        """Coerce a CLI/API string to a mode, defaulting to auto-detect."""
        if value is None or value == "":
            return cls.AUTO
        if isinstance(value, cls):
            return value
        normalised = str(value).strip().lower()
        # "fil" is the other ISO code for Filipino; treat it as Tagalog.
        if normalised == "fil":
            return cls.TAGALOG
        # Everyday names for Cebuano; ISO 639-3 is ``ceb``.
        if normalised in ("bisaya", "binisaya", "cebuano", "sugbuanon"):
            return cls.CEBUANO
        for mode in cls:
            if mode.value == normalised:
                return mode
        # Any other Whisper-supported code passes through as auto-detect with
        # an explicit hint; the router handles it via the Whisper backend.
        return cls.AUTO


#: Modes that are implemented end to end today.
IMPLEMENTED_LANGUAGE_MODES = (
    LanguageMode.ENGLISH,
    LanguageMode.TAGALOG,
    LanguageMode.CEBUANO,
    LanguageMode.AUTO,
)


# Languages supported by Whisper (sample; Whisper supports 99+ languages)
SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "tl": "Filipino / Tagalog",
    "fil": "Filipino / Tagalog",
    "ceb": "Cebuano / Bisaya",
    "taglish": "Taglish (mixed English/Tagalog)",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ar": "Arabic",
    "hi": "Hindi",
    "id": "Indonesian",
    "ms": "Malay",
}

# Default output directory for meeting documents
DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "meeting-notes")

# Ollama default endpoint (for local LLM summarization)
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"

# Audio recording defaults
DEFAULT_SAMPLE_RATE = 16000  # Hz – Whisper expects 16 kHz
DEFAULT_CHANNELS = 1         # Mono
DEFAULT_CHUNK_SECONDS = 30   # Seconds per audio chunk for live transcription

#: Environment variable overriding where downloaded/exported models live.
MODELS_DIR_ENV = "MEETING_NOTES_MODELS_DIR"


def models_dir() -> Path:
    """Directory holding exported ONNX models and other engine assets.

    Defaults to ``~/.meeting-notes/models``. The desktop app overrides this
    with its Tauri app-data directory via :data:`MODELS_DIR_ENV`.
    """
    override = os.environ.get(MODELS_DIR_ENV)
    if override:
        # resolve() pins a relative override to a stable absolute path —
        # model downloads must not land wherever the CWD happens to be.
        return Path(override).expanduser().resolve()
    return Path.home() / ".meeting-notes" / "models"


@dataclass
class AppConfig:
    """Runtime configuration for meeting-notes."""

    # Transcription
    whisper_model: str = DEFAULT_WHISPER_MODEL
    language: Optional[str] = None  # None = auto-detect (legacy field)
    language_mode: LanguageMode = LanguageMode.AUTO
    backend: Optional[str] = None   # force a backend by name; None = router picks
    device: str = "cpu"             # "cpu", "cuda", or "auto"
    compute_type: str = "auto"      # CTranslate2 quantisation
    #: Label transcript segments with "Person N" speakers (sherpa-onnx).
    #: Safe to leave on: silently skipped when the engine or its models are
    #: unavailable — see meeting_notes.diarize.
    diarize: bool = True

    # Summarizer
    use_ollama: bool = True
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    template: str = "meeting"  # or "consultation" — see summarizer.TEMPLATES

    # Document output
    output_dir: str = DEFAULT_OUTPUT_DIR
    output_format: str = "markdown"  # "markdown" or "docx"

    # Audio
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS

    def __post_init__(self) -> None:
        # Accept a plain string for language_mode (JSON payloads, CLI values).
        if not isinstance(self.language_mode, LanguageMode):
            self.language_mode = LanguageMode.from_value(self.language_mode)

        # Legacy: callers passing only `language` still get the right mode.
        if self.language and self.language_mode is LanguageMode.AUTO:
            self.language_mode = LanguageMode.from_value(self.language)

    def validate(self) -> None:
        """Raise ValueError if configuration values are invalid."""
        if self.whisper_model not in WHISPER_MODELS:
            raise ValueError(
                f"Invalid whisper_model '{self.whisper_model}'. "
                f"Choose from: {', '.join(WHISPER_MODELS)}"
            )
        if self.language_mode is LanguageMode.TAGLISH:
            raise ValueError(
                "Taglish mode is not implemented yet. It needs the two-pass "
                "merge in backends.router.merge_taglish plus a Taglish test "
                "set to validate against. Use 'en' or 'tl' for now."
            )
        if self.language_mode not in IMPLEMENTED_LANGUAGE_MODES:
            raise ValueError(
                f"Unsupported language mode '{self.language_mode.value}'. "
                f"Choose from: "
                f"{', '.join(m.value for m in IMPLEMENTED_LANGUAGE_MODES)}"
            )
        # Imported here to avoid a module-level cycle: summarizer imports
        # nothing from config, but keeping the dependency local is cheap.
        from meeting_notes.summarizer import TEMPLATE_NAMES  # noqa: PLC0415

        if self.template not in TEMPLATE_NAMES:
            raise ValueError(
                f"Invalid template '{self.template}'. "
                f"Choose from: {', '.join(TEMPLATE_NAMES)}"
            )
        if self.output_format not in ("markdown", "docx"):
            raise ValueError(
                f"Invalid output_format '{self.output_format}'. "
                "Choose 'markdown' or 'docx'."
            )
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 (mono) or 2 (stereo).")

    @property
    def resolved_language(self) -> Optional[str]:
        """The ISO code to force on the engine, or None to auto-detect."""
        if self.language_mode is LanguageMode.AUTO:
            return None
        return self.language_mode.value
