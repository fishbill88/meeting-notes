"""Configuration for the meeting-notes app."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# Whisper model sizes (larger = more accurate, slower, more disk space)
# tiny   ~39 MB   – fast, lower accuracy
# base   ~74 MB   – good balance for offline use
# small  ~244 MB  – better accuracy
# medium ~769 MB  – high accuracy
# large  ~1.5 GB  – best accuracy
WHISPER_MODELS = ("tiny", "base", "small", "medium", "large")

# Languages supported by Whisper (sample; Whisper supports 99+ languages)
SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "tl": "Filipino / Tagalog",
    "fil": "Filipino / Tagalog",
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


@dataclass
class AppConfig:
    """Runtime configuration for meeting-notes."""

    # Whisper
    whisper_model: str = "base"
    language: Optional[str] = None  # None = auto-detect

    # Summarizer
    use_ollama: bool = True
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL

    # Document output
    output_dir: str = DEFAULT_OUTPUT_DIR
    output_format: str = "markdown"  # "markdown" or "docx"

    # Audio
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS

    def validate(self) -> None:
        """Raise ValueError if configuration values are invalid."""
        if self.whisper_model not in WHISPER_MODELS:
            raise ValueError(
                f"Invalid whisper_model '{self.whisper_model}'. "
                f"Choose from: {', '.join(WHISPER_MODELS)}"
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
