"""Chooses a transcription backend from the configured language mode.

The routing table is the whole v1 architecture:

===============  ==========================  ==================================
Language mode    Backend                     Why
===============  ==========================  ==================================
``en``           faster-whisper              99-language model, strong English
``tl``           NeMo FastConformer          9.34% FLEURS fil_ph vs Whisper's
                 (falls back to Whisper)     unreported/weaker Tagalog
``ceb``          Omnilingual CTC             Whisper has no Cebuano at all;
                 (falls back to Whisper/tl)  Meta's model has 197 h of it
``auto``         faster-whisper              Whisper detects the language
``taglish``      *not implemented*           Needs the two-pass merge below
===============  ==========================  ==================================

Taglish is a merge of the ``en`` and ``tl`` passes, which is why building
those two as independently routed engines is the groundwork for it rather
than a detour around it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from meeting_notes.config import AppConfig, LanguageMode

if TYPE_CHECKING:  # pragma: no cover
    from meeting_notes.backends.base import (
        TranscriptionBackend,
        TranscriptionResult,
    )

logger = logging.getLogger(__name__)

#: Backend names accepted by ``AppConfig.backend`` and ``--backend``.
BACKEND_NAMES = ("faster-whisper", "nemo-tagalog", "omnilingual-ctc")

#: Language modes Whisper cannot be told about. faster-whisper rejects codes
#: outside its 99-token vocabulary, so a fallback decode uses the nearest
#: token — Tagalog for Cebuano, which every public Cebuano Whisper fine-tune
#: also does (documented positive transfer between the two).
WHISPER_LANGUAGE_ALIASES = {"ceb": "tl"}


def resolve_backend(config: AppConfig) -> "TranscriptionBackend":
    """Return the backend that should transcribe under ``config``.

    Args:
        config: Validated app configuration.

    Returns:
        A ready-to-use backend. Model loading stays lazy — this does not
        touch disk or download anything.

    Raises:
        NotImplementedError: If the mode is Taglish.
        ValueError: If ``config.backend`` names an unknown backend.
    """
    if config.language_mode is LanguageMode.TAGLISH:
        raise NotImplementedError(
            "Taglish mode is not implemented. See merge_taglish() in this "
            "module for the intended design."
        )

    if config.backend:
        return _build_named(config.backend, config)

    if config.language_mode is LanguageMode.TAGALOG:
        return _build_tagalog(config)

    if config.language_mode is LanguageMode.CEBUANO:
        return _build_cebuano(config)

    return _build_whisper(config)


def available_backends(config: Optional[AppConfig] = None) -> dict[str, bool]:
    """Report which backends can run right now.

    Used by the CLI's diagnostics and by the desktop sidecar's startup check,
    so a missing model surfaces as a clear message instead of a stack trace on
    first recording.
    """
    config = config or AppConfig()
    status: dict[str, bool] = {}
    for name in BACKEND_NAMES:
        try:
            status[name] = _build_named(name, config).is_available()
        except Exception:  # noqa: BLE001 – diagnostics must never raise
            status[name] = False
    return status


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_named(name: str, config: AppConfig) -> "TranscriptionBackend":
    """Build a backend by explicit name, bypassing language routing."""
    if name == "faster-whisper":
        return _build_whisper(config)
    if name == "nemo-tagalog":
        return _build_tagalog(config, allow_fallback=False)
    if name == "omnilingual-ctc":
        return _build_omnilingual(config, allow_fallback=False)
    raise ValueError(
        f"Unknown backend '{name}'. Choose from: {', '.join(BACKEND_NAMES)}"
    )


def _build_whisper(config: AppConfig) -> "TranscriptionBackend":
    from meeting_notes.backends.whisper import FasterWhisperBackend  # noqa: PLC0415

    language = config.resolved_language
    if language in WHISPER_LANGUAGE_ALIASES:
        language = WHISPER_LANGUAGE_ALIASES[language]

    return FasterWhisperBackend(
        model_size=config.whisper_model,
        language=language,
        device=config.device,
        compute_type=config.compute_type,
    )


def _build_tagalog(
    config: AppConfig,
    allow_fallback: bool = True,
) -> "TranscriptionBackend":
    """Build the Tagalog engine, falling back to Whisper when unexported.

    The fallback is deliberate: a clinician who has not run the one-time ONNX
    export should still get a transcript, just a worse one. Which engine
    actually ran is recorded in ``TranscriptionResult.backend`` so the UI can
    say so rather than quietly pretending.
    """
    from meeting_notes.backends.nemo import NemoTagalogBackend  # noqa: PLC0415

    backend = NemoTagalogBackend()
    if backend.is_available() or not allow_fallback:
        return backend

    logger.warning(
        "Tagalog engine unavailable (%s); falling back to faster-whisper with "
        "language='tl'. Run scripts/export_nemo_to_onnx.py for better Tagalog "
        "accuracy.",
        ", ".join(backend.missing_files()) or "sherpa-onnx not installed",
    )
    return _build_whisper(config)


def _build_omnilingual(
    config: AppConfig,
    allow_fallback: bool = True,
) -> "TranscriptionBackend":
    """Build the Omnilingual CTC engine for whatever language was asked for.

    Used both for the ``ceb`` route and for ``--backend omnilingual-ctc``,
    where the model (language-agnostic) is stamped with the requested code,
    or Cebuano when the mode is auto.
    """
    from meeting_notes.backends.omnilingual import (  # noqa: PLC0415
        DEFAULT_LANGUAGE,
        OmnilingualBackend,
    )

    backend = OmnilingualBackend(language=config.resolved_language or DEFAULT_LANGUAGE)
    if backend.is_available() or not allow_fallback:
        return backend

    logger.warning(
        "Omnilingual engine unavailable (sherpa-onnx not installed); falling "
        "back to faster-whisper with language='tl' — Whisper has no Cebuano."
    )
    return _build_whisper(config)


def _build_cebuano(config: AppConfig) -> "TranscriptionBackend":
    """Route Cebuano/Bisaya to Omnilingual, degrading to Whisper-as-Tagalog."""
    return _build_omnilingual(config, allow_fallback=True)


# ---------------------------------------------------------------------------
# Taglish — scaffolded, not implemented
# ---------------------------------------------------------------------------


def merge_taglish(
    english: "TranscriptionResult",
    tagalog: "TranscriptionResult",
) -> "TranscriptionResult":
    """Merge an English-primary and a Tagalog-primary pass into one transcript.

    **Not implemented.** This is the hook Phase 7 fills in.

    Whisper picks one primary language at the start of the audio and applies it
    throughout, so secondary-language phrases come out garbled or
    transliterated. The documented workaround for heavily bilingual audio is to
    decode twice — once with each language forced — and merge per segment.

    The intended algorithm:

    1. Align the two segment lists on their time ranges (they will not share
       boundaries; both engines segment independently).
    2. For each aligned span pick the higher-confidence decode. Whisper's
       ``avg_logprob`` is already carried on
       :class:`~meeting_notes.backends.base.TranscriptSegment` as
       ``confidence`` for exactly this. The NeMo backend reports no score, so
       this needs either a score exported from sherpa-onnx or a different
       tiebreak — an open question.
    3. Tag each kept segment with the language that won, populating ``lang``.
    4. Run the winner through ``taglish.normalize`` to repair hybrid forms
       (``nag-meeting``, ``i-refer``) that neither engine spells correctly.

    Blocked on a Taglish test set — without one there is no way to tell a
    working merge from a plausible-looking broken one.
    """
    raise NotImplementedError(
        "Taglish merge is scaffolded but not implemented — see the docstring "
        "for the intended algorithm and what it is blocked on."
    )
