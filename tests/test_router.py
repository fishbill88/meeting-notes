"""Tests for language-mode → backend routing."""

from __future__ import annotations

import pytest

from meeting_notes.backends.router import (
    BACKEND_NAMES,
    available_backends,
    merge_taglish,
    resolve_backend,
)
from meeting_notes.config import AppConfig, LanguageMode


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_english_routes_to_whisper():
    config = AppConfig(language_mode=LanguageMode.ENGLISH)
    assert resolve_backend(config).name == "faster-whisper"


def test_auto_routes_to_whisper():
    config = AppConfig(language_mode=LanguageMode.AUTO)
    assert resolve_backend(config).name == "faster-whisper"


def test_english_forces_language_on_the_engine():
    config = AppConfig(language_mode=LanguageMode.ENGLISH)
    assert resolve_backend(config).language == "en"


def test_auto_leaves_language_unset_for_detection():
    config = AppConfig(language_mode=LanguageMode.AUTO)
    assert resolve_backend(config).language is None


def test_tagalog_routes_to_nemo_when_available(monkeypatch):
    monkeypatch.setattr(
        "meeting_notes.backends.nemo.NemoTagalogBackend.is_available",
        lambda self: True,
    )
    config = AppConfig(language_mode=LanguageMode.TAGALOG)
    assert resolve_backend(config).name == "nemo-tagalog"


def test_tagalog_falls_back_to_whisper_when_unexported(monkeypatch):
    monkeypatch.setattr(
        "meeting_notes.backends.nemo.NemoTagalogBackend.is_available",
        lambda self: False,
    )
    config = AppConfig(language_mode=LanguageMode.TAGALOG)
    backend = resolve_backend(config)

    # Degraded, but still transcribes — and forces Tagalog on Whisper.
    assert backend.name == "faster-whisper"
    assert backend.language == "tl"


def test_explicit_nemo_backend_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(
        "meeting_notes.backends.nemo.NemoTagalogBackend.is_available",
        lambda self: False,
    )
    config = AppConfig(language_mode=LanguageMode.TAGALOG, backend="nemo-tagalog")
    # Asking for it by name means you want to know when it is missing.
    assert resolve_backend(config).name == "nemo-tagalog"


def test_explicit_backend_overrides_language_routing(monkeypatch):
    monkeypatch.setattr(
        "meeting_notes.backends.nemo.NemoTagalogBackend.is_available",
        lambda self: True,
    )
    config = AppConfig(language_mode=LanguageMode.TAGALOG, backend="faster-whisper")
    assert resolve_backend(config).name == "faster-whisper"


def test_unknown_backend_name_raises():
    config = AppConfig(backend="wav2vec")
    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_backend(config)


# ---------------------------------------------------------------------------
# Cebuano / Bisaya → Omnilingual
# ---------------------------------------------------------------------------


def test_cebuano_routes_to_omnilingual_when_available(monkeypatch):
    monkeypatch.setattr(
        "meeting_notes.backends.omnilingual.OmnilingualBackend.is_available",
        lambda self: True,
    )
    backend = resolve_backend(AppConfig(language_mode=LanguageMode.CEBUANO))
    assert backend.name == "omnilingual-ctc"
    assert backend.language == "ceb"


def test_cebuano_falls_back_to_whisper_as_tagalog(monkeypatch):
    monkeypatch.setattr(
        "meeting_notes.backends.omnilingual.OmnilingualBackend.is_available",
        lambda self: False,
    )
    backend = resolve_backend(AppConfig(language_mode=LanguageMode.CEBUANO))

    # Whisper has no Cebuano token; the nearest one is Tagalog. Passing "ceb"
    # through would make faster-whisper raise.
    assert backend.name == "faster-whisper"
    assert backend.language == "tl"


def test_explicit_omnilingual_backend_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(
        "meeting_notes.backends.omnilingual.OmnilingualBackend.is_available",
        lambda self: False,
    )
    config = AppConfig(language_mode=LanguageMode.CEBUANO, backend="omnilingual-ctc")
    assert resolve_backend(config).name == "omnilingual-ctc"


def test_explicit_omnilingual_backend_stamps_requested_language(monkeypatch):
    monkeypatch.setattr(
        "meeting_notes.backends.omnilingual.OmnilingualBackend.is_available",
        lambda self: True,
    )
    config = AppConfig(language_mode=LanguageMode.ENGLISH, backend="omnilingual-ctc")
    assert resolve_backend(config).language == "en"


def test_whisper_never_receives_the_cebuano_code():
    # Forcing Whisper on a Cebuano session must alias the language it cannot
    # accept, not crash inside faster-whisper.
    config = AppConfig(language_mode=LanguageMode.CEBUANO, backend="faster-whisper")
    assert resolve_backend(config).language == "tl"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_available_backends_reports_every_name():
    status = available_backends()
    assert set(status) == set(BACKEND_NAMES)
    assert all(isinstance(v, bool) for v in status.values())


def test_available_backends_never_raises(monkeypatch):
    monkeypatch.setattr(
        "meeting_notes.backends.nemo.NemoTagalogBackend.is_available",
        lambda self: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert available_backends()["nemo-tagalog"] is False


# ---------------------------------------------------------------------------
# Taglish — scaffolded, must stay explicitly unimplemented
# ---------------------------------------------------------------------------


def test_taglish_routing_raises_not_implemented():
    config = AppConfig()
    config.language_mode = LanguageMode.TAGLISH  # bypass validate()
    with pytest.raises(NotImplementedError, match="Taglish"):
        resolve_backend(config)


def test_merge_taglish_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        merge_taglish(None, None)
