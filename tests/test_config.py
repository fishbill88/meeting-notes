"""Tests for the configuration module."""

import pytest

from meeting_notes.config import (
    DEFAULT_WHISPER_MODEL,
    MODELS_DIR_ENV,
    WHISPER_MODELS,
    AppConfig,
    LanguageMode,
    models_dir,
)


def test_default_config_is_valid():
    cfg = AppConfig()
    cfg.validate()  # should not raise


def test_invalid_whisper_model_raises():
    cfg = AppConfig(whisper_model="huge")
    with pytest.raises(ValueError, match="Invalid whisper_model"):
        cfg.validate()


def test_invalid_output_format_raises():
    cfg = AppConfig(output_format="pdf")
    with pytest.raises(ValueError, match="Invalid output_format"):
        cfg.validate()


def test_invalid_sample_rate_raises():
    cfg = AppConfig(sample_rate=0)
    with pytest.raises(ValueError, match="sample_rate"):
        cfg.validate()


def test_invalid_channels_raises():
    cfg = AppConfig(channels=3)
    with pytest.raises(ValueError, match="channels"):
        cfg.validate()


@pytest.mark.parametrize("model", WHISPER_MODELS)
def test_all_whisper_model_sizes_accepted(model):
    cfg = AppConfig(whisper_model=model)
    cfg.validate()


@pytest.mark.parametrize("fmt", ["markdown", "docx"])
def test_both_output_formats_accepted(fmt):
    cfg = AppConfig(output_format=fmt)
    cfg.validate()


def test_default_model_is_the_turbo_english_engine():
    assert AppConfig().whisper_model == DEFAULT_WHISPER_MODEL
    assert DEFAULT_WHISPER_MODEL in WHISPER_MODELS


# ---------------------------------------------------------------------------
# LanguageMode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("en", LanguageMode.ENGLISH),
        ("tl", LanguageMode.TAGALOG),
        ("fil", LanguageMode.TAGALOG),  # the other ISO code for Filipino
        ("TL", LanguageMode.TAGALOG),
        ("ceb", LanguageMode.CEBUANO),
        ("Bisaya", LanguageMode.CEBUANO),  # everyday name for Cebuano
        ("cebuano", LanguageMode.CEBUANO),
        ("auto", LanguageMode.AUTO),
        ("taglish", LanguageMode.TAGLISH),
        (None, LanguageMode.AUTO),
        ("", LanguageMode.AUTO),
        ("es", LanguageMode.AUTO),  # unknown codes detect through Whisper
    ],
)
def test_language_mode_from_value(value, expected):
    assert LanguageMode.from_value(value) == expected


def test_language_mode_from_value_is_idempotent():
    assert LanguageMode.from_value(LanguageMode.TAGALOG) is LanguageMode.TAGALOG


@pytest.mark.parametrize("mode", ["en", "tl", "ceb", "auto"])
def test_implemented_modes_validate(mode):
    AppConfig(language_mode=mode).validate()


def test_taglish_is_rejected_with_a_useful_message():
    cfg = AppConfig(language_mode="taglish")
    with pytest.raises(ValueError, match="not implemented yet"):
        cfg.validate()


def test_string_language_mode_is_coerced():
    assert AppConfig(language_mode="tl").language_mode is LanguageMode.TAGALOG


def test_legacy_language_field_sets_the_mode():
    # Callers built before language_mode existed must keep working.
    assert AppConfig(language="tl").language_mode is LanguageMode.TAGALOG


def test_explicit_mode_wins_over_legacy_language():
    cfg = AppConfig(language="tl", language_mode=LanguageMode.ENGLISH)
    assert cfg.language_mode is LanguageMode.ENGLISH


def test_resolved_language_is_none_for_auto():
    assert AppConfig(language_mode="auto").resolved_language is None


@pytest.mark.parametrize("mode,code", [("en", "en"), ("tl", "tl"), ("ceb", "ceb")])
def test_resolved_language_returns_iso_code(mode, code):
    assert AppConfig(language_mode=mode).resolved_language == code


# ---------------------------------------------------------------------------
# Models directory
# ---------------------------------------------------------------------------


def test_models_dir_defaults_under_home(monkeypatch):
    monkeypatch.delenv(MODELS_DIR_ENV, raising=False)
    assert models_dir().name == "models"


def test_models_dir_honours_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path))
    assert models_dir() == tmp_path
