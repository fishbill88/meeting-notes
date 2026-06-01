"""Tests for the configuration module."""

import pytest

from meeting_notes.config import AppConfig, WHISPER_MODELS


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
