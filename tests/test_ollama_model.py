"""Tests for Ollama model resolution.

Ollama matches model names exactly, and a failed Ollama call falls back
silently to the extractive summarizer. A name mismatch therefore has no visible
symptom beyond notes that are quietly much worse — which is exactly the kind of
thing that only shows up in front of a client.
"""

from __future__ import annotations

import pytest

from meeting_notes.summarizer import Summarizer, _match_model


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


def install_catalogue(monkeypatch, names, *, fail=False):
    """Fake the Ollama /api/tags response."""
    import requests

    def fake_get(url, timeout=None):
        if fail:
            raise ConnectionError("Ollama is down")
        return FakeResponse({"models": [{"name": n} for n in names]})

    monkeypatch.setattr(requests, "get", fake_get)


# ---------------------------------------------------------------------------
# _match_model
# ---------------------------------------------------------------------------


def test_exact_name_wins():
    assert _match_model("llama3", ["mistral", "llama3"]) == "llama3"


def test_tagged_variant_matches_untagged_config():
    # The actual bug: configured "llama3.2", installed "llama3.2:latest".
    assert _match_model("llama3.2", ["llama3.2:latest"]) == "llama3.2:latest"


def test_untagged_config_does_not_match_a_different_family():
    # "llama3" must not silently grab "llama3.2:latest" as if it were the same
    # model — but any working model still beats the extractive fallback.
    assert _match_model("llama3", ["llama3.2:latest"]) == "llama3.2:latest"


def test_exact_match_preferred_over_family_match():
    assert _match_model("llama3", ["llama3.2:latest", "llama3"]) == "llama3"


def test_falls_back_to_the_first_installed_model():
    assert _match_model("gemma", ["qwen3:latest", "mistral"]) == "qwen3:latest"


# ---------------------------------------------------------------------------
# Summarizer.resolve_model
# ---------------------------------------------------------------------------


def test_resolves_to_an_installed_model(monkeypatch):
    install_catalogue(monkeypatch, ["llama3.2:latest"])
    assert Summarizer(ollama_model="llama3").resolve_model() == "llama3.2:latest"


def test_keeps_the_configured_model_when_present(monkeypatch):
    install_catalogue(monkeypatch, ["llama3", "mistral"])
    assert Summarizer(ollama_model="llama3").resolve_model() == "llama3"


def test_keeps_configured_name_when_ollama_is_unreachable(monkeypatch):
    # The caller's own error handling should run, not a guess.
    install_catalogue(monkeypatch, [], fail=True)
    assert Summarizer(ollama_model="llama3").resolve_model() == "llama3"


def test_keeps_configured_name_when_nothing_is_installed(monkeypatch):
    install_catalogue(monkeypatch, [])
    assert Summarizer(ollama_model="llama3").resolve_model() == "llama3"


def test_resolution_is_cached(monkeypatch):
    calls = []
    import requests

    def counting_get(url, timeout=None):
        calls.append(url)
        return FakeResponse({"models": [{"name": "llama3.2:latest"}]})

    monkeypatch.setattr(requests, "get", counting_get)

    summarizer = Summarizer(ollama_model="llama3")
    summarizer.resolve_model()
    summarizer.resolve_model()

    assert len(calls) == 1, "catalogue should be fetched once per Summarizer"


def test_unreachable_result_is_not_cached(monkeypatch):
    # Ollama may start after the app does; a transient failure must not pin
    # the wrong model for the process lifetime.
    install_catalogue(monkeypatch, [], fail=True)
    summarizer = Summarizer(ollama_model="llama3")
    assert summarizer.resolve_model() == "llama3"

    install_catalogue(monkeypatch, ["llama3.2:latest"])
    assert summarizer.resolve_model() == "llama3.2:latest"


def test_ignores_entries_without_a_name(monkeypatch):
    import requests

    monkeypatch.setattr(
        requests,
        "get",
        lambda url, timeout=None: FakeResponse(
            {"models": [{}, {"name": "mistral"}]}
        ),
    )
    assert Summarizer(ollama_model="llama3").resolve_model() == "mistral"


def test_extractive_path_never_touches_ollama(monkeypatch):
    import requests

    def explode(*args, **kwargs):
        raise AssertionError("Ollama must not be contacted when disabled")

    monkeypatch.setattr(requests, "get", explode)
    monkeypatch.setattr(requests, "post", explode)

    notes = Summarizer(use_ollama=False).summarize("Some meeting text here.")
    assert notes.summary
