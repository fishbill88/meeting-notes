"""Summarizer module – generates structured meeting notes from a transcript.

Two backends are supported, both fully offline:

1. **Ollama** – a locally-running LLM server (recommended).  Install from
   https://ollama.com and pull a model, e.g.::

       ollama pull llama3

2. **Extractive fallback** – a lightweight, dependency-free approach that
   scores sentences and picks the most informative ones.  Used automatically
   when Ollama is unavailable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MeetingNotes:
    """Structured meeting notes produced by the summarizer."""

    title: str
    summary: str
    key_topics: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    language: str = "en"

    def to_markdown(self, transcript: Optional[str] = None, date_str: Optional[str] = None) -> str:
        """Render the notes as a Markdown string."""
        lines: list[str] = []

        lines.append(f"# {self.title}")
        lines.append("")

        if date_str:
            lines.append(f"**Date:** {date_str}")
            lines.append("")

        if self.language and self.language != "en":
            lines.append(f"**Language detected:** {self.language}")
            lines.append("")

        lines.append("## Summary")
        lines.append("")
        lines.append(self.summary)
        lines.append("")

        if self.key_topics:
            lines.append("## Key Topics")
            lines.append("")
            for topic in self.key_topics:
                lines.append(f"- {topic}")
            lines.append("")

        if self.decisions:
            lines.append("## Decisions Made")
            lines.append("")
            for decision in self.decisions:
                lines.append(f"- {decision}")
            lines.append("")

        if self.action_items:
            lines.append("## Action Items")
            lines.append("")
            for item in self.action_items:
                lines.append(f"- [ ] {item}")
            lines.append("")

        if transcript:
            lines.append("## Full Transcript")
            lines.append("")
            lines.append(transcript)
            lines.append("")

        return "\n".join(lines)


class Summarizer:
    """Generates structured meeting notes from a plain-text transcript.

    Args:
        use_ollama: Whether to attempt Ollama first.
        ollama_url: Base URL of the Ollama server.
        ollama_model: Name of the Ollama model to use (e.g. ``"llama3"``).
    """

    def __init__(
        self,
        use_ollama: bool = True,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3",
    ) -> None:
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model

    def summarize(self, transcript: str, language: str = "en") -> MeetingNotes:
        """Generate structured meeting notes from *transcript*.

        Tries Ollama first; falls back to extractive summarization if Ollama
        is unavailable or returns an error.

        Args:
            transcript: Full plain-text transcript of the meeting.
            language: Detected language code (e.g. ``"tl"`` for Filipino).

        Returns:
            :class:`MeetingNotes` instance.
        """
        if not transcript.strip():
            return MeetingNotes(
                title="Meeting Notes",
                summary="(No transcript content to summarize.)",
                language=language,
            )

        if self.use_ollama:
            try:
                return self._summarize_ollama(transcript, language)
            except Exception:  # noqa: BLE001
                pass  # Fall through to extractive

        return self._summarize_extractive(transcript, language)

    # ------------------------------------------------------------------
    # Ollama backend
    # ------------------------------------------------------------------

    def _summarize_ollama(self, transcript: str, language: str) -> MeetingNotes:
        """Call a local Ollama LLM to produce structured notes."""
        import json
        import requests  # noqa: PLC0415

        prompt = _build_prompt(transcript, language)

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        raw = response.json().get("response", "")
        return _parse_llm_response(raw, language)

    # ------------------------------------------------------------------
    # Extractive fallback
    # ------------------------------------------------------------------

    def _summarize_extractive(self, transcript: str, language: str) -> MeetingNotes:
        """Lightweight extractive summarization – no external dependencies."""
        sentences = _split_sentences(transcript)

        if not sentences:
            return MeetingNotes(
                title="Meeting Notes",
                summary=transcript[:500],
                language=language,
            )

        # Score sentences by word frequency (TF-based)
        scored = _score_sentences(sentences)

        # Top sentences for summary (up to 5)
        top_n = min(5, max(1, len(sentences) // 4))
        top_indices = sorted(
            sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)[:top_n]
        )
        summary = " ".join(sentences[i] for i in top_indices)

        # Detect action items: sentences containing imperative/action keywords
        action_keywords = {
            "en": ["will", "should", "must", "need to", "action", "follow up",
                   "assign", "deadline", "by ", "todo", "to-do", "complete"],
            "tl": ["kailangang", "gagawin", "dapat", "susunod", "follow up",
                   "tapusin", "itutuloy"],
        }
        lang_keywords = action_keywords.get(language, action_keywords["en"])
        actions = [
            s.strip()
            for s in sentences
            if any(kw in s.lower() for kw in lang_keywords)
        ][:8]

        # Key topics: take the longest sentences as likely topic sentences
        topic_candidates = sorted(sentences, key=len, reverse=True)[:6]
        key_topics = list(dict.fromkeys(
            _first_n_words(s, 10) for s in topic_candidates
        ))[:5]

        title = _derive_title(transcript)

        return MeetingNotes(
            title=title,
            summary=summary,
            key_topics=key_topics,
            action_items=actions,
            language=language,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_prompt(transcript: str, language: str) -> str:
    """Build the LLM prompt for structured meeting note generation."""
    lang_hint = f" (transcript language code: {language})" if language != "en" else ""

    return f"""You are an expert meeting note taker{lang_hint}.

Given the following meeting transcript, produce structured meeting notes in JSON format with these fields:
- "title": a short descriptive title for this meeting (string)
- "summary": a concise summary of the meeting (2-4 sentences, string)
- "key_topics": list of main topics discussed (list of strings)
- "decisions": list of decisions made during the meeting (list of strings)
- "action_items": list of action items or follow-ups (list of strings)

Respond ONLY with valid JSON. No explanations outside the JSON.

TRANSCRIPT:
{transcript[:4000]}

JSON:"""


def _parse_llm_response(raw: str, language: str) -> MeetingNotes:
    """Parse the LLM JSON response into a :class:`MeetingNotes`."""
    import json

    # Extract JSON block if the model wrapped it in markdown code fences
    json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    json_str = json_match.group(1) if json_match else raw.strip()

    # Try to find the first complete JSON object
    brace_match = re.search(r"\{[\s\S]*\}", json_str)
    if brace_match:
        json_str = brace_match.group(0)

    data = json.loads(json_str)

    return MeetingNotes(
        title=data.get("title", "Meeting Notes"),
        summary=data.get("summary", ""),
        key_topics=data.get("key_topics", []),
        decisions=data.get("decisions", []),
        action_items=data.get("action_items", []),
        language=language,
    )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitter using punctuation
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _score_sentences(sentences: list[str]) -> list[float]:
    """Score sentences by normalised term frequency."""
    from collections import Counter

    words = []
    for s in sentences:
        words.extend(re.findall(r"\b\w+\b", s.lower()))

    # Remove very common stop words (English + Filipino)
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "it", "this", "that", "we", "i", "he", "she", "they", "you",
        # Filipino stop words
        "ang", "ng", "sa", "na", "at", "ay", "ko", "mo", "niya", "namin",
        "natin", "ninyo", "nila", "ito", "iyon", "si", "ni", "kay",
    }
    freq = Counter(w for w in words if w not in stopwords)

    scores = []
    for s in sentences:
        sentence_words = re.findall(r"\b\w+\b", s.lower())
        score = sum(freq[w] for w in sentence_words if w not in stopwords)
        scores.append(score / max(len(sentence_words), 1))

    return scores


def _first_n_words(text: str, n: int) -> str:
    """Return the first *n* words of *text* followed by '…' if truncated."""
    words = text.split()
    if len(words) <= n:
        return text
    return " ".join(words[:n]) + "…"


def _derive_title(transcript: str) -> str:
    """Derive a short meeting title from the beginning of the transcript."""
    first_100 = transcript.strip()[:100]
    words = first_100.split()[:6]
    title = " ".join(words).strip(".,!?;:")
    return title if title else "Meeting Notes"
