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

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MeetingNotes:
    """Structured meeting notes produced by the summarizer."""

    title: str
    summary: str
    key_topics: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    language: str = "en"
    #: Drugs, conditions and procedures the model flagged. Only the
    #: consultation template asks for these; the clients merge in their own
    #: lexicon matches on top.
    medical_terms: list[str] = field(default_factory=list)

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
        self._resolved_model: Optional[str] = None

    def summarize(
        self,
        transcript: str,
        language: str = "en",
        template: str = "meeting",
    ) -> MeetingNotes:
        """Generate structured notes from *transcript*.

        Tries Ollama first; falls back to extractive summarization if Ollama
        is unavailable or returns an error.

        Args:
            transcript: Full plain-text transcript.
            language: Detected language code (e.g. ``"tl"`` for Filipino).
            template: ``"meeting"`` for generic notes, or ``"consultation"``
                for a clinical chief-complaint / history / assessment / plan
                structure. See :data:`TEMPLATES`.

        Returns:
            :class:`MeetingNotes` instance.
        """
        if template not in TEMPLATES:
            raise ValueError(
                f"Invalid template '{template}'. "
                f"Choose from: {', '.join(TEMPLATES)}"
            )

        if not transcript.strip():
            return MeetingNotes(
                title=TEMPLATES[template]["empty_title"],
                summary="(No transcript content to summarize.)",
                language=language,
            )

        if self.use_ollama:
            try:
                return self._summarize_ollama(transcript, language, template)
            except Exception:  # noqa: BLE001
                pass  # Fall through to extractive

        return self._summarize_extractive(transcript, language)

    # ------------------------------------------------------------------
    # Ollama backend
    # ------------------------------------------------------------------

    def resolve_model(self) -> str:
        """Pick a model Ollama actually has, preferring the configured one.

        Ollama matches names exactly, so a configured ``llama3`` against an
        installed ``llama3.2:latest`` returns 404 — and because an Ollama
        failure silently falls back to the extractive summarizer, the only
        symptom is notes that are quietly much worse. Rather than pin a name
        that has to be right on every machine, ask what is installed.

        Resolution order: exact name, then name ignoring the ``:tag``, then
        whatever is installed. Falls back to the configured name when the
        catalogue cannot be read, so the caller's error handling still runs.
        """
        if self._resolved_model is not None:
            return self._resolved_model

        configured = self.ollama_model
        try:
            import requests  # noqa: PLC0415

            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            available = [
                m["name"] for m in response.json().get("models", []) if m.get("name")
            ]
        except Exception:  # noqa: BLE001
            return configured

        if not available:
            return configured

        chosen = _match_model(configured, available)
        if chosen != configured:
            logger.warning(
                "Ollama model %r is not installed; using %r instead. "
                "Installed: %s",
                configured,
                chosen,
                ", ".join(available),
            )

        self._resolved_model = chosen
        return chosen

    def _summarize_ollama(
        self,
        transcript: str,
        language: str,
        template: str = "meeting",
    ) -> MeetingNotes:
        """Call a local Ollama LLM to produce structured notes."""
        import json
        import requests  # noqa: PLC0415

        prompt = _build_prompt(transcript, language, template)

        payload = {
            "model": self.resolve_model(),
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


#: Note templates. Both produce the same JSON fields — the desktop app, the
#: document writers and the TS clients all depend on that shape — but they ask
#: the model to fill them with different things.
TEMPLATES: dict[str, dict] = {
    "meeting": {
        "label": "Meeting notes",
        "empty_title": "Meeting Notes",
        "role": "an expert meeting note taker",
        "fields": (
            '- "title": a short descriptive title for this meeting (string)\n'
            '- "summary": a concise summary of the meeting (2-4 sentences, string)\n'
            '- "key_topics": list of main topics discussed (list of strings)\n'
            '- "decisions": list of decisions made during the meeting (list of strings)\n'
            '- "action_items": list of action items or follow-ups (list of strings)'
        ),
        "guidance": "",
        "source_label": "TRANSCRIPT",
    },
    "consultation": {
        "label": "Clinical consultation",
        "empty_title": "Consultation Notes",
        "role": "an experienced clinical scribe documenting a doctor-patient consultation",
        "fields": (
            '- "title": chief complaint in a few words, e.g. "Chest pain, 3 days" (string)\n'
            '- "summary": the history of present illness as clinical prose (3-6 sentences, string)\n'
            '- "key_topics": findings — symptoms, vitals, examination results (list of strings)\n'
            '- "decisions": assessment — diagnoses, clinical impressions (list of strings)\n'
            '- "action_items": plan — medications with dosage, tests ordered, follow-up (list of strings)\n'
            '- "medical_terms": drug names, conditions and procedures mentioned (list of strings)'
        ),
        # Two things this has to get right, and both are easy to get wrong:
        # the record must be English because PH charts are, but flattening the
        # patient's own words into clinical language loses information a
        # clinician may need to re-read later.
        "guidance": (
            "\nRules:\n"
            "- Write the notes in ENGLISH even when the consultation was in "
            "Tagalog or mixed Tagalog-English. Philippine medical records are "
            "kept in English.\n"
            "- Preserve the patient's own words as a direct quote, in their "
            "original language, wherever the exact phrasing carries clinical "
            "meaning.\n"
            "- Record only what was actually said. Never infer a diagnosis, "
            "dosage or test result that is not in the transcript.\n"
            "- Where something was unclear or inaudible, say so rather than "
            "guessing.\n"
        ),
        "source_label": "CONSULTATION TRANSCRIPT",
    },
}

#: Template names, for CLI choices and API validation.
TEMPLATE_NAMES = tuple(TEMPLATES)


def _match_model(configured: str, available: list[str]) -> str:
    """Choose the best installed Ollama model for a configured name."""
    if configured in available:
        return configured

    # "llama3.2" should match an installed "llama3.2:latest".
    base = configured.split(":", 1)[0]
    for name in available:
        if name.split(":", 1)[0] == base:
            return name

    # Nothing close. Any working model beats silently degrading to extractive.
    return available[0]


def _build_prompt(
    transcript: str,
    language: str,
    template: str = "meeting",
) -> str:
    """Build the LLM prompt for structured note generation."""
    spec = TEMPLATES.get(template, TEMPLATES["meeting"])
    lang_hint = f" (transcript language code: {language})" if language != "en" else ""

    return f"""You are {spec["role"]}{lang_hint}.

Given the following transcript, produce structured notes in JSON format with these fields:
{spec["fields"]}
{spec["guidance"]}
Respond ONLY with valid JSON. No explanations outside the JSON.

{spec["source_label"]}:
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
        medical_terms=data.get("medical_terms", []),
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
