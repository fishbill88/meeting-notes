"""Summarizer module – generates structured meeting notes from a transcript.

Two backends are supported, both fully offline:

1. **Ollama** – a locally-running LLM server (recommended).  Install from
   https://ollama.com and pull a model, e.g.::

       ollama pull llama3

   A long transcript is summarised in parts and the parts merged, so the notes
   cover the whole meeting rather than its first few minutes.

2. **Extractive fallback** – a lightweight, dependency-free approach that
   scores sentences and picks the most informative ones.  Used automatically
   when Ollama is unavailable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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
    #: What wrote these notes: ``"ollama:<model>"``, ``"extractive"`` (the no-model fallback), or
    #: ``""`` when nothing was summarised. A host shows this, because the two are very different
    #: quality and the fallback is reached SILENTLY whenever Ollama is not running — without it a
    #: person reads a sentence-picker's output as though a model had understood the meeting.
    generator: str = ""
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
        provider: Optional[Any] = None,
        on_part: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> None:
        #: Called ``(done, total)`` as each part of a LONG transcript is summarised, so a host can show
        #: "part 3 of 7" instead of a spinner for the several minutes a long meeting takes.
        self.on_part = on_part
        #: Polled between parts; true raises :class:`~meeting_notes.cancel.PipelineCancelled`.
        self.should_cancel = should_cancel
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self._resolved_model: Optional[str] = None
        #: A language model bundled with the engine (``llm.BundledProvider``: ``.name`` and
        #: ``.generate(prompt)``). Tried BEFORE Ollama; ``None`` means Ollama alone, as before. The
        #: pipeline passes the installed one; a bare ``Summarizer()`` — every test — never reaches for it.
        self.provider = provider

    def summarize(
        self,
        transcript: str,
        language: str = "en",
        template: str = "meeting",
        speaker_text: Optional[str] = None,
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

        # What the language model reads: the transcript with WHO SPOKE on each line when the caller has
        # it. The extractive fallback keeps the plain text — "Person 1:" prefixes would end up in the
        # sentences it picks.
        llm_text = speaker_text or transcript

        # Order: the bundled model, then Ollama, then the extractive fallback. A model that is merely
        # unreachable (not installed, would not start) falls through to the next; the notes record which
        # one actually wrote them, because the difference in quality is large.
        if self.provider is not None:
            try:
                notes = self._summarize_ollama(llm_text, language, template)
                notes.generator = self.provider.name
                return notes
            except Exception:  # noqa: BLE001
                logger.warning("the bundled language model failed; falling back", exc_info=True)

        if self.use_ollama:
            provider, self.provider = self.provider, None   # this attempt is Ollama's, not the bundled model's
            try:
                notes = self._summarize_ollama(llm_text, language, template)
                notes.generator = f"ollama:{self._resolved_model or self.ollama_model}"
                return notes
            except Exception:  # noqa: BLE001
                pass  # Fall through to extractive
            finally:
                self.provider = provider

        notes = self._summarize_extractive(transcript, language)
        notes.generator = "extractive"
        return notes

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
        """Call a local Ollama LLM to produce structured notes.

        A transcript that fits the model's context goes in one call. A longer
        one is summarised in parts and the parts merged — see
        :meth:`_summarize_long`.
        """
        if len(transcript) <= SINGLE_PASS_CHARS:
            return self._ollama_notes(
                _build_prompt(transcript, language, template), language
            )
        return self._summarize_long(transcript, language, template)

    def _ollama_notes(self, prompt: str, language: str) -> MeetingNotes:
        """One Ollama round trip, parsed into notes."""
        return _parse_llm_response(self._ollama_generate(prompt), language)

    def _ollama_generate(self, prompt: str) -> str:
        """Send *prompt* to the language model and return the raw response text.

        The bundled model when one is set, else Ollama. (The name is historical: every call to ANY model
        in this class goes through here, which is what lets the long-transcript logic stay one copy.)
        """
        if self.provider is not None:
            return self.provider.generate(prompt)

        import requests  # noqa: PLC0415

        payload = {
            "model": self.resolve_model(),
            "prompt": prompt,
            "stream": False,
            # Without this Ollama uses its own default window (2–4k tokens
            # depending on version) and silently drops the front of a prompt
            # that does not fit. The prompt below is sized to this value.
            "options": {"num_ctx": OLLAMA_NUM_CTX},
        }

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        return response.json().get("response", "")

    def _summarize_long(
        self,
        transcript: str,
        language: str,
        template: str,
    ) -> MeetingNotes:
        """Map-reduce: notes per part of the transcript, then one merged set.

        **Why this exists.** The prompt used to be cut at 4,000 characters, so
        the notes for any meeting longer than a few minutes described only its
        opening — with nothing to say that the rest had been ignored.

        A part whose call fails degrades to the extractive summary of that part
        rather than failing the meeting, because by then earlier parts have
        already cost real time. The exception is the first call failing to
        reach Ollama at all: that means it is not running, and the caller's
        whole-transcript fallback is the faster answer.
        """
        import requests  # noqa: PLC0415

        from meeting_notes.llm import LlmUnavailable  # noqa: PLC0415

        chunks = _chunk_transcript(transcript, CHUNK_CHARS)
        total = len(chunks)
        partials: list[MeetingNotes] = []
        model_answered = False

        for index, chunk in enumerate(chunks, start=1):
            if self.should_cancel and self.should_cancel():
                from meeting_notes.cancel import PipelineCancelled  # noqa: PLC0415

                raise PipelineCancelled()
            prompt = _build_prompt(
                chunk, language, template, part=(index, total)
            )
            try:
                partials.append(self._ollama_notes(prompt, language))
                model_answered = True
            except (requests.RequestException, LlmUnavailable):
                if not model_answered:
                    raise
                logger.warning("Ollama failed on part %d/%d", index, total)
                partials.append(self._summarize_extractive(chunk, language))
            except Exception:  # noqa: BLE001 — unparseable model output
                logger.warning(
                    "Unusable model output for part %d/%d", index, total
                )
                partials.append(self._summarize_extractive(chunk, language))
            finally:
                if self.on_part:
                    self.on_part(index, total)

        return self._reduce(partials, language, template)

    def _reduce(
        self,
        partials: list[MeetingNotes],
        language: str,
        template: str,
    ) -> MeetingNotes:
        """Merge ordered partial notes into one set.

        Many parts can themselves overflow the context, so this merges in
        rounds: each round halves the count at least, so it always ends.
        """
        level = partials
        while len(level) > 1:
            batches = _batch_partials(level, REDUCE_CHARS)
            level = [self._merge(batch, language, template) for batch in batches]
        return level[0]

    def _merge(
        self,
        parts: list[MeetingNotes],
        language: str,
        template: str,
    ) -> MeetingNotes:
        """Ask the model to merge *parts*; fall back to a plain union."""
        if len(parts) == 1:
            return parts[0]
        try:
            return self._ollama_notes(
                _build_reduce_prompt(parts, language, template), language
            )
        except Exception:  # noqa: BLE001
            logger.warning("Merge call failed; combining parts directly")
            return _merge_plain(parts, language)

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
        # Chosen by a benchmark of the bundled model on the Recordings/ fixtures (NOTES-PLAN.md), not by
        # taste. Without these a 4B model (a) named a person as doing something the transcript gave to
        # someone else, (b) wrote Bisaya notes out of garbled Bisaya input, copying the garble into its
        # decisions, and (c) filled gaps in a noisy transcript with plausible decisions. With them it
        # attributes correctly from context, falls back to "Person 2" when it cannot tell, writes English,
        # and invents noticeably less. English is the notes' language because that is how the company
        # writes its records; change the first rule if that stops being true.
        "guidance": (
            "\nRules:\n"
            "- Write the notes in ENGLISH, even when the meeting was in Tagalog, Bisaya or a mix.\n"
            "- Use only what is in the transcript. Do not invent decisions, action items, owners, dates or "
            "names. If a part is garbled or unclear, leave it out rather than guessing.\n"
            "- The transcript marks who is speaking as Person 1, Person 2 and so on. Say who is doing "
            "what only when the transcript makes it clear; otherwise do not name anyone.\n"
        ),
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


#: Context window requested from Ollama, in tokens.
OLLAMA_NUM_CTX = 8192
#: Per-call timeout. A part is a few thousand tokens of prompt on a CPU-only
#: laptop, so the old 120 s was tight even before parts existed.
OLLAMA_TIMEOUT_SECONDS = 300
#: Transcripts up to this many characters go to the model in one call. Sized
#: for OLLAMA_NUM_CTX with room for the instructions and the answer, assuming
#: the worst case of ~2.5 characters per token (Taglish and Bisaya tokenise
#: poorly): 12,000 chars ≈ 4,800 tokens in, ~1,000 out, against 8,192.
SINGLE_PASS_CHARS = 12_000
#: Size of each part of a longer transcript. Smaller than the single-pass
#: limit so a part is a faster, more focused call.
CHUNK_CHARS = 8_000
#: Largest combined size of the notes handed to one merge call.
REDUCE_CHARS = 12_000


def _build_prompt(
    transcript: str,
    language: str,
    template: str = "meeting",
    part: Optional[tuple[int, int]] = None,
) -> str:
    """Build the LLM prompt for structured note generation.

    The caller is responsible for the transcript fitting the context window
    (see :data:`SINGLE_PASS_CHARS`); this never truncates, because a silent
    cut is exactly how notes came to describe only a meeting's opening.

    Args:
        part: ``(index, total)`` when *transcript* is one section of a longer
            meeting, 1-based.
    """
    spec = TEMPLATES.get(template, TEMPLATES["meeting"])
    lang_hint = f" (transcript language code: {language})" if language != "en" else ""
    part_hint = (
        f"\nThis is part {part[0]} of {part[1]} of a longer meeting. "
        "Write notes for this part only.\n"
        if part
        else ""
    )

    return f"""You are {spec["role"]}{lang_hint}.
{part_hint}
Given the following transcript, produce structured notes in JSON format with these fields:
{spec["fields"]}
{spec["guidance"]}
Respond ONLY with valid JSON. No explanations outside the JSON.

{spec["source_label"]}:
{transcript}

JSON:"""


def _build_reduce_prompt(
    parts: list[MeetingNotes],
    language: str,
    template: str = "meeting",
) -> str:
    """Prompt that merges notes written for consecutive parts of one meeting."""
    import json

    spec = TEMPLATES.get(template, TEMPLATES["meeting"])
    lang_hint = f" (transcript language code: {language})" if language != "en" else ""
    rendered = "\n\n".join(
        f"PART {index} of {len(parts)}:\n"
        + json.dumps(_notes_to_dict(part), ensure_ascii=False, indent=1)
        for index, part in enumerate(parts, start=1)
    )

    return f"""You are {spec["role"]}{lang_hint}.

The notes below were written for consecutive parts of ONE meeting, in order. Merge them into a single set of notes for the whole meeting, in JSON format with these fields:
{spec["fields"]}
{spec["guidance"]}
Rules:
- Cover the whole meeting, first part to last. Do not favour the first or the last part.
- If the same topic, decision or action item appears in several parts, list it once.
- Keep every distinct decision and action item, with its owner and deadline when given.
- The title is the topic of the whole meeting.
Respond ONLY with valid JSON. No explanations outside the JSON.

NOTES BY PART:
{rendered}

JSON:"""


def _notes_to_dict(notes: MeetingNotes) -> dict:
    """The fields a merge needs, leaving out empty ones to save context."""
    data = {
        "title": notes.title,
        "summary": notes.summary,
        "key_topics": notes.key_topics,
        "decisions": notes.decisions,
        "action_items": notes.action_items,
        "medical_terms": notes.medical_terms,
    }
    return {key: value for key, value in data.items() if value}


def _batch_partials(
    parts: list[MeetingNotes],
    max_chars: int,
) -> list[list[MeetingNotes]]:
    """Group consecutive notes so each group fits one merge call.

    Every group takes at least two items (when two remain), so each round of
    merging at least halves the count and the loop in ``_reduce`` terminates
    even when a single item is larger than *max_chars*.
    """
    import json

    batches: list[list[MeetingNotes]] = []
    current: list[MeetingNotes] = []
    size = 0
    for part in parts:
        part_size = len(json.dumps(_notes_to_dict(part), ensure_ascii=False))
        if len(current) >= 2 and size + part_size > max_chars:
            batches.append(current)
            current, size = [], 0
        current.append(part)
        size += part_size
    if current:
        batches.append(current)
    return batches


def _dedupe(items: list[str]) -> list[str]:
    """Drop repeats, ignoring case and surrounding whitespace; keep order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = str(item).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _merge_plain(parts: list[MeetingNotes], language: str) -> MeetingNotes:
    """Combine notes without a model: used when a merge call fails."""
    return MeetingNotes(
        title=parts[0].title,
        summary=" ".join(p.summary for p in parts if p.summary),
        key_topics=_dedupe([t for p in parts for t in p.key_topics]),
        decisions=_dedupe([d for p in parts for d in p.decisions]),
        action_items=_dedupe([a for p in parts for a in p.action_items]),
        medical_terms=_dedupe([m for p in parts for m in p.medical_terms]),
        language=language,
    )


def _chunk_transcript(text: str, max_chars: int) -> list[str]:
    """Split *text* into consecutive parts of at most *max_chars* characters.

    Cuts fall on line boundaries when the transcript has lines (a timed
    transcript does), else on sentence boundaries, and only inside a sentence
    as a last resort. Nothing is dropped: the parts joined cover the text.
    """
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            units.append(line)
            continue
        for sentence in _split_sentences(line):
            if len(sentence) <= max_chars:
                units.append(sentence)
            else:
                units.extend(_split_words(sentence, max_chars))

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        if current and size + len(unit) + 1 > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(unit)
        size += len(unit) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_words(text: str, max_chars: int) -> list[str]:
    """Last-resort split of an unpunctuated run, on word boundaries."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for word in text.split():
        # A single "word" longer than the limit (no spaces at all) is cut.
        while len(word) > max_chars:
            if current:
                pieces.append(" ".join(current))
                current, size = [], 0
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        if current and size + len(word) + 1 > max_chars:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += len(word) + 1
    if current:
        pieces.append(" ".join(current))
    return pieces


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


def speaker_text(segments: list) -> Optional[str]:
    """The transcript as ``"Person 1: …"`` lines, consecutive turns by one speaker merged.

    ``None`` when speakers cannot be told apart (diarization off or failed, or only one voice): labels
    would add tokens and no information. Merging turns keeps a long meeting's prompt smaller, and gives
    the long-transcript chunker whole turns to cut between.
    """
    labelled = [(getattr(s, "speaker", None), str(getattr(s, "text", "")).strip()) for s in segments]
    labelled = [(who, text) for who, text in labelled if text]
    if len({who for who, _ in labelled if who}) < 2:
        return None

    lines: list[str] = []
    last: Optional[str] = None
    for who, text in labelled:
        who = who or "Unknown"
        if who == last and lines:
            lines[-1] += " " + text
        else:
            lines.append(f"{who}: {text}")
            last = who
    return "\n".join(lines)


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
