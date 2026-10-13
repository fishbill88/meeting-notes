"""Enhance: expand a person's OWN notes from the transcript, without ever rewriting them.

The person types short lines while a meeting runs ("pricing?", "Ana - send the build"). Afterwards the
transcript is available; Enhance puts detail under each of those lines from what was actually said near
the moment they typed it, and adds what they did not write down (the decisions and action items the notes
step already found).

**The invariant is built, not requested.** Every line the person wrote comes out in their order and in their
exact words, because the output tree is assembled here from THEIR lines; the model is only asked for detail
to hang beneath each one, and what it returns is attached by line number. A model that drops, reorders or
rewords a line cannot change the structure — there is nothing for it to drop. That matters because the
person's trust in the feature is "my notes stay mine".

**Every added line must be checkable.** Detail comes back with the transcript time it came from; it is kept
only if that time is real and the words overlap what was said near it. A model that invents is thereby
mostly discarded rather than trusted, and each kept line carries ``source_t`` so the viewer can jump to it.

The tree (``enhanced["items"]``) is ``{kind: "mine" | "added" | "section", text, source_t?, t?, children}``.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from meeting_notes.cancel import PipelineCancelled

logger = logging.getLogger("meeting_notes.enhance")

#: Transcript within this many seconds of a typed line is "near" it. Wide enough to catch a discussion the
#: person noted as it wrapped up; narrow enough that "pricing?" at 12:40 is not answered from minute 50.
WINDOW_SECONDS = 180.0

#: A whole transcript up to this many characters goes to the model in one call; longer ones go in batches
#: of lines with just the transcript near them.
ONE_CALL_CHARS = 10_000
BATCH_CHARS = 7_000

MAX_LINES = 400
MAX_LINE_CHARS = 2_000
MAX_DETAILS_PER_LINE = 3

#: Of a detail's content words, at least this share (and at least two) must appear in the transcript it cites.
GROUNDING_SHARE = 0.34

_STOP = frozenset(
    "a an the and or but if so to of in on at for with from by as is are was were be been being it its this "
    "that these those i you he she we they them his her our your their my me do does did not no yes just "
    "ang ng sa na at ay mga ko mo ka ni si kay para pero kung yung nung lang naman din rin po".split()
)


@dataclass
class Turn:
    """One speaker's consecutive speech, with where it started."""

    start: float
    end: float
    speaker: Optional[str]
    text: str

    def label(self) -> str:
        return f"[{clock(self.start)}] {self.speaker + ': ' if self.speaker else ''}{self.text}"


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def parse_clock(text: str) -> Optional[float]:
    """``"12:40"`` or ``"1:02:03"`` (optionally in brackets) to seconds; ``None`` for anything else."""
    match = re.fullmatch(r"\s*\[?\s*(?:(\d{1,2}):)?(\d{1,3}):(\d{2})\s*\]?\s*", str(text or ""))
    if not match:
        return None
    hours, minutes, secs = (int(match.group(1) or 0), int(match.group(2)), int(match.group(3)))
    if secs > 59:
        return None
    return float(hours * 3600 + minutes * 60 + secs)


def turns_from_transcript(transcript: dict) -> list[Turn]:
    """Speaker turns from a stored transcript (segment times are centiseconds), consecutive same-speaker
    segments merged. A transcript with no segments becomes one untimed turn so Enhance still has text."""
    segments = transcript.get("segments") or []
    turns: list[Turn] = []
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start, end = float(seg.get("t0") or 0) / 100.0, float(seg.get("t1") or 0) / 100.0
        speaker = seg.get("speaker") or None
        if turns and turns[-1].speaker == speaker:
            turns[-1].text += " " + text
            turns[-1].end = end
        else:
            turns.append(Turn(start=start, end=end, speaker=speaker, text=text))
    if not turns and str(transcript.get("text") or "").strip():
        turns.append(Turn(start=0.0, end=0.0, speaker=None, text=str(transcript["text"]).strip()))
    return turns


def clean_lines(lines: Any) -> list[dict]:
    """The person's lines, validated: ``[{"t": seconds | None, "text": str}]``, blank lines dropped.

    Blank lines are spacing, not content, so they are the one thing the "every line comes back" rule does
    not cover. Text is otherwise kept byte for byte (indentation and bullets are the person's structure).
    """
    out: list[dict] = []
    for item in lines or []:
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").rstrip()
        if not text.strip():
            continue
        t = item.get("t")
        try:
            t = None if t is None else max(0.0, float(t))
        except (TypeError, ValueError):
            t = None
        out.append({"t": t, "text": text[:MAX_LINE_CHARS]})
        if len(out) >= MAX_LINES:
            break
    return out


# ------------------------------------------------------------------ matching words

def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[^\W_]{3,}", text.lower()) if w not in _STOP}


def _overlap(detail: str, evidence: str) -> tuple[int, float]:
    """How many of the detail's content words appear in the evidence, and what share of them."""
    mine = _words(detail)
    if not mine:
        return 0, 0.0
    hits = len(mine & _words(evidence))
    return hits, hits / len(mine)


def _near(turns: list[Turn], t: Optional[float], window: float = WINDOW_SECONDS) -> list[int]:
    """Indexes of the turns within the window of time ``t``; every turn when ``t`` is unknown."""
    if t is None:
        return list(range(len(turns)))
    found = [i for i, turn in enumerate(turns) if turn.end >= t - window and turn.start <= t + window]
    return found or list(range(len(turns)))


def _grounded(detail: str, turns: list[Turn], at: Optional[float]) -> Optional[float]:
    """The start time of the turn that backs ``detail`` up, or ``None`` when nothing does.

    ``at`` is where the model said it came from. A turn near it that supports the detail wins (the model
    saw the transcript, so its citation is the best evidence of which turn it meant); failing that, a turn
    within a minute of it. Beyond that the citation was a guess and the detail is dropped. A turn's
    neighbours count as context for the words, but the turn's OWN words break ties.
    """
    for tolerance in ((20.0, 60.0) if at is not None else (None,)):
        best: Optional[tuple[int, int, float, float]] = None   # turn's own hits, window hits, -distance, start
        for i, turn in enumerate(turns):
            if tolerance is not None and not (turn.start - tolerance <= at <= turn.end + tolerance):
                continue
            evidence = " ".join(t.text for t in turns[max(0, i - 1):i + 2])
            hits, share = _overlap(detail, evidence)
            if hits < 2 or share < GROUNDING_SHARE:
                continue
            own, _ = _overlap(detail, turn.text)
            score = (own, hits, -abs(turn.start - at) if at is not None else 0.0, turn.start)
            if best is None or score[:3] > best[:3]:
                best = score
        if best is not None:
            return best[3]
    return None


# ------------------------------------------------------------------ the model's part

def build_prompt(lines: list[tuple[int, str, Optional[float]]], turns: list[Turn], language: str) -> str:
    transcript = "\n".join(turn.label() for turn in turns)
    notes = "\n".join(
        f"{n}. {text.strip()}" + (f"   (typed at {clock(t)})" if t is not None else "")
        for n, text, t in lines
    )
    return (
        "Below are one person's own short notes from a meeting, and the transcript of that meeting. "
        "For each numbered note, add up to "
        f"{MAX_DETAILS_PER_LINE} short detail bullets that the TRANSCRIPT says about it.\n\n"
        "Rules:\n"
        "- Use only what is in the transcript. If it says nothing about a note, return an empty list for it.\n"
        "- Each bullet is one plain sentence, written in English, and does not repeat the note.\n"
        "- Each bullet has \"at\": the [mm:ss] time of the transcript line it comes from, copied exactly.\n"
        "- Where a note has a typed-at time, prefer what was said near that time.\n"
        "- Leave out anything garbled or unclear. Name a person only when the transcript makes it clear, "
        "and never say he or she: use the name, \"Person 2\", or \"they\".\n"
        "- Do not say what happened afterwards or how things turned out unless the transcript says so.\n"
        "- Do not add notes of your own and do not change the numbering.\n\n"
        "Answer with JSON only, in this shape: "
        "{\"lines\": [{\"n\": 1, \"details\": [{\"text\": \"...\", \"at\": \"12:40\"}]}]}\n\n"
        f"NOTES:\n{notes}\n\nTRANSCRIPT:\n{transcript}\n"
    )


def parse_response(raw: str) -> dict[int, list[dict]]:
    """``{line number: [{"text", "at"}]}`` from the model's JSON; tolerant of fences and stray prose."""
    import json

    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw or "")
    body = match.group(1) if match else (raw or "").strip()
    brace = re.search(r"\{[\s\S]*\}", body)
    data = json.loads(brace.group(0) if brace else body)
    out: dict[int, list[dict]] = {}
    for item in data.get("lines") or []:
        try:
            n = int(item.get("n"))
        except (TypeError, ValueError, AttributeError):
            continue
        details = []
        for d in item.get("details") or []:
            if isinstance(d, str):
                d = {"text": d}
            if isinstance(d, dict) and str(d.get("text") or "").strip():
                details.append({"text": str(d["text"]).strip(), "at": d.get("at")})
        out.setdefault(n, []).extend(details)
    return out


def _batches(
    lines: list[tuple[int, str, Optional[float]]], turns: list[Turn]
) -> list[tuple[list[tuple[int, str, Optional[float]]], list[Turn]]]:
    """Lines grouped so each group's transcript excerpt fits one call. Whole transcript when it is short."""
    if sum(len(t.text) + 16 for t in turns) <= ONE_CALL_CHARS:
        return [(lines, turns)]

    groups: list[tuple[list[tuple[int, str, Optional[float]]], set[int]]] = []
    for line in lines:
        wanted = set(_near(turns, line[2])) if line[2] is not None else _best_turns(line[1], turns)
        if groups:
            merged = groups[-1][1] | wanted
            if sum(len(turns[i].text) + 16 for i in merged) <= BATCH_CHARS:
                groups[-1][0].append(line)
                groups[-1][1].update(wanted)
                continue
        groups.append(([line], set(wanted)))

    result = []
    for group_lines, indexes in groups:
        ordered = sorted(indexes)
        # A single line's window can itself exceed the budget; keep the turns nearest its time.
        while sum(len(turns[i].text) + 16 for i in ordered) > BATCH_CHARS and len(ordered) > 1:
            anchor = group_lines[0][2] or 0.0
            ordered.remove(max(ordered, key=lambda i: abs(turns[i].start - anchor)))
        result.append((group_lines, [turns[i] for i in ordered]))
    return result


def _best_turns(text: str, turns: list[Turn], count: int = 12) -> set[int]:
    """For an untimed line, the turns that share the most words with it."""
    scored = sorted(range(len(turns)), key=lambda i: -_overlap(text, turns[i].text)[0])
    chosen = {i for i in scored[:count] if _overlap(text, turns[i].text)[0] > 0}
    return chosen or set(scored[:count])


# ------------------------------------------------------------------ the whole thing

def enhance(
    lines: list[dict],
    transcript: dict,
    notes: Optional[dict] = None,
    generate: Optional[Callable[[str], str]] = None,
    generator: str = "extractive",
    on_progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Build the enhanced tree for ``lines`` (see :func:`clean_lines`) against ``transcript``.

    ``generate`` is the model (``prompt -> raw JSON text``); ``None`` or a failing model means the
    extractive path, which still attaches the closest spoken sentences with their times. Raises
    :class:`~meeting_notes.cancel.PipelineCancelled` between calls when ``should_cancel`` says so.
    """
    lines = clean_lines(lines)
    turns = turns_from_transcript(transcript)
    numbered = [(i + 1, line["text"], line["t"]) for i, line in enumerate(lines)]

    details: dict[int, list[dict]] = {}
    used = generator if generate else "extractive"
    if numbered and turns:
        if generate:
            try:
                details = _ask(generate, numbered, turns, transcript, on_progress, should_cancel)
            except PipelineCancelled:
                raise
            except Exception:  # noqa: BLE001 — a model that fails must not lose the person's notes
                logger.warning("enhance: the language model failed; using the extractive path", exc_info=True)
                used = "extractive"
                details = {}
        if not generate or used == "extractive":
            details = _extractive(numbered, turns)

    items: list[dict] = []
    for n, text, t in numbered:
        item: dict = {"kind": "mine", "text": text, "children": [
            {"kind": "added", "text": d["text"], "source_t": d["source_t"], "children": []}
            for d in details.get(n, [])
        ]}
        if t is not None:
            item["t"] = t
        items.append(item)

    notes = notes or {}
    for heading, key in (("Decisions", "decisions"), ("Action items", "action_items")):
        extra = [str(x).strip() for x in (notes.get(key) or []) if str(x).strip()]
        if extra:
            items.append({"kind": "section", "text": heading, "children": [
                {"kind": "added", "text": x, "source_t": None, "children": []} for x in extra
            ]})

    return {"items": items, "generator": used, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())}


def _ask(generate, numbered, turns, transcript, on_progress, should_cancel) -> dict[int, list[dict]]:
    language = str(transcript.get("language") or "en")
    batches = _batches(numbered, turns)
    out: dict[int, list[dict]] = {}
    for index, (group, excerpt) in enumerate(batches):
        if should_cancel and should_cancel():
            raise PipelineCancelled()
        if on_progress:
            on_progress(index / len(batches), f"Adding detail to your notes, part {index + 1} of {len(batches)}…"
                        if len(batches) > 1 else "Adding detail to your notes…")
        parsed = parse_response(generate(build_prompt(group, excerpt, language)))
        for n, text, t in group:
            kept: list[dict] = []
            for d in parsed.get(n, [])[:MAX_DETAILS_PER_LINE]:
                at = parse_clock(d.get("at"))
                source = _grounded(d["text"], excerpt, at)
                if source is not None:
                    kept.append({"text": d["text"], "source_t": source})
            out[n] = kept
    if on_progress:
        on_progress(1.0, "Adding detail to your notes…")
    return out


def _extractive(numbered, turns) -> dict[int, list[dict]]:
    """No model: for each line, the nearest spoken sentences that share its words, with their times."""
    out: dict[int, list[dict]] = {}
    for n, text, t in numbered:
        scored: list[tuple[int, float, str, float]] = []   # hits, closeness, sentence, turn start
        for i in _near(turns, t):
            turn = turns[i]
            for sentence in re.split(r"(?<=[.!?])\s+", turn.text):
                hits, _share = _overlap(text, sentence)
                if hits >= 1 and len(sentence.split()) >= 4:
                    closeness = -abs(turn.start - t) if t is not None else 0.0
                    scored.append((hits, closeness, sentence.strip(), turn.start))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        out[n] = [{"text": sentence, "source_t": start} for _h, _c, sentence, start in scored[:2]]
    return out


def flatten_text(enhanced: dict) -> str:
    """All the words of an enhanced tree, for search."""
    parts: list[str] = []

    def walk(items):
        for item in items or []:
            parts.append(str(item.get("text") or ""))
            walk(item.get("children"))

    walk((enhanced or {}).get("items"))
    return "\n".join(parts)


def to_markdown(enhanced: dict) -> str:
    """The tree as Markdown: the person's lines as written, added detail as indented bullets with times."""
    out: list[str] = []
    for item in (enhanced or {}).get("items") or []:
        if item.get("kind") == "section":
            out += ["", f"## {item['text']}"]
            out += [f"- {c['text']}" for c in item.get("children") or []]
            continue
        out.append(item["text"])
        for child in item.get("children") or []:
            when = f" ({clock(child['source_t'])})" if child.get("source_t") is not None else ""
            out.append(f"  - {child['text']}{when}")
    return "\n".join(out).strip() + "\n"
