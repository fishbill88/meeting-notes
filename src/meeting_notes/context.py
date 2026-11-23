"""What the host knows about a recording that the speech model cannot hear: the meeting's title, the people in it,
and the team's own words. It is given to faster-whisper as ``hotwords``, so a name comes out spelled the way its
owner spells it rather than the way it sounds ("Marco Reyes", not "Mark O'Rayes").

**Why ``hotwords`` and not ``initial_prompt``** (faster-whisper takes both). ``initial_prompt`` only seeds the
running "previous text", which slides out of the decoder's view after about 223 tokens, i.e. roughly the first
minute of speech. ``hotwords`` is put in front of EVERY 30-second window's prompt, so a name said 40 minutes in is
still helped. Read from faster-whisper 1.2.1's ``WhisperModel.get_prompt``.

**Why it is kept short.** The hint and the previous text share the decoder's 448-token window with what it writes,
and a long hint is what Whisper copies into the transcript when the audio goes quiet or turns to noise. Hence a
character budget (:data:`MAX_HINT_CHARS`) and :func:`is_echo`, which finds a segment that is only the hint read
back.

**Only with the speech filter on** (``backends/whisper.py``). Measured on the Taglish fixture: with a hint in
every window, Whisper decoded the silence after the last word as "Thank you very much.", timed past the end of the
file. faster-whisper's speech filter keeps silence away from the decoder, and with it the same run invented
nothing. A filter that cannot load means no hint.

**Only Whisper uses it.** The Tagalog engine's hotwords need its transducer export (the default export is CTC,
which has no biasing) and write a file beside the shared model, which would leave people's names on disk after the
job; Omnilingual is CTC too. Tagalog and Bisaya recordings that fall back to Whisper do get the hint.

**Privacy.** The names are personal data. They live in the job's config in memory only: never logged, never saved
with the meeting, never echoed by ``Job.public``.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Optional

#: One title, name or term, after cleaning. A real name or a term is far shorter; this stops a pasted paragraph.
MAX_ITEM_CHARS = 60
#: More names than a call has people would bias the decoder towards names nobody said. The host sends none for a
#: bigger group; this is the backstop.
MAX_NAMES = 25
MAX_TERMS = 40
#: The whole hint. About 60-100 tokens: room for a title and a dozen names, and far below the 223 tokens
#: faster-whisper would allow, which would leave the decoder little room for the previous text.
MAX_HINT_CHARS = 300
#: A segment this many words long that reads as a run of the hint is an echo, not speech (see drop_echoes).
ECHO_WORDS = 6


@dataclass(frozen=True)
class TranscriptionContext:
    """The cleaned context of one recording. Build it with :func:`parse_context`."""

    title: Optional[str] = None
    names: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not (self.title or self.names or self.terms)


def _clean(value: object) -> Optional[str]:
    """One item: control, format and pictographic characters out (none of them can be spoken), whitespace
    collapsed, cut to :data:`MAX_ITEM_CHARS`. None for anything that is not a string or is empty afterwards."""
    if not isinstance(value, str):
        return None
    kept = "".join(
        " " if ch.isspace() else ch
        for ch in value
        if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Co", "Cn", "So")
    )
    text = " ".join(kept.split())
    if len(text) > MAX_ITEM_CHARS:
        text = text[:MAX_ITEM_CHARS].rstrip()
    return text or None


def _clean_list(value: object, limit: int) -> tuple[str, ...]:
    """A list of items, cleaned, first spelling kept on a repeat (ignoring case), at most ``limit``. Anything that
    is not a list is no items; an item that is not a string is skipped."""
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean(item)
        if text is None or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        out.append(text)
        if len(out) == limit:
            break
    return tuple(out)


def parse_context(raw: Optional[str]) -> Optional[TranscriptionContext]:
    """The ``context`` form field: a JSON object ``{"title": str, "names": [str], "terms": [str]}``, every key
    optional. None when the field is absent or says nothing.

    Raises:
        ValueError: The field is not a JSON object. A host that sends something malformed has a bug worth
            hearing about; items of the wrong type inside a good object are only skipped.
    """
    if raw is None or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError("context must be a JSON object") from exc
    if not isinstance(data, dict):
        raise ValueError("context must be a JSON object")
    context = TranscriptionContext(
        title=_clean(data.get("title")),
        names=_clean_list(data.get("names"), MAX_NAMES),
        terms=_clean_list(data.get("terms"), MAX_TERMS),
    )
    return None if context.is_empty() else context


def hint_text(context: Optional[TranscriptionContext], budget: int = MAX_HINT_CHARS) -> Optional[str]:
    """The ``hotwords`` string: ``"Sprint sync. Ana Cruz, Marco Reyes, UAT."``

    The title first, then the names, then the terms, each kept only while the whole fits ``budget``, so what is
    cut is the least important. A name or term that repeats the title (a direct call is titled by the other
    person's name) or an earlier item is left out. None when there is nothing to say.
    """
    if context is None:
        return None
    head = context.title or ""
    seen = {head.casefold()} if head else set()
    items: list[str] = []
    for item in (*context.names, *context.terms):
        key = item.casefold()
        if key in seen:
            continue
        if len(_render(head, [*items, item])) > budget:
            break
        seen.add(key)
        items.append(item)
    return _render(head, items) or None


def _render(head: str, items: list[str]) -> str:
    parts = []
    if head:
        parts.append(head.rstrip(".") + ".")
    if items:
        parts.append(", ".join(items) + ".")
    return " ".join(parts)


def _words(text: str) -> list[str]:
    """Lower-case words, punctuation ignored, any script."""
    return "".join(ch if ch.isalnum() else " " for ch in text.casefold()).split()


def _is_run_of(needle: list[str], haystack: list[str]) -> bool:
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def is_echo(text: str, hint: Optional[str]) -> bool:
    """Whether a decoded segment is the hint read back rather than speech: its words are a contiguous run of the
    hint's own words, at least :data:`ECHO_WORDS` long (or the whole hint, if the hint is shorter).

    A person saying one name, or "Sprint sync, Ana", is far below that. Reading six attendees aloud in exactly the
    order the host listed them is the one real utterance this would drop, and it costs one line.
    """
    if not hint:
        return False
    hint_words = _words(hint)
    if len(hint_words) < 3:
        return False
    words = _words(text)
    return len(words) >= min(ECHO_WORDS, len(hint_words)) and _is_run_of(words, hint_words)
