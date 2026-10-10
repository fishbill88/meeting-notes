"""Text normalisation for scoring.

A WER figure means nothing without saying how the text was normalised — the
same system can score 9% or 19% on identical audio depending on whether you
fold case and strip punctuation. Every number this harness reports is produced
by :func:`normalize_text` with the settings recorded alongside it.

Deliberately conservative: fold case, strip punctuation, collapse whitespace.
No stemming, no stopword removal, no spelling correction — those flatter the
system under test rather than measuring it.

Tagalog-aware in one respect: the hyphen in affixed forms (``nag-meeting``,
``i-refer``, ``pa-check``) is *not* treated as punctuation, because the hybrid
form is one token. Splitting it would silently score ``nag meeting`` as
correct, hiding exactly the failure mode Taglish support has to fix.
"""

from __future__ import annotations

import re
import unicodedata

#: Punctuation stripped before scoring. Hyphen and apostrophe are absent on
#: purpose — both are word-internal in Tagalog orthography.
_PUNCTUATION = r"""!"#$%&()*+,./:;<=>?@[\]^_`{|}~«»“”„‘’…–—"""

_PUNCT_RE = re.compile(f"[{re.escape(_PUNCTUATION)}]")
_WHITESPACE_RE = re.compile(r"\s+")
_EDGE_HYPHEN_RE = re.compile(r"(^-+)|(-+$)")


def normalize_text(
    text: str,
    lowercase: bool = True,
    strip_punctuation: bool = True,
    unicode_form: str = "NFKC",
) -> str:
    """Normalise ``text`` for comparison against a reference.

    Args:
        text: Raw transcript or reference.
        lowercase: Fold case.
        strip_punctuation: Remove punctuation other than word-internal
            hyphens and apostrophes.
        unicode_form: Unicode normalisation form. NFKC folds the typographic
            variants that differ between dataset sources.

    Returns:
        Normalised text, whitespace-collapsed and trimmed.
    """
    if not text:
        return ""

    text = unicodedata.normalize(unicode_form, text)

    if lowercase:
        text = text.lower()

    if strip_punctuation:
        text = _PUNCT_RE.sub(" ", text)

    text = _WHITESPACE_RE.sub(" ", text).strip()

    # A hyphen only counts as word-internal; drop dangling ones left behind by
    # punctuation stripping (e.g. an em-dash rendered as "-").
    tokens = [_EDGE_HYPHEN_RE.sub("", tok) for tok in text.split(" ")]
    return " ".join(tok for tok in tokens if tok)


def tokenize(text: str) -> list[str]:
    """Split normalised text into scoring tokens."""
    normalised = normalize_text(text)
    return normalised.split(" ") if normalised else []
