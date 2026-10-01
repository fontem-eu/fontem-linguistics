"""Brand-like tokens in a text to translate, wrapped in <...>.

The translation prompt asks the model to translate every word that is not
a proper name. Told only that, it also translates coined names that look
like words ("QUEST" -> "Suche", "save2safe" -> "speichern2sicher"). Marked
spans are names it must leave as they are.

What is marked is only what no language uses as a word: tokens mixing
letters and digits, camel humps (OptiNERG, BASgas), dotted abbreviations
and legal forms (C.N.R., S.p.A., e.V., GmbH), capitalised acronyms that are
not common words (CPAM, but not PHASE), and an all-caps text of one or two
words that are not common words. Word frequencies come from wordfreq,
which covers 21 of the 24 EU languages; a word is common if any of them
uses it at least once per million words.

Measured 2026-10-01 against 250 hand-labelled names and titles: 88% of the
marks are brands, acronyms or legal forms, 1% are words a translator
should translate; zero-shot NER models (GLiNER, GLiNER-X, GLiNER2) marked
whole descriptive titles and did far worse.
"""
from __future__ import annotations

import re
from functools import lru_cache

from wordfreq import zipf_frequency

#: The EU languages wordfreq carries (hr as "sh"; no et, ga, mt).
_WORDFREQ_LANGS = ("bg", "cs", "da", "de", "el", "en", "es", "fi", "fr", "hu", "it", "lt",
                   "lv", "nl", "pl", "pt", "ro", "sh", "sk", "sl", "sv")
#: A word some EU language uses at least once per million words.
_COMMON_ZIPF = 3.0

_TOKEN = re.compile(r"[^\s/,;:()\[\]«»\"“”‘’'–—]+")
_ROMAN = re.compile(r"^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$")
_CAMEL = re.compile(r"[a-z][A-Z]|[A-Z]{2,}[a-z]{2,}")
_DOTTED = re.compile(r"(?:[A-Za-z]{1,3}\.){2,}[A-Za-z]{0,3}\.?")
_ORDINAL = re.compile(r"\d+(?:st|nd|rd|th|e|er|ème)")
_LEGAL_FORMS = frozenset({
    "GmbH", "gGmbH", "mbH", "AG", "SpA", "S.p.A.", "s.r.l.", "S.r.l.", "SRL", "S.A.", "SA",
    "e.V.", "AöR", "UAB", "a.s.", "s.r.o.", "Kft.", "Zrt.", "Oy", "AB", "ApS", "A/S", "BV",
    "NV", "SAS", "SARL",
})


@lru_cache(maxsize=65536)
def is_common_word(word: str) -> bool:
    """Whether some EU language uses ``word`` as an ordinary word."""
    w = word.lower()
    return any(zipf_frequency(w, lang) >= _COMMON_ZIPF for lang in _WORDFREQ_LANGS)


def _brand_like(word: str, upper_text: bool) -> bool:
    letters = [ch for ch in word if ch.isalpha()]
    if not letters or (word.isupper() and _ROMAN.match(word)):
        return False
    alnum = any(ch.isdigit() for ch in word) and len(letters) >= 2 and not _ORDINAL.fullmatch(word)
    dotted = _DOTTED.fullmatch(word) is not None
    legal = word in _LEGAL_FORMS
    acronym = word.isupper() and len(letters) >= 2 and not is_common_word(word)
    if upper_text:
        # All capitals carry no signal: only what no language uses as a word.
        return alnum or dotted or legal or (acronym and len(letters) <= 6)
    return bool(_CAMEL.search(word)) or alnum or dotted or legal or acronym


def brand_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the brand-like tokens, in order."""
    upper_text = text.isupper()
    spans = []
    tokens = list(_TOKEN.finditer(text))
    for m in tokens:
        word = m.group()
        core = word if word in _LEGAL_FORMS or _DOTTED.fullmatch(word) else word.strip(".-")
        if core and _brand_like(core, upper_text):
            start = m.start() + word.index(core)
            spans.append((start, start + len(core)))
    if upper_text and not spans and len(tokens) <= 2:
        words = [m.group() for m in tokens]
        if words and not all(is_common_word(w.strip(".-")) for w in words) or len(words) == 1:
            spans.append((tokens[0].start(), tokens[-1].end()))        # QUEST, ALTER ENERGIES
    return spans


def mark_brands(text: str) -> str:
    """The text with each brand-like token wrapped in <...>; adjacent ones
    (separated by spaces only) share one pair."""
    merged: list[tuple[int, int]] = []
    for start, end in brand_spans(text):
        if merged and not text[merged[-1][1]:start].strip():
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    out, last = [], 0
    for start, end in merged:
        out += [text[last:start], "<", text[start:end], ">"]
        last = end
    out.append(text[last:])
    return "".join(out)
