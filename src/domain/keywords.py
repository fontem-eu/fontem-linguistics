"""Keyword extraction: tokenizer + stop-word removal for the 24 EU languages.

First stage of the query-analysis pipeline behind ``POST /keywords`` — search
backends call it to reduce a free-text query to its content-bearing keywords
before matching titles. Stop-word lists come from ``stopwordsiso`` (ISO-639-1
keyed); Maltese has no published list, so ``mt`` text passes through
unfiltered rather than failing.

Three behaviours here are load-bearing for search and covered by tests:

* Digit-bearing tokens are never removed — legal identifiers ("2024", "1385",
  "10") must survive even though some stop-word lists contain bare numbers.
* With no explicit language we detect by stop-word hit count; zero hits means
  there is nothing to remove and the tokens pass through unchanged. We never
  subtract the union of all 24 lists — a content word in one language is a
  stop word in another ("die" is a German stop word, an English content word).
* Removal never empties the result: if every token is a stop word the
  original tokens are returned ("the who" must not become a search for
  nothing).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import stopwordsiso

from src.domain.languages import EU_OFFICIAL_LANGS

# Word characters minus underscore; keeps digits so identifiers survive.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_HAS_DIGIT = re.compile(r"\d")

_STOPWORDS: dict[str, frozenset[str]] = {
    lang: frozenset(stopwordsiso.stopwords(lang))
    for lang in EU_OFFICIAL_LANGS
    if stopwordsiso.has_lang(lang)
}


@dataclass(frozen=True)
class KeywordResult:
    lang: str | None
    tokens: list[str]
    keywords: list[str]
    removed: list[str]


def tokenize(text: str) -> list[str]:
    """Lowercase Unicode word tokens, digits kept, underscore excluded."""
    return _TOKEN_RE.findall(text.lower())


def detect_lang(tokens: list[str]) -> str | None:
    """Language with the most stop-word hits; None when nothing matches.

    Ties break by EU_OFFICIAL_LANGS order, which keeps detection
    deterministic. This is a heuristic sized for short search queries,
    not a general-purpose detector.
    """
    best: str | None = None
    best_hits = 0
    for lang in EU_OFFICIAL_LANGS:
        stops = _STOPWORDS.get(lang)
        if not stops:
            continue
        hits = sum(1 for t in tokens if t in stops)
        if hits > best_hits:
            best, best_hits = lang, hits
    return best


def extract_keywords(text: str, lang: str | None = None) -> KeywordResult:
    """Tokenize ``text`` and strip stop words for ``lang`` (or detected)."""
    tokens = tokenize(text)
    if not tokens:
        return KeywordResult(lang=lang, tokens=[], keywords=[], removed=[])

    resolved = lang.lower() if lang else detect_lang(tokens)
    stops = _STOPWORDS.get(resolved) if resolved else None
    if not stops:
        return KeywordResult(
            lang=resolved, tokens=tokens, keywords=list(tokens), removed=[],
        )

    keywords: list[str] = []
    removed: list[str] = []
    for tok in tokens:
        if tok in stops and not _HAS_DIGIT.search(tok):
            removed.append(tok)
        else:
            keywords.append(tok)
    if not keywords:
        return KeywordResult(
            lang=resolved, tokens=tokens, keywords=list(tokens), removed=[],
        )
    return KeywordResult(
        lang=resolved, tokens=tokens, keywords=keywords, removed=removed,
    )
