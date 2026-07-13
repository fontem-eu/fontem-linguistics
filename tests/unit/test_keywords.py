"""Unit tests for the keyword-extraction domain module."""
from __future__ import annotations

from src.domain.keywords import detect_lang, extract_keywords, tokenize


def test_tokenize_lowercases_and_keeps_digits():
    assert tokenize("Regulation 10/2011 on PLASTIC") == [
        "regulation", "10", "2011", "on", "plastic",
    ]


def test_tokenize_excludes_underscore_and_punctuation():
    assert tokenize("foo_bar, baz!") == ["foo", "bar", "baz"]


def test_english_stopwords_removed():
    r = extract_keywords("the directive on combating violence against women")
    assert r.lang == "en"
    assert r.keywords == ["directive", "combating", "violence", "women"]
    assert "the" in r.removed and "against" in r.removed


def test_french_detected_and_filtered():
    r = extract_keywords("la lutte contre les violences faites aux femmes")
    assert r.lang == "fr"
    assert r.keywords == ["lutte", "violences", "femmes"]


def test_explicit_lang_overrides_detection():
    # "die" is a German stop word but an English content word.
    en = extract_keywords("die hard", lang="en")
    de = extract_keywords("die hard", lang="de")
    assert "die" in en.keywords
    assert "die" not in de.keywords


def test_digit_bearing_tokens_never_removed():
    # The English stop-word list contains bare numbers like "10"; legal
    # identifiers must survive regardless.
    r = extract_keywords("regulation 10/2011 of the commission", lang="en")
    assert "10" in r.keywords and "2011" in r.keywords


def test_all_stopword_query_passes_through():
    r = extract_keywords("the who")
    assert r.keywords == ["the", "who"]
    assert not r.removed


def test_uncovered_language_passes_through():
    # Maltese has no published stopwordsiso list.
    r = extract_keywords("Gvern ta' Malta", lang="mt")
    assert r.keywords == ["gvern", "ta", "malta"]
    assert not r.removed


def test_unknown_lang_code_passes_through():
    r = extract_keywords("hello world", lang="xx")
    assert r.keywords == ["hello", "world"]


def test_no_stopword_hits_means_no_removal():
    r = extract_keywords("zqxwv kjhgf")
    assert r.lang is None
    assert r.keywords == ["zqxwv", "kjhgf"]


def test_detect_lang_empty_tokens():
    assert detect_lang([]) is None


def test_empty_text():
    r = extract_keywords("   ")
    assert not r.tokens
    assert not r.keywords
