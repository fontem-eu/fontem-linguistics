"""Shared pieces of an OpenAI-compatible chat backend.

Mistral and Nebius speak the same wire protocol — `/chat/completions`, a JSON
response format, a `usage` block priced per million tokens — so the prompt,
the retry policy and the response validation live here once rather than being
copied per provider. What differs is the base URL, the model name and the
price, which is configuration rather than code.

Nothing here knows which provider it serves: the caller passes its own error
types, so a failure surfaces with that provider's name attached.
"""
from __future__ import annotations

import asyncio
import json
import random
import re

import httpx

from src.domain.brand_marks import only_names, protect, restore

#: EU official languages, code -> the name a model will recognise. Spelling
#: out the name matters for the smaller ones: a model given only `mt` has
#: been known to answer in Malay rather than Maltese.
LANG_FULLNAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pl": "Polish", "pt": "Portuguese", "nl": "Dutch",
    "ro": "Romanian", "cs": "Czech", "hu": "Hungarian", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "el": "Greek", "bg": "Bulgarian",
    "hr": "Croatian", "sk": "Slovak", "sl": "Slovene", "lt": "Lithuanian",
    "lv": "Latvian", "et": "Estonian", "ga": "Irish", "mt": "Maltese",
}


def lang_fullname(code: str) -> str:
    """The language's name, or the code itself when we do not carry one."""
    return LANG_FULLNAMES.get(code, code)


#: BCP-47 "undetermined". A caller that cannot tell the source language
#: passes this rather than a guess: telling a model a Norwegian title is
#: English is worse than asking it to look.
UNDETERMINED = "und"


#: Told nothing about names, the model left more than half of French and
#: German authority names untranslated, copied verbatim into every language
#: ("Ville de Dugny" for "City of Dugny"; 2026-10-01, 526 names and titles:
#: 29% of texts untranslated, 1.9% with this sentence).
NAMES_INSTRUCTION = (
    "Keep proper names of places, people, companies, brands and projects as they are "
    "written, but translate every other word, including words that say what kind of "
    "body or thing something is (for example 'Ville de', 'Mairie de', 'Université de'). "
)

#: The sentence for the placeholders protect() puts where brand-like names
#: were: without them the names sentence alone translated coined names
#: ("QUEST" -> "Suche"); asked to keep <marked> names, the model still
#: translated some and left brackets in a fifth of its answers.
PLACEHOLDER_INSTRUCTION = (
    "Placeholders such as {1} stand for names: copy each one unchanged, exactly once, "
    "into every translation. "
)

#: Bulgarian and Greek names are transliterated, and the names sentence
#: alone let that bleed into other languages; this sentence halves it
#: (10 texts of 526 to 5). What still bleeds is caught on parsing.
SCRIPT_INSTRUCTION = (
    "Write Bulgarian in Cyrillic and Greek in Greek letters; write every other language "
    "in Latin letters. "
)


def build_translate_prompt(text: str, source_lang: str, targets: list[str],
                           names: bool = True) -> str:
    """One prompt, one JSON object back, keyed by language code.

    With ``source_lang="und"`` the caller does not know the language, so the
    model is asked for every target and for the ``source_lang`` it found. The
    first wording ("for a target that is the text's own language, return the
    text unchanged") made Gemma return the source for EVERY target: 1,127 of
    1,392 strings on the first prod run. The wording below held copies to
    the source language alone on the same titles.

    Brand-like tokens leave the text as placeholders (protect) that
    parse_translation_response() puts back. ``names=False`` is the plain
    prompt without names guidance or placeholders: the fallback for an
    answer in the wrong script or with a placeholder lost, which got every
    such text right in the 2026-10-01 test.
    """
    protected, placeholders = protect(text) if names else (text, [])
    pretty = ", ".join(f"{code} ({lang_fullname(code)})" for code in targets)
    if source_lang == UNDETERMINED:
        keys = ", ".join(f'"{t}"' for t in ["source_lang", *targets])
        opening = (
            "Translate the following text into every target language below. "
            "Write each value in its own target language; a value may equal the "
            "original text only for the language the text is already written in. "
            'Also give "source_lang": the ISO 639-1 code of the text\'s language. '
        )
    else:
        keys = ", ".join(f'"{t}"' for t in targets)
        opening = (
            "Translate the following text from "
            f"{lang_fullname(source_lang)} into the target languages. "
        )
    guidance = ""
    if names:
        guidance = NAMES_INSTRUCTION + SCRIPT_INSTRUCTION
        if placeholders:
            guidance += PLACEHOLDER_INSTRUCTION
    return (
        opening + guidance
        + "Preserve institutional terminology, do not paraphrase. Return strict "
        f"JSON with keys: {keys}. No prose, no explanation.\n"
        f"Target languages: {pretty}.\n"
        f"Text: {protected}"
    )


#: Letters of the two non-Latin scripts among the EU languages.
_NON_LATIN = re.compile(r"[\u0370-\u03ff\u0400-\u04ff]")
_NON_LATIN_TARGETS = frozenset({"bg", "el"})


# The text and prompt variant are what the answer is checked against.
def parse_translation_response(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    data: dict, targets: list[str], error_cls: type[Exception], source_text: str = "",
    names: bool = True,
) -> tuple[dict[str, str], dict]:
    """Validate a chat completion into ``({lang: text}, usage)``.

    A response missing a target is an error, not a partial result: the caller
    reserved budget for the whole set, and a silent gap would be written to
    the graph as a translated record with languages quietly absent. So is a
    Latin-script language written in Cyrillic or Greek when the original has
    no such letters ("Město Пеннес Мирабо" for Czech): the Bulgarian answer
    bleeding into the others. With ``names`` the brand-like spans of
    ``source_text`` were sent as placeholders: they are put back, and an
    answer that lost or doubled one is an error. Both errors carry a flag
    the backend acts on by asking again with the plain prompt.
    """
    try:
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise error_cls(f"malformed chat response: {exc}") from exc

    missing = [t for t in targets if t not in parsed or not isinstance(parsed[t], str)]
    if missing:
        raise error_cls(f"missing/malformed target(s) in response: {missing}")
    out = {t: parsed[t] for t in targets}
    if names and source_text:
        _, placeholders = protect(source_text)
        restored = {t: restore(v, placeholders) for t, v in out.items()}
        broken = [t for t, v in restored.items() if v is None]
        if broken:
            exc = error_cls(f"placeholder lost or doubled in target(s): {broken}")
            exc.retry_plainly = True     # type: ignore[attr-defined]
            raise exc
        out = {t: v for t, v in restored.items() if v is not None}
    if not _NON_LATIN.search(source_text):
        wrong = [t for t in targets if t not in _NON_LATIN_TARGETS and _NON_LATIN.search(out[t])]
        if wrong:
            exc = error_cls(f"Cyrillic or Greek letters in Latin-script target(s): {wrong}")
            # Declared on NebiusError and MistralError; the caller asks again.
            exc.retry_plainly = True     # type: ignore[attr-defined]
            raise exc
    return out, (data.get("usage") or {})


def untranslatable(text: str) -> bool:
    """A text that is nothing but brand-like names: its own translation."""
    return only_names(text)


#: What a detection may answer: a two-letter ISO 639-1 code, or "und".
_LANG_CODE = re.compile(r"^(?:[a-z]{2}|und)$")


def build_detect_prompt(texts: list[str]) -> str:
    """Ask for each text's language, keyed by its position.

    Keys rather than a list so an answer cannot slide one place along: a
    skipped item leaves its key missing instead of shifting every later
    code onto the wrong text. Judged from the words, because a Swedish
    buyer's English title is English. The EU's languages are named as the
    expected set: without them Gemma called Croatian and Slovene titles
    Serbian, which no EU notice is written in.
    """
    listing = json.dumps({str(i): t for i, t in enumerate(texts)}, ensure_ascii=False)
    expected = ", ".join(f"{code} ({name})" for code, name in LANG_FULLNAMES.items())
    return (
        "Identify the language each text below is written in, judged from its "
        "ordinary words, not from any country, city or organisation it names. The "
        f"texts are public procurement titles, almost always in one of: {expected}. "
        "Answer another ISO 639-1 code only when a text is plainly in none of these "
        '(for example Norwegian, "no"). Answer "und" only when a text has no '
        "ordinary words at all: nothing but numbers, reference codes or names. "
        "Return strict JSON with the same keys as the input, each mapped to a "
        "two-letter lowercase code or \"und\". No prose, no explanation.\n"
        f"Texts: {listing}"
    )


def parse_detect_response(
    data: dict, n_texts: int, error_cls: type[Exception],
) -> tuple[list[str | None], dict]:
    """``([code or None per text], usage)``.

    An unreadable completion is an error for the whole call; one text left
    unanswered, or answered with something that is not a language code, is
    None for that text alone. Its neighbours' answers stand on their own keys.
    """
    try:
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise error_cls(f"malformed chat response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise error_cls("malformed chat response: not a JSON object")
    codes: list[str | None] = []
    for i in range(n_texts):
        value = parsed.get(str(i))
        code = value.strip().lower() if isinstance(value, str) else ""
        codes.append(code if _LANG_CODE.match(code) else None)
    return codes, (data.get("usage") or {})


async def post_with_retries(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    client: httpx.AsyncClient,
    path: str,
    payload: dict,
    max_retries: int,
    transient_cls: type[Exception],
    error_cls: type[Exception],
) -> dict:
    """POST with jittered backoff on 5xx/429/network; 4xx raises immediately.

    A 4xx that is not 429 is the caller's fault — a bad model name, a revoked
    key — and retrying it only burns the deadline.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(path, json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = transient_cls(str(exc))
        else:
            if resp.status_code >= 500 or resp.status_code == 429:
                last_exc = transient_cls(f"status={resp.status_code} body={resp.text[:200]}")
            elif resp.status_code >= 400:
                raise error_cls(f"status={resp.status_code} body={resp.text[:200]}")
            else:
                return resp.json()
        if attempt < max_retries:
            await asyncio.sleep((2 ** attempt) * 0.5 + random.uniform(0, 0.25))
    raise last_exc if last_exc else error_cls("request failed with no exception recorded")


def price_usd(usage: dict, price_input_per_mtok: float, price_output_per_mtok: float) -> float:
    """What the provider's own usage block says the call cost.

    Priced from reported tokens rather than an estimate, so a wrong guess
    about prompt size cannot quietly overshoot a budget.
    """
    tokens_in = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)
    return (tokens_in * price_input_per_mtok
            + tokens_out * price_output_per_mtok) / 1_000_000
