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


def build_translate_prompt(text: str, source_lang: str, targets: list[str]) -> str:
    """One prompt, one JSON object back, keyed by language code.

    With ``source_lang="und"`` the caller does not know the language, so the
    model is asked for every target and for the ``source_lang`` it found. The
    first wording ("for a target that is the text's own language, return the
    text unchanged") made Gemma return the source for EVERY target: 1,127 of
    1,392 strings on the first prod run. The wording below held copies to
    the source language alone on the same titles.
    """
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
    return (
        opening +
        "Preserve institutional terminology, do not paraphrase. Return strict "
        f"JSON with keys: {keys}. No prose, no explanation.\n"
        f"Target languages: {pretty}.\n"
        f"Text: {text}"
    )


def parse_translation_response(
    data: dict, targets: list[str], error_cls: type[Exception],
) -> tuple[dict[str, str], dict]:
    """Validate a chat completion into ``({lang: text}, usage)``.

    A response missing a target is an error, not a partial result: the caller
    reserved budget for the whole set, and a silent gap would be written to
    the graph as a translated record with languages quietly absent.
    """
    try:
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise error_cls(f"malformed chat response: {exc}") from exc

    missing = [t for t in targets if t not in parsed or not isinstance(parsed[t], str)]
    if missing:
        raise error_cls(f"missing/malformed target(s) in response: {missing}")
    return {t: parsed[t] for t in targets}, (data.get("usage") or {})


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
