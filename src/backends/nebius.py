"""Nebius AI Studio chat backend — translation and language identification.

Why it exists: Mistral is priced for a product that occasionally translates a
name, not for translating millions of contract titles into 23 languages.
Nebius serves open models on the same OpenAI-compatible protocol at a
fraction of the token price, which is what makes a bulk backfill affordable.

Translation only on purpose. Embeddings stay on LaBSE (self-hosted, signed
mirror, one vector space the consolidator's dedup rules already trust) — a
second embedding provider would produce vectors nothing may compare against.

Measured 2026-09-23 against `google/gemma-3-27b-it`, one real Polish contract
title into 23 languages: 138 prompt + 830 completion tokens, 12.2s, all 23
present, and the hard languages (Maltese, Irish, Latvian, Estonian) came back
in the right language with the right register.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

import httpx

from src.backends.openai_chat import (
    build_detect_prompt,
    build_summarize_prompt,
    build_tagged_translate_prompt,
    build_translate_prompt,
    clean_summary,
    cut_short,
    fit_summary,
    parse_detect_response,
    parse_tagged_translation_response,
    parse_translation_response,
    post_with_retries,
    price_usd,
    summary_overshoot_prompt,
)
from src.infra.metrics import (
    LLM_SPEND_USD,
    TRANSLATION_SHORT_RETRIES,
    TRANSLATION_TAGGED_RETRIES,
)


class NebiusError(Exception):
    """Non-retriable Nebius failure, or a malformed payload."""


class NebiusTransientError(Exception):
    """5xx, 429 or a network timeout — retriable by the service."""


class NebiusBatchRefused(NebiusError):
    """Nebius would not create a batch job.

    Seen 2026-09-29 as 403 {"detail": "Creating new batch job is temporarily
    unavailable"} while uploads and real-time calls worked: the account can
    lose batch creation on its own. A caller with another way to get the
    work done should take it rather than fail.
    """


#: Batch states in which the provider is still working.
BATCH_WORKING = frozenset({"validating", "in_progress", "finalizing", "cancelling"})


#: Output a single answer may carry: under the 8,192 tokens an answer is cut
#: at, with room for the model running long.
OUTPUT_TOKEN_BUDGET = 6_000


def target_groups(text_chars: int, targets: list[str]) -> list[list[str]]:
    """``targets`` split so each call's answer fits OUTPUT_TOKEN_BUDGET.

    Measured 2026-10-05 on DeepSeek-V4-Flash: an answer runs about 0.37
    tokens per source character per target, Greek and Bulgarian highest,
    plus the tags around it. A 1,000-character lobbying goal into 23
    languages is ~8,700 tokens — past the cut — so it goes as two calls.
    A title stays one call.
    """
    per_target = 12 + 0.45 * text_chars
    size = max(1, int(OUTPUT_TOKEN_BUDGET // per_target))
    return [targets[i:i + size] for i in range(0, len(targets), size)] or [targets]


#: A line that opens a list item: the marker is kept as written, and only
#: what follows it is translated.
_LIST_MARKER = re.compile(r"\s*(?:[-–—•*·▪]|\d{1,3}[.)])\s+")

#: The line breaks of a text, with the whitespace around them.
_LINE_BREAK = re.compile(r"(\s*\n\s*)")

#: A line worth translating has a letter in it ("***" and "2025" do not).
_LETTER = re.compile(r"[^\W\d_]")

#: Lines of one text asked at the same time.
LINES_AT_ONCE = 4


def _lines_of(text: str) -> tuple[list[str], dict[int, tuple[str, str]]]:
    """``text`` split at its line breaks — lines at even indices, the breaks
    between them at odd — and the lines to translate, by index, as (list
    marker kept as written, the rest)."""
    parts = _LINE_BREAK.split(text)
    lines: dict[int, tuple[str, str]] = {}
    for i in range(0, len(parts), 2):
        marker = _LIST_MARKER.match(parts[i])
        prefix = marker.group(0) if marker else ""
        if _LETTER.search(parts[i][len(prefix):]):
            lines[i] = (prefix, parts[i][len(prefix):])
    return parts, lines


@dataclass
class NebiusBackend:  # pylint: disable=too-many-instance-attributes
    """Configuration bag for one hosted chat model; fields map to env vars."""

    api_url: str
    api_key: str
    chat_model: str
    timeout_s: float
    max_retries: int
    price_input_per_mtok: float
    price_output_per_mtok: float
    client: httpx.AsyncClient
    batch_price_input_per_mtok: float = 0.05
    batch_price_output_per_mtok: float = 0.15
    #: Request parameters the model needs beyond the prompt (reasoning_effort).
    chat_extra: dict = field(default_factory=dict)

    @classmethod
    def build(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        cls,
        api_url: str,
        api_key: str,
        chat_model: str,
        timeout_s: float,
        max_retries: int,
        price_input_per_mtok: float,
        price_output_per_mtok: float,
        batch_price_input_per_mtok: float = 0.05,
        batch_price_output_per_mtok: float = 0.15,
        chat_extra: dict | None = None,
    ) -> "NebiusBackend":
        client = httpx.AsyncClient(
            base_url=api_url.rstrip("/"),
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return cls(
            api_url=api_url,
            api_key=api_key,
            chat_model=chat_model,
            timeout_s=timeout_s,
            max_retries=max_retries,
            price_input_per_mtok=price_input_per_mtok,
            price_output_per_mtok=price_output_per_mtok,
            client=client,
            batch_price_input_per_mtok=batch_price_input_per_mtok,
            batch_price_output_per_mtok=batch_price_output_per_mtok,
            chat_extra=dict(chat_extra or {}),
        )

    async def translate(
        self, text: str, source_lang: str, targets: list[str],
    ) -> dict[str, str]:
        translations, _cost = await self.translate_with_cost(text, source_lang, targets)
        return translations

    async def translate_with_cost(
        self, text: str, source_lang: str, targets: list[str],
    ) -> tuple[dict[str, str], float]:
        """Translate, and report what the provider says it charged.

        The cost comes back rather than being swallowed so the spend cap can
        settle on the real number instead of leaving an estimate standing —
        which is the difference between a budget and a guess.

        An answer whose JSON will not parse is asked once more as tagged
        lines, and the cost is both calls'. A text long enough that its 23
        translations would not fit one answer goes out as several calls, a
        share of the targets each (see `target_groups`). A text of several
        lines goes a line at a time (see `_by_line`).
        """
        if "\n" in text.strip():
            return await self._by_line(text, source_lang, targets)
        return await self._one_line(text, source_lang, targets)

    async def _by_line(
        self, text: str, source_lang: str, targets: list[str],
    ) -> tuple[dict[str, str], float]:
        """Each line of ``text`` translated on its own and put back where it
        was, list markers and blank lines as written.

        Asked for a text of several paragraphs at once, DeepSeek-V4-Flash
        often translates the first and stops, in JSON and in tagged lines
        alike: 46 of 518 lobbying goals with line breaks on 2026-10-09, and
        asked again tagged 7 of those 46 still came back short. Sent a line
        at a time, as every title is, with short lines asked again (see
        `ask_short_again`), none of 96 such goals did — the 46 among them —
        and all 2,162 translations kept their line breaks.
        """
        parts, lines = _lines_of(text)
        at_once = asyncio.Semaphore(LINES_AT_ONCE)

        async def one(line: str) -> tuple[dict[str, str], float]:
            async with at_once:
                return await self._one_line(line, source_lang, targets)

        answers = await asyncio.gather(*(one(line) for _prefix, line in lines.values()))
        out = {t: list(parts) for t in targets}
        for i, (translations, _cost) in zip(lines, answers):
            for t in targets:
                out[t][i] = lines[i][0] + translations[t]        # the list marker, as written
        return {t: "".join(p) for t, p in out.items()}, sum(cost for _t, cost in answers)

    async def _one_line(
        self, text: str, source_lang: str, targets: list[str],
    ) -> tuple[dict[str, str], float]:
        groups = target_groups(len(text), targets)
        if len(groups) == 1:
            return await self._translate_once(text, source_lang, targets)
        answers = await asyncio.gather(*(self._translate_once(text, source_lang, g)
                                         for g in groups))
        merged: dict[str, str] = {}
        for translations, _cost in answers:
            merged.update(translations)
        return merged, sum(cost for _t, cost in answers)

    async def _translate_once(
        self, text: str, source_lang: str, targets: list[str],
    ) -> tuple[dict[str, str], float]:
        data = await self._chat(self.translate_payload(text, source_lang, targets))
        try:
            translations, usage = parse_translation_response(data, targets, NebiusError)
        except NebiusError:
            # The JSON did not survive the text — quotes inside a name, as a
            # rule. What that answer cost is spent all the same.
            broken = data.get("usage") or {}
            self._record_chat_spend(broken)
            TRANSLATION_TAGGED_RETRIES.labels(path="realtime").inc()
            translations, cost = await self.translate_tagged_with_cost(text, source_lang, targets)
            return translations, self.actual_chat_usd(broken) + cost
        self._record_chat_spend(usage)
        translations, again = await self.ask_short_again(
            text, source_lang, translations, "realtime")
        return translations, self.actual_chat_usd(usage) + again

    async def ask_short_again(
        self, text: str, source_lang: str, translations: dict[str, str], path: str,
    ) -> tuple[dict[str, str], float]:
        """``translations`` with any that came back cut short (see
        `cut_short`) asked for again as tagged lines, and what asking cost:
        nothing when every one is whole. ``path`` (realtime or batch) is for
        the metric."""
        short = cut_short(text, translations)
        if not short:
            return translations, 0.0
        TRANSLATION_SHORT_RETRIES.labels(path=path).inc()
        again, cost = await self.translate_tagged_with_cost(text, source_lang, short)
        return {**translations, **again}, cost

    async def translate_tagged_with_cost(
        self, text: str, source_lang: str, targets: list[str],
    ) -> tuple[dict[str, str], float]:
        """Translate with one tagged line per target instead of JSON: for a
        text whose JSON answer would not parse (see
        `build_tagged_translate_prompt`)."""
        data = await self._chat(self.translate_payload(text, source_lang, targets, tagged=True))
        usage = data.get("usage") or {}
        self._record_chat_spend(usage)      # spent whether or not the answer reads
        translations, _usage = parse_tagged_translation_response(data, targets, NebiusError)
        return translations, self.actual_chat_usd(usage)

    async def summarize_with_cost(
        self, text: str, lang: str, max_chars: int, about: str | None = None,
    ) -> tuple[str, float]:
        """A summary of ``text`` in ``lang`` of at most ``max_chars``
        characters, and what it cost. One that comes back too long is asked
        to shorten itself once; one still too long is cut at the last
        sentence that fits."""
        data = await self._chat(self._chat_payload(
            build_summarize_prompt(text, lang, max_chars, about), json_mode=False))
        summary, cost = self._summary_of(data)
        if len(summary) > max_chars:
            data = await self._chat(self._chat_payload(
                summary_overshoot_prompt(summary, max_chars), json_mode=False))
            shorter, extra = self._summary_of(data)
            summary, cost = (shorter or summary), cost + extra
        return fit_summary(summary, max_chars), cost

    def _summary_of(self, data: dict) -> tuple[str, float]:
        usage = data.get("usage") or {}
        self._record_chat_spend(usage)
        try:
            summary = clean_summary(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise NebiusError(f"malformed chat response: {exc}") from exc
        if not summary:
            raise NebiusError("empty summary")
        return summary, self.actual_chat_usd(usage)

    def translate_payload(self, text: str, source_lang: str, targets: list[str],
                          tagged: bool = False) -> dict:
        """The chat request for one text: the same whether it is sent now or
        as a line of a batch, so both paths get the same translation. JSON
        unless ``tagged``, which is plain text and so not in JSON mode."""
        if tagged:
            return self._chat_payload(build_tagged_translate_prompt(text, source_lang, targets),
                                      json_mode=False)
        return self._chat_payload(build_translate_prompt(text, source_lang, targets))

    def _chat_payload(self, prompt: str, json_mode: bool = True) -> dict:
        """A chat request for the configured model, with its extra parameters."""
        payload = {
            "model": self.chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            **self.chat_extra,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def _chat(self, payload: dict) -> dict:
        return await post_with_retries(
            self.client, "/chat/completions", payload,
            max_retries=self.max_retries,
            transient_cls=NebiusTransientError, error_cls=NebiusError,
        )

    async def detect_with_cost(self, texts: list[str]) -> tuple[list[str | None], float]:
        """Each text's ISO 639-1 code (None where the model gave none), and
        what the call was charged. One call for the whole list."""
        data = await self._chat(self._chat_payload(build_detect_prompt(texts)))
        codes, usage = parse_detect_response(data, len(texts), NebiusError)
        self._record_chat_spend(usage)
        return codes, self.actual_chat_usd(usage)

    # ── Batch inference ───────────────────────────────────────────

    def batch_line(self, custom_id: str, text: str, source_lang: str,
                   targets: list[str]) -> dict:
        """One request of a batch file, in the OpenAI batch format."""
        return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions",
                "body": self.translate_payload(text, source_lang, targets)}

    async def submit_batch(self, lines: list[dict], metadata: dict[str, str]) -> tuple[str, str]:
        """Upload the requests and start a batch: ``(batch_id, input_file_id)``.

        A refusal to create the batch removes the uploaded file again, so a
        provider that keeps refusing does not collect our files.
        """
        body = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
        resp = await self._send("POST", "/files", files={
            "file": ("requests.jsonl", body.encode("utf-8"), "application/jsonl")},
            data={"purpose": "batch"})
        file_id = resp.json()["id"]
        try:
            resp = await self._send("POST", "/batches", json={
                "input_file_id": file_id, "endpoint": "/v1/chat/completions",
                "completion_window": "24h", "metadata": metadata,
            })
        except NebiusError:
            await self.delete_file(file_id)
            raise
        return resp.json()["id"], file_id

    async def get_batch(self, batch_id: str) -> dict:
        """The batch object: ``status``, ``output_file_id``, ``error_file_id``."""
        return (await self._send("GET", f"/batches/{batch_id}")).json()

    async def file_lines(self, file_id: str) -> list[dict]:
        """A result file, one JSON object per line."""
        text = (await self._send("GET", f"/files/{file_id}/content")).text
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    async def delete_file(self, file_id: str) -> None:
        """Best effort: a file left behind costs storage, not correctness."""
        try:
            await self._send("DELETE", f"/files/{file_id}")
        except (NebiusError, NebiusTransientError):
            pass

    async def _send(self, method: str, path: str, **kwargs) -> httpx.Response:
        """One call to the files/batches API. 5xx, 429 and network failures
        are transient; a 403 on creating a batch is the provider refusing
        batches; any other 4xx is an error."""
        try:
            resp = await self.client.request(method, path, **kwargs)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise NebiusTransientError(f"{method} {path}: {exc}") from exc
        if resp.status_code >= 500 or resp.status_code == 429:
            raise NebiusTransientError(
                f"{method} {path}: status={resp.status_code} body={resp.text[:200]}")
        if resp.status_code == 403 and path == "/batches":
            raise NebiusBatchRefused(f"status=403 body={resp.text[:200]}")
        if resp.status_code >= 400:
            raise NebiusError(f"{method} {path}: status={resp.status_code} body={resp.text[:200]}")
        return resp

    async def aclose(self) -> None:
        await self.client.aclose()

    # ── Spend ─────────────────────────────────────────────────────

    def _record_chat_spend(self, usage: dict) -> None:
        LLM_SPEND_USD.labels(provider="nebius", endpoint="chat").inc(
            self.actual_chat_usd(usage))

    def estimate_chat_usd(self, text_chars: int, n_targets: int) -> float:
        """Pre-call estimate, for the spend cap's reservation.

        Deliberately pessimistic on output: a title translated into 23
        languages produces roughly its own length per target, and reserving
        too little is how a cap gets overshot before `finalize` corrects it.
        1 token ~ 4 characters.
        """
        est_in_tokens = max(40, text_chars // 4 + 80)
        est_out_tokens = max(20, n_targets * text_chars // 4)
        return (
            est_in_tokens * self.price_input_per_mtok
            + est_out_tokens * self.price_output_per_mtok
        ) / 1_000_000

    def estimate_detect_usd(self, text_chars: int, n_texts: int) -> float:
        """Pre-call reservation for a detection: the texts plus the
        instruction in, one short key-and-code pair per text out."""
        est_in_tokens = text_chars // 4 + 120 + 4 * n_texts
        est_out_tokens = 10 * n_texts + 10
        return (
            est_in_tokens * self.price_input_per_mtok
            + est_out_tokens * self.price_output_per_mtok
        ) / 1_000_000

    def actual_chat_usd(self, usage: dict) -> float:
        """What the last call actually cost, for `SpendCap.finalize`."""
        return price_usd(usage, self.price_input_per_mtok, self.price_output_per_mtok)

    def estimate_batch_usd(self, text_chars: int, n_targets: int) -> float:
        """Reservation for one batch request, at batch prices.

        Pessimistic, because a batch is reserved whole before any of it runs:
        the prompt's own ~130 tokens plus the text in, and per target a token
        for every two characters plus the JSON key around it out. Measured
        output runs lower (830 tokens for one title into 23 on 2026-09-23;
        ~500 on average for authority names), Greek and Bulgarian highest.
        """
        est_in_tokens = text_chars // 3 + 160
        est_out_tokens = n_targets * (text_chars // 2 + 10)
        return (
            est_in_tokens * self.batch_price_input_per_mtok
            + est_out_tokens * self.batch_price_output_per_mtok
        ) / 1_000_000

    def batch_chat_usd(self, usage: dict) -> float:
        """What one batch request cost: its usage at batch prices."""
        return price_usd(usage, self.batch_price_input_per_mtok, self.batch_price_output_per_mtok)

    def record_batch_spend(self, usd: float) -> None:
        """Count a finished batch's cost where every other spend is counted."""
        LLM_SPEND_USD.labels(provider="nebius", endpoint="batch").inc(usd)
