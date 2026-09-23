"""Nebius AI Studio chat backend — translation only.

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

from dataclasses import dataclass

import httpx

from src.backends.openai_chat import (
    build_translate_prompt,
    parse_translation_response,
    post_with_retries,
    price_usd,
)
from src.infra.metrics import LLM_SPEND_USD


class NebiusError(Exception):
    """Non-retriable Nebius failure, or a malformed payload."""


class NebiusTransientError(Exception):
    """5xx, 429 or a network timeout — retriable by the service."""


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
        """
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "user", "content": build_translate_prompt(text, source_lang, targets)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        data = await post_with_retries(
            self.client, "/chat/completions", payload,
            max_retries=self.max_retries,
            transient_cls=NebiusTransientError, error_cls=NebiusError,
        )
        translations, usage = parse_translation_response(data, targets, NebiusError)
        cost = self.actual_chat_usd(usage)
        self._record_chat_spend(usage)
        return translations, cost

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

    def actual_chat_usd(self, usage: dict) -> float:
        """What the last call actually cost, for `SpendCap.finalize`."""
        return price_usd(usage, self.price_input_per_mtok, self.price_output_per_mtok)
