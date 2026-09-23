"""Mistral chat + embed wrappers. Direct httpx; no SDK dependency."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.backends.openai_chat import (
    build_translate_prompt,
    parse_translation_response,
    post_with_retries,
)
from src.infra.metrics import LLM_SPEND_USD, MISTRAL_SPEND_USD


class MistralError(Exception):
    """Raised when Mistral returns a non-retriable error or malformed payload."""


class MistralTransientError(Exception):
    """Raised on 5xx, 429, or network timeouts — retriable by the service."""


# Language names, the prompt and the retry policy live in openai_chat:
# Nebius speaks the same protocol and must not drift from this one.


# MistralBackend is a configuration-bag dataclass for a single hosted backend;
# splitting the 10 fields into sub-structs would obscure that they all map
# directly to env vars in infra/config.py.
@dataclass
class MistralBackend:  # pylint: disable=too-many-instance-attributes
    api_url: str
    api_key: str
    chat_model: str
    embed_model: str
    timeout_s: float
    max_retries: int
    price_input_per_mtok: float
    price_output_per_mtok: float
    price_embed_per_mtok: float
    client: httpx.AsyncClient

    @property
    def embed_encoder_id(self) -> str:
        """Best-effort signed-mirror identity for a hosted API we don't
        control. ``mistral-embed@api-<model>`` rather than a version SHA —
        Mistral doesn't expose per-revision identifiers, and we don't mirror
        this backend. Downstream consumers MUST NOT compare Mistral vectors
        to LaBSE vectors (different encoder family, different vector space);
        the encoder-id prefix is how they tell."""
        return f"mistral-embed@api-{self.embed_model}"

    @classmethod
    # Keyword-only factory mirroring the dataclass fields one-to-one; named
    # parameters at the call site beat any kwargs-dict alternative.
    def build(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        cls,
        api_url: str,
        api_key: str,
        chat_model: str,
        embed_model: str,
        timeout_s: float,
        max_retries: int,
        price_input_per_mtok: float,
        price_output_per_mtok: float,
        price_embed_per_mtok: float,
    ) -> "MistralBackend":
        client = httpx.AsyncClient(
            base_url=api_url.rstrip("/"),
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return cls(
            api_url=api_url, api_key=api_key, chat_model=chat_model, embed_model=embed_model,
            timeout_s=timeout_s, max_retries=max_retries,
            price_input_per_mtok=price_input_per_mtok,
            price_output_per_mtok=price_output_per_mtok,
            price_embed_per_mtok=price_embed_per_mtok,
            client=client,
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def translate(
        self, text: str, source_lang: str, targets: list[str]
    ) -> dict[str, str]:
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "user", "content": build_translate_prompt(text, source_lang, targets)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        data = await self._post_with_retries("/chat/completions", payload)
        translations, usage = parse_translation_response(data, targets, MistralError)
        self._record_chat_spend(usage)
        return translations

    async def embed(self, text: str) -> list[float]:
        payload = {"model": self.embed_model, "input": [text]}
        data = await self._post_with_retries("/embeddings", payload)
        try:
            vec = data["data"][0]["embedding"]
        except (KeyError, IndexError) as exc:
            raise MistralError(f"malformed embed response: {exc}") from exc

        if (
            not isinstance(vec, list)
            or not vec
            or not all(isinstance(v, (int, float)) for v in vec)
        ):
            raise MistralError("embed response vector is malformed")

        usage = data.get("usage") or {}
        self._record_embed_spend(usage)
        return [float(v) for v in vec]

    # ── Internals ─────────────────────────────────────────────────

    async def _post_with_retries(self, path: str, payload: dict) -> dict:
        return await post_with_retries(
            self.client, path, payload,
            max_retries=self.max_retries,
            transient_cls=MistralTransientError, error_cls=MistralError,
        )

    def _record_chat_spend(self, usage: dict) -> None:
        p_in = usage.get("prompt_tokens", 0)
        p_out = usage.get("completion_tokens", 0)
        cost = (p_in * self.price_input_per_mtok + p_out * self.price_output_per_mtok) / 1_000_000
        MISTRAL_SPEND_USD.labels(endpoint="chat").inc(cost)
        LLM_SPEND_USD.labels(provider="mistral", endpoint="chat").inc(cost)

    def _record_embed_spend(self, usage: dict) -> None:
        p_in = usage.get("prompt_tokens", 0)
        cost = p_in * self.price_embed_per_mtok / 1_000_000
        MISTRAL_SPEND_USD.labels(endpoint="embed").inc(cost)
        LLM_SPEND_USD.labels(provider="mistral", endpoint="embed").inc(cost)

    def estimate_chat_usd(self, text_chars: int, n_targets: int) -> float:
        """Rough pre-call estimate for spend-cap reservation. 1 token ≈ 4 chars."""
        est_in_tokens = max(40, text_chars // 4 + 80)
        est_out_tokens = max(20, n_targets * text_chars // 4)
        return (
            est_in_tokens * self.price_input_per_mtok
            + est_out_tokens * self.price_output_per_mtok
        ) / 1_000_000

    def estimate_embed_usd(self, text_chars: int) -> float:
        est_tokens = max(10, text_chars // 4)
        return est_tokens * self.price_embed_per_mtok / 1_000_000
