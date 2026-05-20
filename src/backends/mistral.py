"""Mistral chat + embed wrappers. Direct httpx; no SDK dependency."""
from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass

import httpx
from loguru import logger

from src.infra.metrics import MISTRAL_SPEND_USD


class MistralError(Exception):
    """Raised when Mistral returns a non-retriable error or malformed payload."""


class MistralTransientError(Exception):
    """Raised on 5xx, 429, or network timeouts — retriable by the service."""


_LANG_FULLNAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pl": "Polish", "pt": "Portuguese", "nl": "Dutch",
    "ro": "Romanian", "cs": "Czech", "hu": "Hungarian", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "el": "Greek", "bg": "Bulgarian",
    "hr": "Croatian", "sk": "Slovak", "sl": "Slovene", "lt": "Lithuanian",
    "lv": "Latvian", "et": "Estonian", "ga": "Irish", "mt": "Maltese",
}


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

    @staticmethod
    def _lang_fullname(code: str) -> str:
        return _LANG_FULLNAMES.get(code.lower(), code.upper())

    def _build_translate_prompt(self, text: str, source_lang: str, targets: list[str]) -> str:
        src = self._lang_fullname(source_lang)
        wanted = {t: self._lang_fullname(t) for t in targets}
        keys = ", ".join(f'"{t}"' for t in targets)
        pretty_targets = ", ".join(f"{code} ({name})" for code, name in wanted.items())
        return (
            "Translate the following text from "
            f"{src} into the target languages. Preserve institutional "
            "terminology, do not paraphrase. Return strict JSON with keys: "
            f"{keys}. No prose, no explanation.\n"
            f"Target languages: {pretty_targets}.\n"
            f"Text: {text}"
        )

    async def translate(
        self, text: str, source_lang: str, targets: list[str]
    ) -> dict[str, str]:
        prompt = self._build_translate_prompt(text, source_lang, targets)
        payload = {
            "model": self.chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        data = await self._post_with_retries("/chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise MistralError(f"malformed chat response: {exc}") from exc

        missing = [t for t in targets if t not in parsed or not isinstance(parsed[t], str)]
        if missing:
            raise MistralError(f"missing/malformed target(s) in response: {missing}")

        usage = data.get("usage") or {}
        self._record_chat_spend(usage)
        return {t: parsed[t] for t in targets}

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
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.client.post(path, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = MistralTransientError(str(exc))
            else:
                if resp.status_code >= 500 or resp.status_code == 429:
                    last_exc = MistralTransientError(
                        f"status={resp.status_code} body={resp.text[:200]}"
                    )
                elif resp.status_code >= 400:
                    # 4xx non-429 are not retriable — bad request, auth, etc.
                    raise MistralError(
                        f"status={resp.status_code} body={resp.text[:500]}"
                    )
                else:
                    return resp.json()

            if attempt < self.max_retries:
                backoff = (2 ** attempt) * 0.25 + random.random() * 0.1  # nosec B311
                logger.warning("mistral attempt {} failed: {}; retry in {:.2f}s",
                               attempt + 1, last_exc, backoff)
                await asyncio.sleep(backoff)
        if last_exc is None:
            raise MistralError("no attempt recorded an outcome")
        raise last_exc

    def _record_chat_spend(self, usage: dict) -> None:
        p_in = usage.get("prompt_tokens", 0)
        p_out = usage.get("completion_tokens", 0)
        cost = (p_in * self.price_input_per_mtok + p_out * self.price_output_per_mtok) / 1_000_000
        MISTRAL_SPEND_USD.labels(endpoint="chat").inc(cost)

    def _record_embed_spend(self, usage: dict) -> None:
        p_in = usage.get("prompt_tokens", 0)
        cost = p_in * self.price_embed_per_mtok / 1_000_000
        MISTRAL_SPEND_USD.labels(endpoint="embed").inc(cost)

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
