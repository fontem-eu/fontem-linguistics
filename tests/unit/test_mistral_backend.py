"""Mistral backend: request/response parsing + retry/error paths."""
from __future__ import annotations

import json

import httpx
import pytest

from src.backends.mistral import (
    MistralBackend,
    MistralError,
    MistralTransientError,
)


pytestmark = pytest.mark.asyncio


def _build(handler) -> MistralBackend:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(
        transport=transport,
        base_url="https://api.mistral.ai/v1",
        headers={"Authorization": "Bearer test-key-never-used"},
    )
    return MistralBackend(
        api_url="https://api.mistral.ai/v1",
        api_key="test-key-never-used",
        chat_model="mistral-small-latest",
        embed_model="mistral-embed",
        timeout_s=5.0,
        max_retries=2,
        price_input_per_mtok=0.20,
        price_output_per_mtok=0.60,
        price_embed_per_mtok=0.10,
        client=client,
    )


async def test_translate_parses_json_mode_response():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/chat/completions"
        body = json.loads(req.content)
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({
                    "en": "Ministry of Defence",
                    "fr": "Ministère de la Défense",
                })}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40},
            },
        )

    backend = _build(handler)
    out = await backend.translate("Ministero della Difesa", "it", ["en", "fr"])
    assert out == {"en": "Ministry of Defence", "fr": "Ministère de la Défense"}
    await backend.aclose()


async def test_translate_raises_on_missing_target():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"en": "Hello"})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    backend = _build(handler)
    with pytest.raises(MistralError, match="missing/malformed"):
        await backend.translate("Bonjour", "fr", ["en", "de"])
    await backend.aclose()


async def test_translate_raises_on_malformed_json_content():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "not-json"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )

    backend = _build(handler)
    with pytest.raises(MistralError, match="malformed chat response"):
        await backend.translate("hi", "en", ["fr"])
    await backend.aclose()


async def test_translate_retries_on_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"en": "ok"})}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    backend = _build(handler)
    out = await backend.translate("x", "fr", ["en"])
    assert out == {"en": "ok"}
    assert calls["n"] == 2
    await backend.aclose()


async def test_translate_gives_up_after_max_retries():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    backend = _build(handler)
    with pytest.raises(MistralTransientError):
        await backend.translate("x", "fr", ["en"])
    await backend.aclose()


async def test_translate_raises_on_4xx_immediately():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    backend = _build(handler)
    with pytest.raises(MistralError, match="status=400"):
        await backend.translate("x", "fr", ["en"])
    assert calls["n"] == 1  # no retry on 4xx non-429
    await backend.aclose()


async def test_translate_retries_on_429():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(429, text="slow down")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"fr": "salut"})}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    backend = _build(handler)
    out = await backend.translate("hi", "en", ["fr"])
    assert out == {"fr": "salut"}
    assert calls["n"] == 2
    await backend.aclose()


async def test_embed_parses_vector():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/embeddings"
        return httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}],
                "usage": {"prompt_tokens": 5},
            },
        )

    backend = _build(handler)
    vec = await backend.embed("hello")
    assert vec == [0.1, 0.2, 0.3, 0.4]
    await backend.aclose()


async def test_embed_raises_on_empty_vector():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": []}]})

    backend = _build(handler)
    with pytest.raises(MistralError):
        await backend.embed("hello")
    await backend.aclose()


async def test_estimates_are_positive():
    backend = _build(lambda r: httpx.Response(200, json={}))
    assert backend.estimate_chat_usd(text_chars=30, n_targets=6) > 0
    assert backend.estimate_embed_usd(text_chars=30) > 0
    await backend.aclose()


async def test_embed_raises_on_missing_data_key():
    def handler(req):
        return httpx.Response(200, json={"not_data": []})

    backend = _build(handler)
    with pytest.raises(MistralError, match="malformed embed"):
        await backend.embed("x")
    await backend.aclose()


async def test_transport_error_retries_then_raises():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        raise httpx.ConnectError("refused")

    backend = _build(handler)
    with pytest.raises(MistralTransientError):
        await backend.translate("x", "en", ["fr"])
    assert calls["n"] == 3   # initial + max_retries=2
    await backend.aclose()


async def test_build_factory_creates_client():
    b = MistralBackend.build(
        api_url="https://example/v1", api_key="test-key-never-used",
        chat_model="m", embed_model="me", timeout_s=5.0, max_retries=1,
        price_input_per_mtok=0.1, price_output_per_mtok=0.3, price_embed_per_mtok=0.05,
    )
    assert b.api_key == "test-key-never-used"
    assert isinstance(b.client, httpx.AsyncClient)
    await b.aclose()
