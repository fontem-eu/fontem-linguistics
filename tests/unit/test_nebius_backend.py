"""Nebius backend: the wire contract, the prompt, and what a call costs."""
from __future__ import annotations

import json

import httpx
import pytest

from src.backends.nebius import NebiusBackend, NebiusError, NebiusTransientError

pytestmark = pytest.mark.asyncio

TARGETS = ["mt", "ga", "et"]


def _build(handler, max_retries: int = 2) -> NebiusBackend:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.studio.nebius.com/v1",
        headers={"Authorization": "Bearer test-key-never-used"},
    )
    return NebiusBackend(
        api_url="https://api.studio.nebius.com/v1",
        api_key="test-key-never-used",
        chat_model="google/gemma-3-27b-it",
        timeout_s=5.0,
        max_retries=max_retries,
        price_input_per_mtok=0.13,
        price_output_per_mtok=0.40,
        client=client,
    )


def _completion(payload: dict, usage: dict | None = None) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(payload)}}],
        "usage": usage or {"prompt_tokens": 138, "completion_tokens": 830},
    })


async def test_translate_returns_every_requested_target():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert body["model"] == "google/gemma-3-27b-it"
        assert body["response_format"] == {"type": "json_object"}
        assert body["temperature"] == 0.0
        return _completion({"mt": "Xogħol", "ga": "Obair", "et": "Töö"})

    got = await _build(handler).translate("Roboty budowlane", "pl", TARGETS)
    assert got == {"mt": "Xogħol", "ga": "Obair", "et": "Töö"}


async def test_the_prompt_names_the_language_not_just_the_code():
    """A model handed only `mt` has been known to answer in Malay. The
    prompt spells out Maltese, Irish and Estonian for exactly that reason."""
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["prompt"] = json.loads(request.content)["messages"][0]["content"]
        return _completion({t: "x" for t in TARGETS})

    await _build(handler).translate("Roboty budowlane", "pl", TARGETS)
    prompt = seen["prompt"]
    for name in ("Maltese", "Irish", "Estonian", "Polish"):
        assert name in prompt
    assert "Roboty budowlane" in prompt


async def test_a_missing_target_is_an_error_not_a_partial_result():
    """Writing a record as translated with languages quietly absent is worse
    than failing: nothing downstream would ever come back for them."""
    async def handler(_request: httpx.Request) -> httpx.Response:
        return _completion({"mt": "Xogħol", "ga": "Obair"})      # et missing

    with pytest.raises(NebiusError, match="missing/malformed"):
        await _build(handler).translate("Roboty budowlane", "pl", TARGETS)


async def test_cost_comes_from_the_providers_usage_block():
    """Measured 2026-09-23 on a real contract title: 138 + 830 tokens.
    At $0.13/$0.40 per Mtok that is $0.00035 — the number the spend cap
    settles on, rather than the pre-call estimate."""
    async def handler(_request: httpx.Request) -> httpx.Response:
        return _completion({t: "x" for t in TARGETS})

    backend = _build(handler)
    _got, cost = await backend.translate_with_cost("Roboty budowlane", "pl", TARGETS)
    assert cost == pytest.approx((138 * 0.13 + 830 * 0.40) / 1_000_000)
    # The estimate under-reserves, which is why finalize exists.
    assert backend.estimate_chat_usd(len("Roboty budowlane"), len(TARGETS)) < cost


async def test_5xx_is_retried_and_4xx_is_not():
    calls = {"n": 0}

    async def flaky(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="upstream busy")
        return _completion({t: "x" for t in TARGETS})

    assert await _build(flaky).translate("t", "pl", TARGETS)
    assert calls["n"] == 2

    calls["n"] = 0

    async def bad_model(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="model not found")

    with pytest.raises(NebiusError, match="404"):
        await _build(bad_model).translate("t", "pl", TARGETS)
    assert calls["n"] == 1, "a 404 is our mistake; retrying only burns the deadline"


async def test_exhausted_retries_surface_as_transient():
    async def always_503(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="still busy")

    with pytest.raises(NebiusTransientError):
        await _build(always_503, max_retries=1).translate("t", "pl", TARGETS)


async def test_malformed_json_is_rejected():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "not json at all"}}],
            "usage": {},
        })

    with pytest.raises(NebiusError, match="malformed"):
        await _build(handler).translate("t", "pl", TARGETS)
