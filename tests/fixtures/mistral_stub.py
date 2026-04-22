"""A FastAPI stub of the Mistral chat + embed endpoints.

Deterministic responses, injectable latency and failure modes. Used as the
backend target in component tests so the real Mistral API is never hit.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


@dataclass
class StubState:
    latency_s: float = 0.0
    # Per-endpoint queues of responses to override the default. Each entry is
    # (status_code, body). Pop from front on each call. When empty, defaults apply.
    chat_responses: list[tuple[int, dict]] = field(default_factory=list)
    embed_responses: list[tuple[int, dict]] = field(default_factory=list)
    calls: list[tuple[str, dict]] = field(default_factory=list)


import re

_KEYS_LINE_RE = re.compile(r"keys:\s*([^\n]+)")
_ISO_RE = re.compile(r'"([a-z]{2,3})"')


def _default_chat_response(prompt_text: str) -> dict:
    m = _KEYS_LINE_RE.search(prompt_text)
    keys = _ISO_RE.findall(m.group(1)) if m else []
    content = json.dumps({k: f"[{k}]stub" for k in keys})
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20},
    }


def make_stub() -> tuple[FastAPI, StubState]:
    state = StubState()
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        state.calls.append(("chat", body))
        if state.latency_s > 0:
            await asyncio.sleep(state.latency_s)
        if state.chat_responses:
            status, payload = state.chat_responses.pop(0)
            return JSONResponse(payload, status_code=status)
        prompt_text = body["messages"][0]["content"]
        return JSONResponse(_default_chat_response(prompt_text))

    @app.post("/v1/embeddings")
    async def embed(request: Request):
        body = await request.json()
        state.calls.append(("embed", body))
        if state.latency_s > 0:
            await asyncio.sleep(state.latency_s)
        if state.embed_responses:
            status, payload = state.embed_responses.pop(0)
            return JSONResponse(payload, status_code=status)
        # Deterministic vector derived from the input length.
        text = body["input"][0]
        vec = [(i % 7) / 10.0 for i in range(1024)]
        return JSONResponse({
            "data": [{"embedding": vec}],
            "usage": {"prompt_tokens": max(5, len(text) // 4)},
        })

    return app, state
