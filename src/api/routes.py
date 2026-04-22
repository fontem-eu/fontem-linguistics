"""HTTP routes. Thin shim — all work happens in the services."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, Response
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.deps import Services
from src.api.schemas import (
    BatchTranslateRequest,
    BatchTranslateResponse,
    EmbedRequest,
    EmbedResponse,
    TranslateRequest,
    TranslateResponse,
)
from src.backends.mistral import MistralError, MistralTransientError
from src.domain.models import (
    BackendUnavailable,
    CircuitOpen,
    SpendCapExceeded,
    TranslationBackend,
)

router = APIRouter()


def _services(request: Request) -> Services:
    svc: Services = request.app.state.services
    return svc


@router.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest, request: Request) -> TranslateResponse:
    try:
        result = await _services(request).translation.translate(
            text=req.text,
            source_lang=req.source_lang,
            targets=req.targets,
            backend=req.backend,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CircuitOpen as exc:
        return _circuit_open_response(exc)
    except SpendCapExceeded as exc:
        return _spend_cap_response(exc)
    except BackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MistralTransientError as exc:
        raise HTTPException(status_code=502, detail=f"mistral transient failure: {exc}") from exc
    except MistralError as exc:
        raise HTTPException(status_code=502, detail=f"mistral error: {exc}") from exc

    return TranslateResponse(
        cached=result.fully_cached,
        backend=result.backend,
        translations=result.translations,
        partial_cached_targets=sorted(result.cached_targets),
    )


# In-process idempotency memo: idempotency_key → last response.
# Size-bounded; resets on pod restart, matching the "at least once" guarantee
# we give callers.
_IDEMPOTENCY_STORE: "OrderedDict[str, BatchTranslateResponse]" = OrderedDict()
_IDEMPOTENCY_MAX = 2048


@router.post("/translate/batch", response_model=BatchTranslateResponse)
async def translate_batch(
    req: BatchTranslateRequest,
    request: Request,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> BatchTranslateResponse:
    if idempotency_key and idempotency_key in _IDEMPOTENCY_STORE:
        return _IDEMPOTENCY_STORE[idempotency_key]

    svc = _services(request).translation

    async def _one(item) -> TranslateResponse:
        try:
            result = await svc.translate(
                text=item.text,
                source_lang=item.source_lang,
                targets=req.targets,
                backend=req.backend,
            )
            return TranslateResponse(
                cached=result.fully_cached,
                backend=result.backend,
                translations=result.translations,
                partial_cached_targets=sorted(result.cached_targets),
            )
        except CircuitOpen as exc:
            # Surface per-item: batch partial completion is acceptable. Return
            # an empty translations dict and let the caller retry the failed
            # items. Simpler than aborting the whole batch.
            logger.warning("batch item circuit-open for backend={}", req.backend)
            return TranslateResponse(
                cached=False, backend=req.backend, translations={},
                partial_cached_targets=[],
            )

    results = await asyncio.gather(*[_one(it) for it in req.items])
    response = BatchTranslateResponse(results=results)

    if idempotency_key:
        _IDEMPOTENCY_STORE[idempotency_key] = response
        if len(_IDEMPOTENCY_STORE) > _IDEMPOTENCY_MAX:
            _IDEMPOTENCY_STORE.popitem(last=False)

    return response


@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest, request: Request) -> EmbedResponse:
    try:
        result = await _services(request).embedding.embed(req.text, req.backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CircuitOpen as exc:
        return _circuit_open_response(exc)
    except SpendCapExceeded as exc:
        return _spend_cap_response(exc)
    except BackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MistralTransientError as exc:
        raise HTTPException(status_code=502, detail=f"mistral transient failure: {exc}") from exc
    except MistralError as exc:
        raise HTTPException(status_code=502, detail=f"mistral error: {exc}") from exc

    return EmbedResponse(
        cached=result.cached, backend=result.backend, dim=result.dim, vector=result.vector,
    )


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> dict:
    try:
        cache = request.app.state.cache
        async with cache.pool.acquire() as con:
            await con.fetchval("SELECT 1")
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"db unavailable: {exc}") from exc


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _circuit_open_response(exc: Exception) -> TranslateResponse:
    # Fail-loud: 503 with explicit header. The route-level return is a shim —
    # raise so FastAPI serializes it consistently.
    raise HTTPException(
        status_code=503,
        detail=str(exc),
        headers={"X-Backend-State": "circuit-open"},
    )


def _spend_cap_response(exc: Exception) -> TranslateResponse:
    raise HTTPException(
        status_code=429,
        detail=str(exc),
        headers={"X-Backend-State": "spend-cap-exceeded"},
    )
