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
    KeywordsRequest,
    KeywordsResponse,
    LanguageInfo,
    LanguagesResponse,
    ModelInfoResponse,
    ModelsResponse,
    TranslateRequest,
    TranslateResponse,
)
from src.domain.catalog import CATALOG
from src.domain.keywords import extract_keywords
from src.domain.languages import EU_OFFICIAL_LANGS, LANG_DISPLAY_NAMES
from src.backends.mistral import MistralError, MistralTransientError
from src.domain.models import (
    BackendUnavailable,
    CircuitOpen,
    SpendCapExceeded,
)

router = APIRouter()


def _services(request: Request) -> Services:
    svc: Services = request.app.state.services
    return svc


@router.post(
    "/translate",
    responses={
        400: {"description": "Invalid request (empty text or empty targets)."},
        429: {"description": "Daily spend cap exceeded for the Mistral backend."},
        502: {"description": "Mistral upstream returned an error or transient failure."},
        503: {"description": "Backend unavailable or circuit breaker open."},
    },
)
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
        raise HTTPException(
            status_code=503, detail=str(exc),
            headers={"X-Backend-State": "circuit-open"},
        ) from exc
    except SpendCapExceeded as exc:
        raise HTTPException(
            status_code=429, detail=str(exc),
            headers={"X-Backend-State": "spend-cap-exceeded"},
        ) from exc
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


@router.post("/translate/batch")
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
        except CircuitOpen:
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


@router.post(
    "/embed",
    responses={
        400: {"description": "Invalid request (empty text)."},
        429: {"description": "Daily spend cap exceeded for the Mistral backend."},
        502: {"description": "Mistral upstream returned an error or transient failure."},
        503: {"description": "Backend unavailable or circuit breaker open."},
    },
)
async def embed(req: EmbedRequest, request: Request) -> EmbedResponse:
    try:
        result = await _services(request).embedding.embed(req.text, req.backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CircuitOpen as exc:
        raise HTTPException(
            status_code=503, detail=str(exc),
            headers={"X-Backend-State": "circuit-open"},
        ) from exc
    except SpendCapExceeded as exc:
        raise HTTPException(
            status_code=429, detail=str(exc),
            headers={"X-Backend-State": "spend-cap-exceeded"},
        ) from exc
    except BackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MistralTransientError as exc:
        raise HTTPException(status_code=502, detail=f"mistral transient failure: {exc}") from exc
    except MistralError as exc:
        raise HTTPException(status_code=502, detail=f"mistral error: {exc}") from exc

    return EmbedResponse(
        cached=result.cached, backend=result.backend, dim=result.dim,
        vector=result.vector, encoder_id=result.encoder_id,
    )


@router.get("/models")
async def models(request: Request) -> ModelsResponse:
    """List available backends with quality scores + encoder identities.

    Callers use this to pick a tier; the encoder_id for embedders reflects
    the currently-loaded signed-mirror revision (null for translators).
    Tolerates the lifespan-not-yet-run case so /models still works before
    services are initialised (e.g. during readiness polling pre-warmup).
    """
    encoder_ids: dict[str, str | None] = {}
    services = getattr(request.app.state, "services", None)
    if services is not None:
        if services.embedding.mistral is not None:
            encoder_ids["mistral-embed"] = services.embedding.mistral.embed_encoder_id
        if services.embedding.labse is not None:
            encoder_ids["labse-local"] = services.embedding.labse.encoder_id
    return ModelsResponse(models=[
        ModelInfoResponse(
            backend=m.backend, kind=m.kind, quality_score=m.quality_score,
            dim=m.dim, languages_supported=m.languages_supported,
            cost_tier=m.cost_tier, description=m.description,
            encoder_id=encoder_ids.get(m.backend),
        )
        for m in CATALOG
    ])


@router.post(
    "/keywords",
    responses={400: {"description": "Text is empty after trimming."}},
)
async def keywords(req: KeywordsRequest) -> KeywordsResponse:
    """Tokenize text and strip stop words (24 EU languages).

    Stateless — no backend, cache or spend cap involved. When ``lang`` is
    omitted the language is detected from stop-word hits; when neither an
    explicit nor a detected language has a published list (e.g. Maltese),
    tokens pass through unfiltered.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must be non-empty")
    result = extract_keywords(req.text, req.lang)
    return KeywordsResponse(
        lang=result.lang,
        tokens=result.tokens,
        keywords=result.keywords,
        removed=result.removed,
    )


@router.get("/languages")
async def languages() -> LanguagesResponse:
    """Canonical list of the 24 EU official languages that callers target."""
    return LanguagesResponse(languages=[
        LanguageInfo(code=c, name=LANG_DISPLAY_NAMES[c]) for c in EU_OFFICIAL_LANGS
    ])


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get(
    "/readyz",
    responses={503: {"description": "Backing database is unavailable."}},
)
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
