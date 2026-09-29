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
    EmbedBatchRequest,
    EmbedBatchResponse,
    BatchTranslateRequest,
    BatchTranslateResponse,
    DetectedLanguage,
    DetectRequest,
    DetectResponse,
    EmbedRequest,
    EmbedResponse,
    JobItemResultModel,
    JobResponse,
    JobSubmitRequest,
    LanguageInfo,
    LanguagesResponse,
    ModelInfoResponse,
    ModelsResponse,
    TranslateRequest,
    TranslateResponse,
)
from src.domain.catalog import CATALOG
from src.domain.languages import EU_OFFICIAL_LANGS, LANG_DISPLAY_NAMES
from src.backends.mistral import MistralError, MistralTransientError
from src.backends.nebius import NebiusBatchRefused, NebiusError, NebiusTransientError
from src.cache.jobs import JobItem, JobRecord
from src.services.jobs import JobRefused, TranslationJobs
from src.domain.models import (
    BackendUnavailable,
    CircuitOpen,
    SpendCapExceeded,
    TranslationBackend,
)

#: What one batch item may fail with and still let the rest of the batch
#: finish. Anything outside this set is a bug and should surface as one.
_BATCH_ITEM_FAILURES = (
    CircuitOpen,
    BackendUnavailable,
    SpendCapExceeded,
    MistralError,
    MistralTransientError,
    NebiusError,
    NebiusTransientError,
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
    except NebiusTransientError as exc:
        raise HTTPException(status_code=502, detail=f"nebius transient failure: {exc}") from exc
    except NebiusError as exc:
        raise HTTPException(status_code=502, detail=f"nebius error: {exc}") from exc

    return TranslateResponse(
        cached=result.fully_cached,
        backend=result.backend,
        translations=result.translations,
        partial_cached_targets=sorted(result.cached_targets),
        cost_usd=result.cost_usd,
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
    window = asyncio.Semaphore(request.app.state.settings.batch_max_concurrency)

    async def _one(item) -> TranslateResponse:
        """One item, inside the concurrency window, never raising.

        Any failure becomes this item's `error` rather than the batch's: one
        bad title, or the budget running out partway, must not throw away the
        translations — and the reported cost — of the items that succeeded.
        """
        async with window:
            try:
                result = await svc.translate(
                    text=item.text,
                    source_lang=item.source_lang,
                    targets=req.targets,
                    backend=req.backend,
                )
            except _BATCH_ITEM_FAILURES as exc:
                logger.warning("batch item failed backend={} error={}", req.backend, exc)
                return TranslateResponse(
                    cached=False, backend=req.backend, translations={},
                    partial_cached_targets=[], error=f"{type(exc).__name__}: {exc}",
                )
            return TranslateResponse(
                cached=result.fully_cached,
                backend=result.backend,
                translations=result.translations,
                partial_cached_targets=sorted(result.cached_targets),
                cost_usd=result.cost_usd,
            )

    results = await asyncio.gather(*[_one(it) for it in req.items])
    response = BatchTranslateResponse(results=results)

    if idempotency_key:
        _IDEMPOTENCY_STORE[idempotency_key] = response
        if len(_IDEMPOTENCY_STORE) > _IDEMPOTENCY_MAX:
            _IDEMPOTENCY_STORE.popitem(last=False)

    return response


#: Texts per detection prompt. Measured 2026-09-28 on 220 prod titles with
#: a stated language: 214 right at 20 per prompt, 212 at 40, for about
#: USD 0.008 per thousand titles either way.
DETECT_CHUNK = 20


@router.post(
    "/detect",
    responses={
        400: {"description": "An empty text, or a backend that does not detect."},
    },
)
async def detect(req: DetectRequest, request: Request) -> DetectResponse:
    """Each text's language, in request order.

    Like the translation batch, a chunk that fails marks its own texts with
    the reason and the rest still answer: a caller keeping what it paid for
    must not lose it to one refused chunk.
    """
    svc = _services(request).translation
    if any(not t or not t.strip() for t in req.texts):
        raise HTTPException(status_code=400, detail="every text must be non-empty")
    if req.backend is not TranslationBackend.NEBIUS:
        raise HTTPException(status_code=400,
                            detail="language detection is served by the nebius backend only")
    window = asyncio.Semaphore(request.app.state.settings.batch_max_concurrency)
    chunks = [req.texts[i:i + DETECT_CHUNK] for i in range(0, len(req.texts), DETECT_CHUNK)]

    async def _one(chunk: list[str]) -> tuple[list[DetectedLanguage], float, str | None]:
        async with window:
            try:
                result = await svc.detect(chunk, req.backend)
            except _BATCH_ITEM_FAILURES as exc:
                logger.warning("detect chunk failed error={}", exc)
                reason = f"{type(exc).__name__}: {exc}"
                return [DetectedLanguage(lang=None, error=reason) for _ in chunk], 0.0, None
        return ([DetectedLanguage(lang=lang, error=None if lang else "no answer for this text")
                 for lang in result.langs], result.cost_usd, result.model)

    answered = await asyncio.gather(*[_one(c) for c in chunks])
    return DetectResponse(
        backend=req.backend,
        model=next((m for _r, _c, m in answered if m), None),
        results=[r for rs, _c, _m in answered for r in rs],
        cost_usd=sum(c for _r, c, _m in answered),
    )


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
    except NebiusTransientError as exc:
        raise HTTPException(status_code=502, detail=f"nebius transient failure: {exc}") from exc
    except NebiusError as exc:
        raise HTTPException(status_code=502, detail=f"nebius error: {exc}") from exc

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
        if services.embedding.minilm is not None:
            encoder_ids["minilm-local"] = services.embedding.minilm.encoder_id
    return ModelsResponse(models=[
        ModelInfoResponse(
            backend=m.backend, kind=m.kind, quality_score=m.quality_score,
            dim=m.dim, languages_supported=m.languages_supported,
            cost_tier=m.cost_tier, description=m.description,
            encoder_id=encoder_ids.get(m.backend),
        )
        for m in CATALOG
    ])


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

@router.post(
    "/embed_batch",
    # No response_model= — the return annotation below already declares it,
    # and FastAPI derives the schema from that.
    responses={
        400: {"description": "Invalid input."},
        # This route raises 429 on SpendCapExceeded exactly like /translate
        # and /embed do, but did not advertise it, so a client generated
        # from the OpenAPI schema had no branch for a spend-cap rejection.
        429: {"description": "Daily spend cap exceeded for the Mistral backend."},
        503: {"description": "Configured backend unavailable / circuit open."},
    },
)
async def embed_batch(req: EmbedBatchRequest, request: Request) -> EmbedBatchResponse:
    """Batched embed — one HTTP round-trip, one BLAS-batched model call.

    Same semantics per element as /embed; ordering preserved. Cache
    hits are per text so mixed batches (some cached, some new) are fine.
    """
    try:
        results = await _services(request).embedding.embed_batch(req.texts, req.backend)
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
        raise HTTPException(
            status_code=503, detail=str(exc),
            headers={"X-Backend-State": "unavailable"},
        ) from exc

    if not results:
        return EmbedBatchResponse(
            backend=req.backend, dim=0, encoder_id="",
            results=[],
        )
    return EmbedBatchResponse(
        backend=req.backend, dim=results[0].dim, encoder_id=results[0].encoder_id,
        results=[
            EmbedResponse(
                cached=r.cached, backend=r.backend, dim=r.dim,
                vector=r.vector, encoder_id=r.encoder_id,
            )
            for r in results
        ],
    )


def _jobs(request: Request) -> TranslationJobs:
    jobs = _services(request).jobs
    if jobs is None:
        raise HTTPException(status_code=503, detail="translation jobs are not running")
    return jobs


def _job_response(job: JobRecord) -> JobResponse:
    return JobResponse(
        job_id=job.job_id, mode=job.mode, status=job.status, n_items=job.n_items,
        cost_usd=job.cost_usd, error=job.error, created_at=job.created_at,
        completed_at=job.completed_at,
        results=None if job.results is None else [
            JobItemResultModel(**r.__dict__) for r in job.results],
    )


@router.post(
    "/translate/jobs",
    status_code=202,
    responses={
        400: {"description": "Empty or duplicate items, or a mode the backend cannot run."},
        429: {"description": "The provider budget for today is spent."},
        502: {"description": "The provider failed to take the batch; send it again later."},
        503: {"description": "Provider batches refused (mode=provider) or backend unavailable."},
    },
)
async def submit_translation_job(req: JobSubmitRequest, request: Request) -> JobResponse:
    """Accept a set of texts to translate; poll GET /translate/jobs/{job_id}."""
    items = [JobItem(id=i.id, text=i.text, source_lang=i.source_lang, targets=i.targets)
             for i in req.items]
    try:
        job = await _jobs(request).submit(items, req.backend, req.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except JobRefused as exc:
        raise HTTPException(status_code=429, detail=str(exc),
                            headers={"X-Backend-State": "spend-cap-exceeded"}) from exc
    except (NebiusBatchRefused, BackendUnavailable) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (NebiusTransientError, NebiusError) as exc:
        raise HTTPException(status_code=502, detail=f"nebius: {exc}") from exc
    return _job_response(job)


@router.get(
    "/translate/jobs/{job_id}",
    responses={
        404: {"description": "No such job (never submitted, or past its retention)."},
        502: {"description": "The provider could not be asked; poll again later."},
        503: {"description": "Translation jobs or the provider backend are not configured."},
    },
)
async def translation_job(job_id: str, request: Request) -> JobResponse:
    """The job's status; its results once completed."""
    try:
        job = await _jobs(request).status(job_id)
    except BackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (NebiusTransientError, NebiusError) as exc:
        raise HTTPException(status_code=502, detail=f"nebius: {exc}") from exc
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id}")
    return _job_response(job)
