"""TranslationJobs — a whole set of texts, translated behind a job id.

A caller that does not need its translations this minute hands over up to a
few thousand texts at once and polls. Two ways to run a job:

* provider — one Nebius batch: the requests go up as a file, Nebius works
  through them at half the real-time price and outside our rate limits, and
  a poll that finds the batch finished reads the results back. Texts the
  translation cache already holds are answered from it and never sent.
* realtime — here, through the ordinary translate path (cache, breaker,
  budget, provider), every job in the pod sharing one window of calls.

"auto" offers the provider the job and runs it in real time when the
provider refuses batches, then leaves the provider alone for a while.

A realtime job is claimed by one worker, which renews the claim while it
runs. A worker that dies stops renewing; the next claim pass (any pod)
takes the job over, and what was translated before comes from the cache.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import time
import uuid

from loguru import logger

from src.backends.mistral import MistralError, MistralTransientError
from src.backends.nebius import (
    BATCH_WORKING,
    NebiusBackend,
    NebiusBatchRefused,
    NebiusError,
    NebiusTransientError,
)
from src.backends.openai_chat import parse_translation_response
from src.cache.jobs import (
    COMPLETED,
    FAILED,
    FINAL,
    QUEUED,
    RUNNING,
    ItemResult,
    JobItem,
    JobRecord,
    JobStore,
)
from src.domain.models import (
    BackendUnavailable,
    CircuitOpen,
    SpendCapExceeded,
    TranslationBackend,
)
from src.infra.metrics import TRANSLATION_JOB_ITEMS, TRANSLATION_JOBS
from src.services.translation import TranslationService, cache_source

AUTO, PROVIDER, REALTIME = "auto", "provider", "realtime"

#: A batch line that came back with one of these can succeed another time.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

#: Words in the code of a batch line's error (a line with no HTTP status)
#: that put the failure on the provider's side rather than on the request.
_RETRYABLE_ERROR_WORDS = ("expired", "cancel", "timeout", "rate", "server", "unavailable")

#: Realtime jobs claimed per pass. Their items share one window anyway.
_CLAIM_LIMIT = 8

#: Attempts per realtime item through an open breaker or a provider hiccup.
_ITEM_ATTEMPTS = 4


class JobRefused(Exception):
    """The job cannot be taken today: the provider budget is spent."""


def _new_id() -> str:
    return uuid.uuid4().hex


def _describe(exc: BaseException | None) -> str:
    return f"{type(exc).__name__}: {exc}"[:300] if exc else "no answer"


def _today() -> dt.date:
    return dt.datetime.now(dt.UTC).date()


# The knobs are the settings' job_* fields, one each.
class TranslationJobs:  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self, store: JobStore, translation: TranslationService, *,
        mode: str = AUTO, max_concurrency: int = 32, provider_retry_s: float = 900.0,
        lease_s: float = 120.0, poll_s: float = 5.0, retention_days: int = 14,
        backoff_s: float = 5.0, clock=time.monotonic,
    ) -> None:
        self.store = store
        self.translation = translation
        self.mode = mode
        self.provider_retry_s = provider_retry_s
        self.lease_s = lease_s
        self.poll_s = poll_s
        self.retention_days = retention_days
        self.backoff_s = backoff_s
        self.worker_id = f"worker-{_new_id()[:12]}"
        self._clock = clock
        self._window = asyncio.Semaphore(max_concurrency)
        self._provider_refused_until = float("-inf")
        self._running: dict[str, asyncio.Task] = {}
        self._refreshing: dict[str, asyncio.Lock] = {}
        self._wake = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self._next_purge = float("-inf")

    # ── Submission ────────────────────────────────────────────────

    async def submit(self, items: list[JobItem], backend: TranslationBackend,
                     mode: str | None = None) -> JobRecord:
        """Accept a job. Raises ValueError for a malformed one, JobRefused
        when the budget is spent, NebiusBatchRefused when ``mode="provider"``
        and the provider will not take it."""
        _validate(items)
        wanted = mode or self.mode
        if wanted not in (AUTO, PROVIDER, REALTIME):
            raise ValueError(f"unknown job mode {wanted!r}")
        if wanted == PROVIDER:
            return await self._submit_provider(items, backend, self._nebius(backend))
        if wanted == AUTO and self._provider_ready(backend):
            try:
                return await self._submit_provider(items, backend, self._nebius(backend))
            except NebiusBatchRefused as exc:
                self._provider_refused_until = self._clock() + self.provider_retry_s
                TRANSLATION_JOBS.labels(mode=PROVIDER, event="refused").inc()
                logger.warning("provider refused a batch, running jobs here for {:.0f}s: {}",
                               self.provider_retry_s, exc)
        return await self._submit_realtime(items, backend)

    def _provider_ready(self, backend: TranslationBackend) -> bool:
        return (backend is TranslationBackend.NEBIUS and self.translation.nebius is not None
                and self._clock() >= self._provider_refused_until)

    def _nebius(self, backend: TranslationBackend) -> NebiusBackend:
        if backend is not TranslationBackend.NEBIUS:
            raise ValueError("provider batches are served by the nebius backend only")
        if self.translation.nebius is None or self.translation.nebius_spend_cap is None:
            raise BackendUnavailable("nebius backend not configured")
        return self.translation.nebius

    async def _submit_realtime(self, items: list[JobItem],
                               backend: TranslationBackend) -> JobRecord:
        caps = {TranslationBackend.NEBIUS: self.translation.nebius_spend_cap,
                TranslationBackend.MISTRAL: self.translation.mistral_spend_cap}
        cap = caps.get(backend)
        if cap is not None and cap.spent_usd >= cap.daily_cap_usd:
            raise JobRefused(f"daily cap ${cap.daily_cap_usd:.2f} already spent")
        job = await self.store.insert(JobRecord(
            job_id=_new_id(), mode=REALTIME, backend=backend.value, status=QUEUED, items=items))
        TRANSLATION_JOBS.labels(mode=REALTIME, event="submitted").inc()
        self._wake.set()
        return job

    async def _submit_provider(self, items: list[JobItem], backend: TranslationBackend,
                               nebius: NebiusBackend) -> JobRecord:
        prefilled, todo = await self._from_cache(items, backend)
        job_id = _new_id()
        if not todo:
            job = await self.store.insert(JobRecord(
                job_id=job_id, mode=PROVIDER, backend=backend.value, status=QUEUED,
                items=items, prefilled=prefilled))
            return await self.store.finish(job_id, status=COMPLETED, results=prefilled,
                                           cost_usd=0.0) or job
        cap = self.translation.nebius_spend_cap
        assert cap is not None  # _nebius checked it
        estimate = sum(nebius.estimate_batch_usd(len(i.text), len(i.targets)) for i in todo)
        try:
            await cap.reserve(estimate)
        except SpendCapExceeded as exc:
            raise JobRefused(str(exc)) from exc
        try:
            batch_id, file_id = await nebius.submit_batch(
                [nebius.batch_line(i.id, i.text, i.source_lang, i.targets) for i in todo],
                {"job_id": job_id})
        except (NebiusError, NebiusTransientError):
            await cap.release(estimate)
            raise
        logger.info("job {}: provider batch {} with {} of {} items (${:.4f} reserved)",
                    job_id, batch_id, len(todo), len(items), estimate)
        TRANSLATION_JOBS.labels(mode=PROVIDER, event="submitted").inc()
        return await self.store.insert(JobRecord(
            job_id=job_id, mode=PROVIDER, backend=backend.value, status=QUEUED, items=items,
            provider_batch_id=batch_id, provider_file_id=file_id, reserved_usd=estimate,
            reserved_on=_today(), prefilled=prefilled))

    async def _from_cache(self, items: list[JobItem], backend: TranslationBackend
                          ) -> tuple[list[ItemResult], list[JobItem]]:
        """(items the cache answers in full, items still to send)."""
        answered, todo = [], []
        for item in items:
            cached = await self.translation.cache.get_translations(
                item.text, cache_source(item.source_lang), item.targets, backend.value)
            if all(t in cached for t in item.targets):
                answered.append(ItemResult(item.id, {t: cached[t] for t in item.targets}))
            else:
                todo.append(item)
        return answered, todo

    # ── Status ────────────────────────────────────────────────────

    async def status(self, job_id: str) -> JobRecord | None:
        """The job as it stands; a provider job still open is asked about
        first, and read back when the provider has finished it."""
        job = await self.store.get(job_id)
        if job is None or job.status in FINAL or job.mode != PROVIDER:
            return job
        lock = self._refreshing.setdefault(job_id, asyncio.Lock())
        async with lock:
            job = await self.store.get(job_id)
            if job is None or job.status in FINAL:
                return job
            return await self._refresh_provider(job)

    async def _refresh_provider(self, job: JobRecord) -> JobRecord:
        nebius = self._nebius(TranslationBackend(job.backend))
        batch = await nebius.get_batch(job.provider_batch_id or "")
        state = batch.get("status")
        if state in BATCH_WORKING:
            if job.status == QUEUED and state != "validating":
                await self.store.set_running(job.job_id)
                job.status = RUNNING
            return job
        by_id = {i.id: i for i in job.items}
        results = {r.id: r for r in job.prefilled}
        answered = await self._read_batch_files(batch, by_id, nebius, results)
        if state != "completed" and not answered:
            TRANSLATION_JOBS.labels(mode=PROVIDER, event="failed").inc()
            await self._settle(job, 0.0)
            return await self._finished(job, FAILED, None, 0.0,
                                        f"provider batch {state}: {batch.get('errors')}"[:500])
        for item in job.items:
            results.setdefault(item.id, ItemResult(
                item.id, error=f"provider batch {state} without this item", retryable=True))
        ordered = [results[i.id] for i in job.items]
        prefilled = {r.id for r in job.prefilled}
        for r in ordered:
            if r.translations and r.id not in prefilled:
                item = by_id[r.id]
                await self.translation.cache.put_translations(
                    item.text, cache_source(item.source_lang), job.backend, r.translations)
        cost = sum(r.cost_usd for r in ordered)
        nebius.record_batch_spend(cost)
        await self._settle(job, cost)
        _count_items(PROVIDER, ordered)
        TRANSLATION_JOBS.labels(mode=PROVIDER, event="completed").inc()
        logger.info("job {}: provider batch {} {}: {} items, ${:.4f}",
                    job.job_id, job.provider_batch_id, state, len(ordered), cost)
        return await self._finished(job, COMPLETED, ordered, cost, None)

    @staticmethod
    async def _read_batch_files(batch: dict, by_id: dict[str, JobItem], nebius: NebiusBackend,
                                results: dict[str, ItemResult]) -> int:
        """Add the output and error files' answers to ``results``; how many."""
        answered = 0
        for file_id in (batch.get("output_file_id"), batch.get("error_file_id")):
            if not file_id:
                continue
            for line in await nebius.file_lines(file_id):
                result = _line_result(line, by_id, nebius)
                if result is not None and result.id not in results:
                    results[result.id] = result
                    answered += 1
        return answered

    async def _settle(self, job: JobRecord, cost: float) -> None:
        """Replace today's reservation with the real cost. A reservation made
        on an earlier day, or by a pod since restarted, is no longer held."""
        cap = self.translation.nebius_spend_cap
        if cap is not None and job.reserved_on == _today():
            await cap.finalize(job.reserved_usd, cost)

    async def _finished(self, job: JobRecord, status: str, results: list[ItemResult] | None,
                        cost: float, error: str | None) -> JobRecord:
        done = await self.store.finish(job.job_id, status=status, results=results,
                                       cost_usd=cost, error=error)
        return done or await self.store.get(job.job_id) or job

    # ── Realtime worker ───────────────────────────────────────────

    def start(self) -> None:
        self._worker = asyncio.create_task(self.run(), name="translation-jobs")

    async def stop(self) -> None:
        """Stop working and hand back the claims, so a pod taking over need
        not wait for them to lapse."""
        claimed = list(self._running)
        tasks = [t for t in (self._worker, *self._running.values()) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for job_id in claimed:
            try:
                await self.store.release(job_id, self.worker_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("could not release job {}: {}", job_id, exc)
        self._running.clear()

    async def run(self) -> None:
        """Claim realtime jobs as they come, forever. A failing pass is
        logged and the next one tries again: the store may be restarting."""
        while True:
            try:
                await self._claim()
                await self._purge()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("translation jobs worker: {}", _describe(exc))
            try:
                await asyncio.wait_for(self._wake.wait(), self.poll_s)
            except TimeoutError:
                pass
            self._wake.clear()

    async def _claim(self) -> None:
        for job in await self.store.claim_realtime(self.worker_id, self.lease_s, _CLAIM_LIMIT):
            if job.job_id not in self._running:
                self._running[job.job_id] = asyncio.create_task(self._run_realtime(job))

    async def _purge(self) -> None:
        if self._clock() >= self._next_purge:
            self._next_purge = self._clock() + 3600
            removed = await self.store.purge(self.retention_days)
            if removed:
                logger.info("translation jobs: purged {} finished jobs", removed)

    async def _run_realtime(self, job: JobRecord) -> None:
        beat = asyncio.create_task(self._heartbeat(job.job_id))
        try:
            backend = TranslationBackend(job.backend)
            results = list(await asyncio.gather(
                *(self._translate_item(item, backend) for item in job.items)))
            cost = sum(r.cost_usd for r in results)
            await self.store.finish(job.job_id, status=COMPLETED, results=results, cost_usd=cost)
            _count_items(REALTIME, results)
            TRANSLATION_JOBS.labels(mode=REALTIME, event="completed").inc()
            logger.info("job {}: {} items in real time, ${:.4f}", job.job_id, len(results), cost)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("job {} stopped: {}; handing it back", job.job_id, _describe(exc))
            await self.store.release(job.job_id, self.worker_id)
        finally:
            beat.cancel()
            self._running.pop(job.job_id, None)

    async def _heartbeat(self, job_id: str) -> None:
        while True:
            await asyncio.sleep(self.lease_s / 3)
            try:
                if not await self.store.heartbeat(job_id, self.worker_id):
                    logger.warning("job {}: claim taken over by another worker", job_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("job {}: heartbeat failed: {}", job_id, _describe(exc))

    async def _translate_item(self, item: JobItem, backend: TranslationBackend) -> ItemResult:
        """One item through the translate path, never raising. An open
        breaker or a provider hiccup is waited out a few times; a spent
        budget ends the item at once, for the caller to send again later."""
        last: Exception | None = None
        for attempt in range(_ITEM_ATTEMPTS):
            async with self._window:
                try:
                    result = await self.translation.translate(
                        text=item.text, source_lang=item.source_lang,
                        targets=item.targets, backend=backend)
                    return ItemResult(item.id, dict(result.translations),
                                      cost_usd=result.cost_usd)
                except (SpendCapExceeded, BackendUnavailable) as exc:
                    return ItemResult(item.id, error=_describe(exc), retryable=True)
                except (CircuitOpen, NebiusTransientError, MistralTransientError) as exc:
                    last = exc
                except (NebiusError, MistralError, ValueError) as exc:
                    return ItemResult(item.id, error=_describe(exc))
            if attempt + 1 < _ITEM_ATTEMPTS:
                await asyncio.sleep(self.backoff_s * 2 ** attempt)
        return ItemResult(item.id, error=_describe(last), retryable=True)


def _validate(items: list[JobItem]) -> None:
    if not items:
        raise ValueError("a job needs at least one item")
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("item ids must be unique within a job")
    for item in items:
        if not item.text.strip() or not item.targets:
            raise ValueError(f"item {item.id}: empty text or no targets")


def _line_result(line: dict, by_id: dict[str, JobItem],
                 nebius: NebiusBackend) -> ItemResult | None:
    """One line of a batch's output or error file, or None for a line that
    names no item of this job."""
    item = by_id.get(str(line.get("custom_id")))
    if item is None:
        return None
    response = line.get("response") or {}
    code, body, error = response.get("status_code"), response.get("body"), line.get("error")
    if error or code != 200 or not isinstance(body, dict):
        detail = error.get("message") if isinstance(error, dict) else error or body
        return ItemResult(item.id, error=f"provider status={code}: {detail}"[:300],
                          retryable=_retryable(code, error))
    try:
        translations, usage = parse_translation_response(body, item.targets, NebiusError)
    except NebiusError as exc:
        return ItemResult(item.id, error=_describe(exc),
                          cost_usd=nebius.batch_chat_usd(body.get("usage") or {}))
    return ItemResult(item.id, translations, cost_usd=nebius.batch_chat_usd(usage))


def _retryable(code, error) -> bool:
    """A failed batch line worth sending again: a retryable HTTP status, or
    without one an error the provider's side caused (the batch expired or
    was cancelled before reaching it, a rate limit, a server error)."""
    if code is not None:
        return code in _RETRYABLE_STATUS
    kind = str(error.get("code") if isinstance(error, dict) else "").lower()
    return any(word in kind for word in _RETRYABLE_ERROR_WORDS)


def _count_items(mode: str, results: list[ItemResult]) -> None:
    for r in results:
        if not r.error:
            outcome = "ok"
        elif r.retryable:
            outcome = "retryable"
        else:
            outcome = "failed"
        TRANSLATION_JOB_ITEMS.labels(mode=mode, outcome=outcome).inc()
