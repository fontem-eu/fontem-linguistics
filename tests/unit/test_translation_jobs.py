"""Translation jobs: provider batches, realtime runs, and the fallback
between them. The store, the cache and the provider are faked; the spend
cap and the Nebius client are the real ones."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
from dataclasses import replace

import httpx
import pytest

from src.backends.nebius import (
    NebiusBackend,
    NebiusBatchRefused,
    NebiusError,
    NebiusTransientError,
)
from src.cache.jobs import COMPLETED, FAILED, QUEUED, RUNNING, JobItem
from src.domain.models import (
    BackendUnavailable,
    CircuitOpen,
    SpendCapExceeded,
    SummaryResult,
    TranslationBackend,
    TranslationResult,
)
from src.infra.spend_cap import SpendCap
from src.services.jobs import JobRefused, TranslationJobs
from src.services.translation import TranslationService, cache_source

pytestmark = pytest.mark.asyncio

NEBIUS = TranslationBackend.NEBIUS
TARGETS = ["en", "de"]


class FakeStore:
    """translation_jobs in memory, with the same once-only finish."""

    def __init__(self) -> None:
        self.jobs = {}
        self.claims: dict[str, str] = {}
        self.fail_claims = 0
        self.fail_finish = 0
        self.heartbeats = 0
        self.heartbeat_answers: list = []
        self.purged = 0

    async def insert(self, job):
        job = replace(job, created_at=dt.datetime.now(dt.UTC))
        self.jobs[job.job_id] = job
        return replace(job)

    async def get(self, job_id):
        job = self.jobs.get(job_id)
        return replace(job) if job else None

    async def claim_realtime(self, worker, _lease_s, limit):
        if self.fail_claims:
            self.fail_claims -= 1
            raise OSError("store restarting")
        out = []
        for job in self.jobs.values():
            if (job.mode == "realtime" and job.status in (QUEUED, RUNNING)
                    and job.job_id not in self.claims):
                self.claims[job.job_id] = worker
                job.status = RUNNING
                out.append(replace(job))
        return out[:limit]

    async def heartbeat(self, job_id, worker):
        self.heartbeats += 1
        if self.heartbeat_answers:
            answer = self.heartbeat_answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        return self.claims.get(job_id) == worker

    async def release(self, job_id, worker):
        if self.claims.get(job_id) == worker:
            del self.claims[job_id]

    async def set_running(self, job_id):
        if self.jobs[job_id].status == QUEUED:
            self.jobs[job_id].status = RUNNING

    async def finish(self, job_id, *, status, results, cost_usd, error=None):
        if self.fail_finish:
            self.fail_finish -= 1
            raise OSError("store gone")
        job = self.jobs[job_id]
        if job.status in (COMPLETED, FAILED):
            return None
        job.status, job.results, job.cost_usd, job.error = status, results, cost_usd, error
        job.completed_at = dt.datetime.now(dt.UTC)
        return replace(job)

    async def purge(self, _days):
        return self.purged


class FakeCache:
    def __init__(self) -> None:
        self.rows: dict[tuple, dict] = {}
        self.puts: list[tuple] = []

    async def get_translations(self, text, source_lang, targets, backend):
        held = self.rows.get((text, source_lang, backend), {})
        return {t: held[t] for t in targets if t in held}

    async def put_translations(self, text, source_lang, backend, translations):
        self.puts.append((text, source_lang, backend, dict(translations)))
        self.rows.setdefault((text, source_lang, backend), {}).update(translations)


class FakeTranslation:
    """TranslationService's surface as jobs use it. ``script`` maps a text
    to what translating it does: an exception, or a list of outcomes."""

    cache_backend = TranslationService.cache_backend      # the real key rule

    def __init__(self, nebius=None, cap_usd: float = 10.0, script=None) -> None:
        self.cache = FakeCache()
        self.nebius = nebius
        self.nebius_spend_cap = SpendCap(daily_cap_usd=cap_usd)
        self.mistral_spend_cap = SpendCap(daily_cap_usd=cap_usd)
        self.script = script or {}
        self.calls: list[str] = []
        self.summarized: list[tuple] = []

    async def summarize(  # pylint: disable=too-many-arguments
            self, text, source_lang, targets, backend, *, max_chars, about):
        self.summarized.append((text, source_lang, list(targets), max_chars, about))
        return SummaryResult({source_lang: f"summary of {text}",
                              **{t: f"summary of {text}@{t}" for t in targets if t != source_lang}},
                             source_lang, backend, cached=False, cost_usd=0.0007)

    async def translate(self, *, text, source_lang, targets, backend):
        self.calls.append(text)
        outcome = self.script.get(text)
        if isinstance(outcome, list):
            outcome = outcome.pop(0) if outcome else None
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, asyncio.Event):
            await outcome.wait()
        assert source_lang
        return TranslationResult({t: f"{text}@{t}" for t in targets}, backend,
                                 frozenset(), cost_usd=0.001)


def _completion(translations: dict, prompt: int = 100, completion: int = 200) -> dict:
    return {"choices": [{"message": {"content": json.dumps(translations)}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def _tagged_completion(translations: dict, prompt: int = 100, completion: int = 200) -> dict:
    tagged = "\n".join(f"<{lang}>{text}</{lang}>" for lang, text in translations.items())
    return {"choices": [{"message": {"content": tagged}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


class ProviderApi:  # pylint: disable=too-many-instance-attributes
    """Nebius files and batches, and the real-time calls a job makes besides,
    scripted per test."""

    def __init__(self, create_status: int = 200, state: str = "completed",
                 output=None, errors=None, batch_errors=None) -> None:
        self.create_status = create_status
        self.state = state
        self.output = output or []
        self.errors = errors or []
        self.batch_errors = batch_errors
        self.uploads: list[str] = []
        self.creates = 0
        self.chat_answers: list[dict] = []      # real-time calls, answered in order
        self.chats: list[dict] = []

    # One return per endpoint it plays, as a router would have.
    async def __call__(  # pylint: disable=too-many-return-statements
        self, request: httpx.Request,
    ) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "POST" and path.endswith("/files"):
            self.uploads.append(request.content.decode("utf-8"))
            return httpx.Response(200, json={"id": "file-in"})
        if method == "POST" and path.endswith("/batches"):
            self.creates += 1
            if self.create_status != 200:
                return httpx.Response(self.create_status, json={"detail": "unavailable"})
            return httpx.Response(200, json={"id": "batch-1", "status": "validating"})
        if method == "GET" and path.endswith("/batches/batch-1"):
            body = {"id": "batch-1", "status": self.state, "errors": self.batch_errors}
            if self.output:
                body["output_file_id"] = "file-out"
            if self.errors:
                body["error_file_id"] = "file-err"
            return httpx.Response(200, json=body)
        if method == "GET" and path.endswith("/content"):
            lines = self.output if "file-out" in path else self.errors
            return httpx.Response(200, text="\n".join(json.dumps(line) for line in lines))
        if method == "DELETE":
            return httpx.Response(200, json={})
        if method == "POST" and path.endswith("/chat/completions") and self.chat_answers:
            self.chats.append(json.loads(request.content))
            return httpx.Response(200, json=self.chat_answers.pop(0))
        return httpx.Response(404, json={"detail": path})


def _nebius(api: ProviderApi) -> NebiusBackend:
    client = httpx.AsyncClient(transport=httpx.MockTransport(api),
                               base_url="https://api.studio.nebius.com/v1")
    return NebiusBackend(
        api_url="https://api.studio.nebius.com/v1", api_key="test-key-never-used",
        chat_model="google/gemma-3-27b-it", timeout_s=5.0, max_retries=0,
        price_input_per_mtok=0.10, price_output_per_mtok=0.30, client=client)


def _items(*texts: str) -> list[JobItem]:
    return [JobItem(id=t, text=t, source_lang="pl", targets=list(TARGETS)) for t in texts]


def _jobs(translation, store=None, **kw) -> tuple[TranslationJobs, FakeStore]:
    store = store or FakeStore()
    kw.setdefault("backoff_s", 0.0)
    return TranslationJobs(store, translation, **kw), store


async def _run_claimed(jobs: TranslationJobs) -> None:
    await jobs._claim()                                     # pylint: disable=protected-access
    await asyncio.gather(*jobs._running.values())           # pylint: disable=protected-access


# ── Realtime ─────────────────────────────────────────────────────


async def test_a_realtime_job_translates_every_item_in_submission_order():
    translation = FakeTranslation()
    jobs, store = _jobs(translation, mode="realtime")
    job = await jobs.submit(_items("b", "a", "c"), NEBIUS)
    assert (job.mode, job.status, job.n_items) == ("realtime", QUEUED, 3)
    await _run_claimed(jobs)
    done = await jobs.status(job.job_id)
    assert done.status == COMPLETED
    assert [r.id for r in done.results] == ["b", "a", "c"]
    assert done.results[0].translations == {"en": "b@en", "de": "b@de"}
    assert done.cost_usd == pytest.approx(0.003)
    assert store.claims == {job.job_id: jobs.worker_id}


async def test_each_item_failure_says_whether_sending_it_again_can_help():
    translation = FakeTranslation(script={
        "spent": SpendCapExceeded("cap"),
        "unset": BackendUnavailable("no key"),
        "bad": NebiusError("missing target(s): ['de']"),
        "flaky": [CircuitOpen("open"), None],
        "down": NebiusTransientError("502"),
    })
    jobs, _ = _jobs(translation, mode="realtime")
    job = await jobs.submit(_items("spent", "unset", "bad", "flaky", "down"), NEBIUS)
    await _run_claimed(jobs)
    got = {r.id: r for r in (await jobs.status(job.job_id)).results}
    assert got["spent"].retryable and "SpendCapExceeded" in got["spent"].error
    assert got["unset"].retryable
    assert not got["bad"].retryable and "missing target" in got["bad"].error
    assert got["flaky"].translations and got["flaky"].error is None
    assert got["down"].retryable and "NebiusTransientError" in got["down"].error
    assert translation.calls.count("down") == 4        # waited out, then handed back


async def test_a_realtime_job_is_refused_once_the_budget_is_spent():
    translation = FakeTranslation(cap_usd=1.0)
    await translation.nebius_spend_cap.reserve(1.0)
    jobs, _ = _jobs(translation, mode="realtime")
    with pytest.raises(JobRefused):
        await jobs.submit(_items("a"), NEBIUS)


async def test_the_worker_runs_submitted_jobs_and_survives_a_failing_pass():
    translation = FakeTranslation()
    jobs, store = _jobs(translation, mode="realtime", poll_s=0.01)
    store.fail_claims = 1
    store.purged = 2
    jobs.start()
    job = await jobs.submit(_items("a", "b"), TranslationBackend.MISTRAL)
    for _ in range(200):
        if store.jobs[job.job_id].status == COMPLETED:
            break
        await asyncio.sleep(0.01)
    await jobs.stop()
    assert store.jobs[job.job_id].status == COMPLETED


async def test_stopping_hands_running_jobs_back_for_another_pod():
    gate = asyncio.Event()
    translation = FakeTranslation(script={"slow": gate})
    jobs, store = _jobs(translation, mode="realtime")
    job = await jobs.submit(_items("slow"), NEBIUS)
    await jobs._claim()                                     # pylint: disable=protected-access
    await asyncio.sleep(0)
    assert store.claims == {job.job_id: jobs.worker_id}
    await jobs.stop()
    assert not store.claims and store.jobs[job.job_id].status == RUNNING


async def test_a_job_whose_result_cannot_be_stored_is_handed_back():
    jobs, store = _jobs(FakeTranslation(), mode="realtime")
    store.fail_finish = 1
    job = await jobs.submit(_items("a"), NEBIUS)
    await _run_claimed(jobs)
    assert job.job_id not in store.claims
    assert store.jobs[job.job_id].status == RUNNING


async def test_the_claim_is_renewed_while_a_long_job_runs():
    gate = asyncio.Event()
    jobs, store = _jobs(FakeTranslation(script={"slow": gate}), mode="realtime", lease_s=0.03)
    store.heartbeat_answers = [OSError("blip"), False]
    job = await jobs.submit(_items("slow"), NEBIUS)
    await jobs._claim()                                     # pylint: disable=protected-access
    await asyncio.sleep(0.08)
    gate.set()
    await asyncio.gather(*jobs._running.values())           # pylint: disable=protected-access
    assert store.heartbeats >= 3
    assert store.jobs[job.job_id].status == COMPLETED


# ── Provider batches ─────────────────────────────────────────────


def _provider_output() -> tuple[list, list]:
    output = [
        {"custom_id": "b", "response": {"status_code": 200,
                                        "body": _completion({"en": "B-en", "de": "B-de"})}},
        {"custom_id": "c", "response": {"status_code": 200,
                                        "body": _completion({"en": "C-en"}, 100, 100)}},
        {"custom_id": "not-ours", "response": {"status_code": 200, "body": _completion({})}},
    ]
    errors = [{"custom_id": "d", "response": {"status_code": 429, "body": {"error": "slow"}},
               "error": None},
              {"custom_id": "f", "response": None,
               "error": {"code": "invalid_request", "message": "context length exceeded"}},
              {"custom_id": "g", "response": None,
               "error": {"code": "batch_expired", "message": "not reached in time"}}]
    return output, errors


async def test_a_provider_job_sends_only_what_the_cache_cannot_answer():
    output, errors = _provider_output()
    api = ProviderApi(state="in_progress", output=output, errors=errors)
    api.chat_answers = [_tagged_completion({"en": "C-en", "de": "C-de"}, 300, 60)]
    translation = FakeTranslation(nebius=_nebius(api))
    translation.cache.rows[("a", "pl", "nebius")] = {"en": "A-en", "de": "A-de"}
    jobs, _ = _jobs(translation, mode="provider")
    job = await jobs.submit(_items("a", "b", "c", "d", "e", "f", "g"), NEBIUS)
    assert (job.mode, job.status) == ("provider", QUEUED)
    assert '"custom_id": "b"' in api.uploads[0] and '"custom_id": "a"' not in api.uploads[0]
    reserved = translation.nebius_spend_cap.spent_usd
    assert reserved > 0

    assert (await jobs.status(job.job_id)).status == RUNNING
    api.state = "completed"
    done = await jobs.status(job.job_id)
    got = {r.id: r for r in done.results}
    assert done.status == COMPLETED and [r.id for r in done.results] == list("abcdefg")
    assert got["a"].translations == {"en": "A-en", "de": "A-de"} and got["a"].cost_usd == 0
    assert got["b"].translations == {"en": "B-en", "de": "B-de"}
    # c's JSON came back without "de": asked once more, now and tagged.
    assert got["c"].translations == {"en": "C-en", "de": "C-de"} and not got["c"].error
    assert len(api.chats) == 1 and "response_format" not in api.chats[0]
    assert "<de>...</de>" in api.chats[0]["messages"][0]["content"]
    assert got["d"].retryable and "status=429" in got["d"].error
    assert got["e"].retryable and "without this item" in got["e"].error
    assert not got["f"].retryable and "context length" in got["f"].error
    assert got["g"].retryable
    batch_cost = ((100 * 0.05 + 200 * 0.15) + (100 * 0.05 + 100 * 0.15)) / 1_000_000
    retry_cost = (300 * 0.10 + 60 * 0.30) / 1_000_000          # real-time prices
    assert got["c"].cost_usd == pytest.approx((100 * 0.05 + 100 * 0.15) / 1_000_000 + retry_cost)
    assert done.cost_usd == pytest.approx(batch_cost + retry_cost)
    assert translation.nebius_spend_cap.spent_usd == pytest.approx(batch_cost + retry_cost)
    assert translation.cache.puts == [("b", "pl", "nebius", {"en": "B-en", "de": "B-de"}),
                                      ("c", "pl", "nebius", {"en": "C-en", "de": "C-de"})]
    again = await jobs.status(job.job_id)                   # final: the provider is not asked
    assert again.status == COMPLETED and api.creates == 1


async def test_an_unreadable_line_whose_tagged_retry_fails_too_is_an_error_with_its_cost():
    """Nothing half-translated is passed off as done: the item fails, final
    unless the retry failed on the provider's side, and keeps what the
    broken line cost."""
    output = [{"custom_id": "a", "response": {"status_code": 200,
                                              "body": _completion({"en": "A-en"}, 100, 100)}},
              {"custom_id": "b", "response": {"status_code": 200,
                                              "body": _completion({"en": "B-en"}, 100, 100)}}]
    api = ProviderApi(output=output)
    api.chat_answers = [_tagged_completion({"en": "A-en"})]     # "de" missing again; b gets a 404
    translation = FakeTranslation(nebius=_nebius(api))
    jobs, _ = _jobs(translation, mode="provider")
    job = await jobs.submit(_items("a", "b"), NEBIUS)
    done = await jobs.status(job.job_id)
    got = {r.id: r for r in done.results}
    assert "tagged response" in got["a"].error and not got["a"].retryable
    assert "status=404" in got["b"].error and not got["b"].retryable
    assert got["a"].cost_usd == pytest.approx((100 * 0.05 + 100 * 0.15) / 1_000_000)
    assert not translation.cache.puts


async def test_a_job_the_cache_answers_in_full_never_reaches_the_provider():
    api = ProviderApi()
    translation = FakeTranslation(nebius=_nebius(api))
    translation.cache.rows[("a", "pl", "nebius")] = {"en": "A-en", "de": "A-de"}
    jobs, _ = _jobs(translation, mode="provider")
    job = await jobs.submit(_items("a"), NEBIUS)
    assert job.status == COMPLETED and job.results[0].translations["de"] == "A-de"
    assert not api.uploads


async def test_a_batch_the_provider_lost_fails_the_job_and_frees_the_budget():
    api = ProviderApi(state="expired", batch_errors={"data": [{"message": "timed out"}]})
    translation = FakeTranslation(nebius=_nebius(api))
    jobs, _ = _jobs(translation, mode="provider")
    job = await jobs.submit(_items("a"), NEBIUS)
    done = await jobs.status(job.job_id)
    assert done.status == FAILED and "expired" in done.error and done.results is None
    assert translation.nebius_spend_cap.spent_usd == 0


async def test_a_reservation_from_an_earlier_day_is_not_settled_today():
    output, _ = _provider_output()
    translation = FakeTranslation(nebius=_nebius(ProviderApi(output=output)))
    jobs, store = _jobs(translation, mode="provider")
    job = await jobs.submit(_items("b"), NEBIUS)
    reserved = translation.nebius_spend_cap.spent_usd
    store.jobs[job.job_id].reserved_on = dt.date(2026, 1, 1)
    assert (await jobs.status(job.job_id)).status == COMPLETED
    assert translation.nebius_spend_cap.spent_usd == reserved


async def test_a_batch_over_budget_is_refused_before_anything_is_uploaded():
    api = ProviderApi()
    translation = FakeTranslation(nebius=_nebius(api), cap_usd=1e-9)
    jobs, _ = _jobs(translation, mode="provider")
    with pytest.raises(JobRefused):
        await jobs.submit(_items("a"), NEBIUS)
    assert not api.uploads


async def test_a_refused_batch_in_provider_mode_is_the_callers_to_handle():
    api = ProviderApi(create_status=403)
    translation = FakeTranslation(nebius=_nebius(api))
    jobs, _ = _jobs(translation, mode="provider")
    with pytest.raises(NebiusBatchRefused):
        await jobs.submit(_items("a"), NEBIUS)
    assert translation.nebius_spend_cap.spent_usd == 0


async def test_auto_runs_the_job_here_when_the_provider_refuses_and_asks_again_later():
    api = ProviderApi(create_status=403)
    now = [0.0]
    translation = FakeTranslation(nebius=_nebius(api))
    jobs, _ = _jobs(translation, mode="auto", provider_retry_s=900, clock=lambda: now[0])
    first = await jobs.submit(_items("a"), NEBIUS)
    second = await jobs.submit(_items("b"), NEBIUS)
    assert (first.mode, second.mode, api.creates) == ("realtime", "realtime", 1)
    now[0] = 901.0
    api.create_status = 200
    third = await jobs.submit(_items("c"), NEBIUS)
    assert (third.mode, api.creates) == ("provider", 2)


async def test_auto_without_nebius_runs_in_real_time():
    jobs, _ = _jobs(FakeTranslation(nebius=None), mode="auto")
    assert (await jobs.submit(_items("a"), NEBIUS)).mode == "realtime"


async def test_a_provider_hiccup_on_submission_frees_the_reservation():
    translation = FakeTranslation(nebius=_nebius(ProviderApi(create_status=502)))
    jobs, _ = _jobs(translation, mode="auto")
    with pytest.raises(NebiusTransientError):
        await jobs.submit(_items("a"), NEBIUS)
    assert translation.nebius_spend_cap.spent_usd == 0


# ── Validation and lookups ───────────────────────────────────────


@pytest.mark.parametrize("items, mode, backend", [
    ([], None, NEBIUS),
    (_items("a", "a"), None, NEBIUS),
    ([JobItem(id="x", text="  ", source_lang="pl", targets=["en"])], None, NEBIUS),
    (_items("a"), "sometime", NEBIUS),
    (_items("a"), "provider", TranslationBackend.MISTRAL),
])
async def test_malformed_jobs_are_rejected(items, mode, backend):
    jobs, _ = _jobs(FakeTranslation(nebius=_nebius(ProviderApi())))
    with pytest.raises(ValueError):
        await jobs.submit(items, backend, mode)


async def test_provider_mode_needs_nebius_configured():
    jobs, _ = _jobs(FakeTranslation(nebius=None), mode="provider")
    with pytest.raises(BackendUnavailable):
        await jobs.submit(_items("a"), NEBIUS)


async def test_an_unknown_job_is_none():
    jobs, _ = _jobs(FakeTranslation())
    assert await jobs.status("nope") is None


async def test_a_provider_job_caches_under_the_model_that_made_it():
    """A translation is reused only while its model is configured: the
    legacy model's cache is not served for another model, and the new
    model's answers are cached under its own key."""
    output, _ = _provider_output()
    nebius = _nebius(ProviderApi(output=output))
    nebius.chat_model = "deepseek-ai/DeepSeek-V4-Flash-0731"
    translation = FakeTranslation(nebius=nebius)
    translation.cache.rows[("b", cache_source("pl"), "nebius")] = {"en": "old", "de": "alt"}
    jobs, _ = _jobs(translation, mode="provider")
    job = await jobs.submit(_items("b"), NEBIUS)
    assert job.status == QUEUED                       # the legacy entry did not answer it
    await jobs.status(job.job_id)
    key = "nebius:deepseek-ai/DeepSeek-V4-Flash-0731"
    assert translation.cache.puts == [("b", cache_source("pl"), key, {"en": "B-en", "de": "B-de"})]


# ── Summaries ────────────────────────────────────────────────────


def _summary_item(**over) -> JobItem:
    values = {"id": "goals-1", "text": "Die Brauwirtschaft vertreten.", "source_lang": "de",
              "targets": ["en", "fr"], "task": "summarize",
              "about": "what the organisation lobbies for"}
    values.update(over)
    return JobItem(**values)


async def test_a_summary_job_is_made_here_with_a_summary_per_language():
    """A registrant's goals summarised: never sent as a provider batch (a
    summary is written, then translated), and the item's result holds the
    summary in its own language as well as in the others."""
    api = ProviderApi()
    translation = FakeTranslation(nebius=_nebius(api))
    jobs, store = _jobs(translation, mode="auto")
    job = await jobs.submit([_summary_item()], NEBIUS)
    assert job.mode == "realtime" and not api.uploads
    await _run_claimed(jobs)
    done = store.jobs[job.job_id]
    assert done.status == COMPLETED
    (result,) = done.results
    assert result.translations == {"de": "summary of Die Brauwirtschaft vertreten.",
                                   "en": "summary of Die Brauwirtschaft vertreten.@en",
                                   "fr": "summary of Die Brauwirtschaft vertreten.@fr"}
    assert result.cost_usd == pytest.approx(0.0007)
    assert translation.summarized == [("Die Brauwirtschaft vertreten.", "de", ["en", "fr"], 280,
                                       "what the organisation lobbies for")]


async def test_a_summary_cannot_be_forced_into_a_provider_batch():
    jobs, _store = _jobs(FakeTranslation(nebius=_nebius(ProviderApi())), mode="auto")
    with pytest.raises(ValueError, match="summaries"):
        await jobs.submit([_summary_item()], NEBIUS, mode="provider")
    with pytest.raises(ValueError, match="unknown task"):
        await jobs.submit([_summary_item(task="paraphrase")], NEBIUS)
