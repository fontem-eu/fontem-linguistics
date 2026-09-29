"""POST/GET /translate/jobs: the HTTP contract over the jobs service."""
from __future__ import annotations

import datetime as dt

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.deps import Services
from src.api.routes import router
from src.backends.nebius import NebiusBatchRefused, NebiusError, NebiusTransientError
from src.cache.jobs import ItemResult, JobRecord
from src.domain.models import BackendUnavailable
from src.services.jobs import JobRefused


class FakeJobs:
    def __init__(self, submit_raises=None, status_raises=None, record=None) -> None:
        self.submit_raises = submit_raises
        self.status_raises = status_raises
        self.record = record
        self.submitted = []

    async def submit(self, items, backend, mode):
        self.submitted.append((items, backend, mode))
        if self.submit_raises:
            raise self.submit_raises
        return JobRecord(job_id="j1", mode="realtime", backend=backend.value,
                         status="queued", items=items)

    async def status(self, job_id):
        if self.status_raises:
            raise self.status_raises
        return self.record if job_id == "j1" else None


def _client(jobs) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.services = Services(translation=None, embedding=None, jobs=jobs)
    return TestClient(app)


_BODY = {"items": [{"id": "1", "text": "Roboty budowlane", "source_lang": "pl",
                    "targets": ["en", "de"]}]}


def test_a_job_is_accepted_with_its_id_and_mode():
    jobs = FakeJobs()
    resp = _client(jobs).post("/translate/jobs", json={**_BODY, "mode": "auto"})
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "j1" and resp.json()["results"] is None
    items, backend, mode = jobs.submitted[0]
    assert items[0].text == "Roboty budowlane" and backend.value == "nebius" and mode == "auto"


def test_a_completed_job_returns_every_result():
    record = JobRecord(
        job_id="j1", mode="provider", backend="nebius", status="completed", items=[],
        results=[ItemResult("1", {"en": "Construction works"}, cost_usd=0.0001),
                 ItemResult("2", error="provider status=429", retryable=True)],
        cost_usd=0.0001, created_at=dt.datetime(2026, 9, 29, tzinfo=dt.UTC),
        completed_at=dt.datetime(2026, 9, 29, 1, tzinfo=dt.UTC))
    body = _client(FakeJobs(record=record)).get("/translate/jobs/j1").json()
    assert body["status"] == "completed" and body["mode"] == "provider"
    assert body["results"][0] == {"id": "1", "translations": {"en": "Construction works"},
                                  "error": None, "retryable": False, "cost_usd": 0.0001}
    assert body["results"][1]["retryable"] is True


def test_an_unknown_job_is_404():
    assert _client(FakeJobs()).get("/translate/jobs/nope").status_code == 404


@pytest.mark.parametrize("exc, status", [
    (ValueError("dup ids"), 400),
    (JobRefused("cap"), 429),
    (NebiusBatchRefused("403"), 503),
    (BackendUnavailable("no key"), 503),
    (NebiusTransientError("502"), 502),
    (NebiusError("400"), 502),
])
def test_submission_failures_map_to_what_the_caller_should_do(exc, status):
    assert _client(FakeJobs(submit_raises=exc)).post(
        "/translate/jobs", json=_BODY).status_code == status


@pytest.mark.parametrize("exc, status", [
    (BackendUnavailable("no key"), 503),
    (NebiusTransientError("502"), 502),
])
def test_a_status_the_provider_cannot_answer_is_retried_later(exc, status):
    assert _client(FakeJobs(status_raises=exc)).get("/translate/jobs/j1").status_code == status


def test_malformed_submissions_never_reach_the_service():
    jobs = FakeJobs()
    client = _client(jobs)
    assert client.post("/translate/jobs", json={"items": []}).status_code == 422
    assert client.post("/translate/jobs", json={**_BODY, "mode": "later"}).status_code == 422
    assert not jobs.submitted


def test_without_a_jobs_service_the_endpoints_say_so():
    client = _client(None)
    assert client.post("/translate/jobs", json=_BODY).status_code == 503
    assert client.get("/translate/jobs/j1").status_code == 503
