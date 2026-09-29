"""The app as the pod starts it: the real lifespan, with only I/O faked.

Every other component test hands the routes a ready-made `app.state`, so
none of them runs `lifespan`. A startup that could not assemble its own
backends passed all of them and then crashlooped in production (v61dc4eb):
the decorator meant for `lifespan` had ended up on the helper beside it.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api.app as app_module
from src.api.app import build_app
from src.infra.config import Settings


class _FakeCache:
    """Postgres is the one dependency startup cannot reach in a unit run."""

    closed = False
    pool = None

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(name="cache")
def _cache(monkeypatch):
    cache = _FakeCache()

    async def connect(**_kw):
        return cache

    async def no_schema(_cache):
        return None

    async def no_weights(_self, _text):
        raise RuntimeError("no model weights in unit tests")

    monkeypatch.setattr(app_module.PostgresCache, "connect", staticmethod(connect))
    monkeypatch.setattr(app_module, "_ensure_schema", no_schema)
    # Startup warms the local encoders and tolerates a failure; failing fast
    # keeps it from reaching for weights.
    monkeypatch.setattr(app_module.LabseLocalBackend, "embed", no_weights)
    monkeypatch.setattr(app_module.MinilmLocalBackend, "embed", no_weights)
    return cache


class _FakeJobStore:
    """What the job worker asks of Postgres, answered in memory."""

    claims: list[str] = []

    def __init__(self, pool) -> None:
        self.pool = pool

    async def claim_realtime(self, worker, _lease_s, _limit):
        _FakeJobStore.claims.append(worker)
        return []

    async def purge(self, _days):
        return 0


def test_the_job_worker_runs_for_the_life_of_the_app(cache, monkeypatch):
    monkeypatch.setattr(app_module, "JobStore", _FakeJobStore)
    _FakeJobStore.claims = []
    settings = Settings(database_url="postgresql://unused", nebius_api_key="test-nebius")
    with TestClient(build_app(settings)) as client:
        assert client.get("/healthz").status_code == 200
        jobs = client.app.state.services.jobs
        assert jobs is not None and jobs.mode == "auto"
    assert _FakeJobStore.claims == [jobs.worker_id]
    assert jobs._worker is not None and jobs._worker.done()  # pylint: disable=protected-access
    assert cache.closed


def test_the_app_starts_with_both_paid_providers(cache):
    settings = Settings(database_url="postgresql://unused",
                        mistral_api_key="test-mistral", nebius_api_key="test-nebius")
    with TestClient(build_app(settings)) as client:
        translation = client.app.state.services.translation
        assert translation.mistral is not None and translation.nebius is not None
    assert cache.closed


def test_a_provider_without_a_key_is_absent_not_broken(cache):
    settings = Settings(database_url="postgresql://unused",
                        mistral_api_key=None, nebius_api_key=None)
    with TestClient(build_app(settings)) as client:
        translation = client.app.state.services.translation
        assert translation.mistral is None and translation.nebius is None
    assert cache.closed
