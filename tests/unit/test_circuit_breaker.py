"""Circuit breaker: closed → open transition, cooldown → half-open, success → closed."""
from __future__ import annotations

import asyncio

import pytest

from src.domain.models import CircuitState
from src.infra.circuit_breaker import CircuitBreaker


pytestmark = pytest.mark.asyncio


async def test_starts_closed_and_allows():
    cb = CircuitBreaker()
    assert await cb.state() is CircuitState.CLOSED
    assert await cb.allow()


async def test_below_min_requests_does_not_open():
    cb = CircuitBreaker(failure_threshold=0.05, min_requests=20)
    for _ in range(5):
        await cb.record_failure()
    assert await cb.state() is CircuitState.CLOSED


async def test_opens_when_failure_rate_exceeds_threshold():
    cb = CircuitBreaker(failure_threshold=0.05, min_requests=20)
    for _ in range(18):
        await cb.record_success()
    for _ in range(3):
        await cb.record_failure()
    # 3 failures / 21 total = 14% > 5% threshold, and total ≥ min_requests
    assert await cb.state() is CircuitState.OPEN
    assert not await cb.allow()


async def test_cooldown_transitions_to_half_open_then_closed_on_success():
    cb = CircuitBreaker(
        failure_threshold=0.0, min_requests=1, cooldown_s=0.1,
    )
    await cb.record_failure()
    assert await cb.state() is CircuitState.OPEN

    await asyncio.sleep(0.15)
    assert await cb.state() is CircuitState.HALF_OPEN
    assert await cb.allow()

    await cb.record_success()
    assert await cb.state() is CircuitState.CLOSED


async def test_half_open_failure_trips_back_open():
    cb = CircuitBreaker(
        failure_threshold=0.0, min_requests=1, cooldown_s=0.1,
    )
    await cb.record_failure()
    await asyncio.sleep(0.15)
    assert await cb.state() is CircuitState.HALF_OPEN

    await cb.record_failure()
    assert await cb.state() is CircuitState.OPEN


async def test_old_events_pruned_outside_window():
    # With a tiny window, record a failure, wait past window, record successes
    # to dominate, confirm breaker stays closed (old failure pruned).
    cb = CircuitBreaker(failure_threshold=0.05, min_requests=5, window_s=1)
    await cb.record_failure()
    await asyncio.sleep(1.1)
    for _ in range(6):
        await cb.record_success()
    assert await cb.state() is CircuitState.CLOSED
