"""Sliding-window circuit breaker. Fail-loud: opens and stays open until cooldown.

Opens when (failures / total) > threshold within `window_s`, provided `min_requests`
samples have been recorded — protects against flakes on a near-idle service.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from src.domain.models import CircuitState
from src.infra.metrics import BREAKER_STATE


@dataclass
class _Event:
    ts: float
    ok: bool


@dataclass
class CircuitBreaker:
    failure_threshold: float = 0.05
    window_s: float = 60
    cooldown_s: float = 30
    min_requests: int = 20
    _events: deque[_Event] = field(default_factory=deque)
    _state: CircuitState = CircuitState.CLOSED
    _opened_at: float | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._events and self._events[0].ts < cutoff:
            self._events.popleft()

    async def state(self) -> CircuitState:
        async with self._lock:
            now = time.monotonic()
            if self._state is CircuitState.OPEN and self._opened_at is not None:
                if now - self._opened_at >= self.cooldown_s:
                    self._state = CircuitState.HALF_OPEN
                    self._publish()
            return self._state

    async def allow(self) -> bool:
        return (await self.state()) is not CircuitState.OPEN

    async def record_success(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._events.append(_Event(now, True))
            self._prune(now)
            if self._state is CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._opened_at = None
                self._publish()

    async def record_failure(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._events.append(_Event(now, False))
            self._prune(now)
            if self._state is CircuitState.HALF_OPEN:
                self._trip(now)
                return
            total = len(self._events)
            if total < self.min_requests:
                return
            fails = sum(1 for e in self._events if not e.ok)
            if fails / total > self.failure_threshold:
                self._trip(now)

    def _trip(self, now: float) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = now
        self._publish()

    def _publish(self) -> None:
        mapping = {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}
        BREAKER_STATE.set(mapping[self._state])
