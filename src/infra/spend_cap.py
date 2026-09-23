"""Daily spend cap (USD). Resets at UTC midnight. Hard 429 when exceeded.

Tracks spend per (pod, date) in-process. Cross-pod accounting would require a
shared store; for v1 the cap is per-pod — accept that N replicas × cap is the
effective daily ceiling, set cap accordingly.

Lifecycle: `reserve(est)` on entry, then `finalize(est, actual)` with the billed
amount after the backend returns, or `release(est)` if the call failed.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from src.domain.models import SpendCapExceeded


@dataclass
class SpendCap:
    daily_cap_usd: float
    _today: date = field(default_factory=lambda: datetime.now(UTC).date())
    _spent_usd: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _roll_over_if_new_day_locked(self) -> None:
        today = datetime.now(UTC).date()
        if today != self._today:
            self._today = today
            self._spent_usd = 0.0

    async def reserve(self, estimated_usd: float) -> None:
        async with self._lock:
            self._roll_over_if_new_day_locked()
            if self._spent_usd + estimated_usd > self.daily_cap_usd:
                raise SpendCapExceeded(
                    f"daily cap ${self.daily_cap_usd:.2f} would be exceeded "
                    f"(spent=${self._spent_usd:.2f}, request≈${estimated_usd:.4f})"
                )
            self._spent_usd += estimated_usd

    @property
    def spent_usd(self) -> float:
        """What has been charged today. Read-only, and read without the lock:
        a caller deciding whether to keep going wants the current figure, not
        a serialised one."""
        return self._spent_usd

    async def finalize(self, estimated_usd: float, actual_usd: float) -> None:
        """Correct the reservation to match the actual billed amount."""
        delta = actual_usd - estimated_usd
        async with self._lock:
            self._roll_over_if_new_day_locked()
            self._spent_usd = max(0.0, self._spent_usd + delta)

    async def release(self, estimated_usd: float) -> None:
        """Roll back a reservation when the backend call failed."""
        async with self._lock:
            self._roll_over_if_new_day_locked()
            self._spent_usd = max(0.0, self._spent_usd - estimated_usd)

    async def spent(self) -> float:
        async with self._lock:
            self._roll_over_if_new_day_locked()
            return self._spent_usd
