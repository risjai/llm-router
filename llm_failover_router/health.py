"""Per-provider health state machine based on consecutive outcomes.

A provider starts HEALTHY. It trips to UNHEALTHY after ``x`` consecutive
failures, and recovers to HEALTHY after ``y`` consecutive successes. Any
outcome of the opposite kind resets the relevant streak — the counts must be
*consecutive*. The machine is pure in-memory logic; the router owns all locking.
"""
from __future__ import annotations

from enum import Enum


class HealthState(Enum):
    """Whether a provider is currently considered healthy."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class ProviderHealth:
    """Tracks one provider's health via consecutive success/failure counts.

    Args:
        x: Consecutive failures required to trip HEALTHY -> UNHEALTHY.
        y: Consecutive successes required to recover UNHEALTHY -> HEALTHY.

    ``record_success`` / ``record_failure`` return ``True`` exactly when the
    call causes a state transition, so the router knows when to recompute
    routing weights.
    """

    def __init__(self, x: int, y: int) -> None:
        self._x = x
        self._y = y
        self._state = HealthState.HEALTHY
        # Only the streak relevant to the current state matters; we keep one
        # counter and reinterpret it per state to avoid stale bookkeeping.
        self._consecutive = 0

    @property
    def state(self) -> HealthState:
        """Current health state (read-only)."""
        return self._state

    def is_healthy(self) -> bool:
        return self._state is HealthState.HEALTHY

    def record_success(self) -> bool:
        if self._state is HealthState.HEALTHY:
            # Successes while healthy simply clear any partial failure streak.
            self._consecutive = 0
            return False
        # UNHEALTHY: count consecutive successes toward recovery.
        self._consecutive += 1
        if self._consecutive >= self._y:
            self._state = HealthState.HEALTHY
            self._consecutive = 0
            return True
        return False

    def record_failure(self) -> bool:
        if self.state is HealthState.UNHEALTHY:
            # Failures while unhealthy clear any partial recovery streak.
            self._consecutive = 0
            return False
        # HEALTHY: count consecutive failures toward tripping.
        self._consecutive += 1
        if self._consecutive >= self._x:
            self._state = HealthState.UNHEALTHY
            self._consecutive = 0
            return True
        return False
