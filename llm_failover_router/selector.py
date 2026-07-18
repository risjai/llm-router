"""Routing weights and the deterministic distribution engine.

Two pieces:

* :func:`compute_weights` — the whole routing policy as one pure function of
  provider health. Every unhealthy provider keeps a ``p%`` probe stream (so it
  can earn its way back to healthy); the leftover "bulk" goes to the
  highest-priority healthy provider, or becomes the reject bucket when none are
  healthy.
* :class:`WeightedSelector` — Smoothed Weighted Round-Robin, which turns those
  integer weights into an exact, smoothly-interleaved sequence of picks.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from llm_failover_router.health import HealthState

TOTAL = 100


def compute_weights(
    states: Sequence[HealthState], p: int
) -> Tuple[List[int], int]:
    """Compute per-provider routing weights and the reject weight.

    Args:
        states: Health of each provider, in priority order (index 0 = primary).
        p: Integer probe percent given to each unhealthy provider.

    Returns:
        ``(weights, reject_weight)`` — integers summing to 100.
    """
    weights = [0] * len(states)
    num_unhealthy = 0
    for i, state in enumerate(states):
        if state is HealthState.UNHEALTHY:
            weights[i] = p
            num_unhealthy += 1

    bulk = TOTAL - p * num_unhealthy

    # The highest-priority healthy provider absorbs the bulk. If nobody is
    # healthy, the bulk is rejected (no provider is invoked for it).
    for i, state in enumerate(states):
        if state is HealthState.HEALTHY:
            weights[i] = bulk
            return weights, 0
    return weights, bulk


class WeightedSelector:
    """Smoothed Weighted Round-Robin selector (the nginx algorithm).

    Given integer weights, :meth:`pick` returns target indices such that, over
    any window, each target's share matches its weight *exactly* (not just in
    expectation) and picks are smoothly interleaved rather than bursty.

    Algorithm per pick:
        1. Add each target's weight to its running ``current``.
        2. Choose the target with the largest ``current``.
        3. Subtract the total weight from the winner's ``current``.

    A target with weight 0 never accumulates and is never chosen.
    """

    def __init__(self, weights: Sequence[int]) -> None:
        self._weights: List[int] = []
        self._current: List[int] = []
        self._total: int = 0
        self.set_weights(weights)

    def set_weights(self, weights: Sequence[int]) -> None:
        """Replace the weights and start a fresh selection epoch."""
        self._weights = list(weights)
        self._current = [0] * len(self._weights)
        self._total = sum(self._weights)

    def pick(self) -> int:
        """Return the index of the next selected target."""
        best = -1
        best_current = None
        for i, w in enumerate(self._weights):
            self._current[i] += w
            if best_current is None or self._current[i] > best_current:
                best_current = self._current[i]
                best = i
        self._current[best] -= self._total
        return best
