"""Tests for compute_weights (pure) and the WeightedSelector (SWRR)."""
from collections import Counter

import pytest

from llm_failover_router.health import HealthState
from llm_failover_router.selector import compute_weights, WeightedSelector

H = HealthState.HEALTHY
U = HealthState.UNHEALTHY


def test_all_healthy_all_to_primary():
    weights, reject = compute_weights([H, H, H], p=10)
    assert weights == [100, 0, 0]
    assert reject == 0


def test_primary_unhealthy_bulk_to_secondary():
    # p to the unhealthy primary; the rest to the highest-priority healthy one.
    weights, reject = compute_weights([U, H, H], p=10)
    assert weights == [10, 90, 0]
    assert reject == 0


def test_two_unhealthy_bulk_to_third():
    weights, reject = compute_weights([U, U, H], p=10)
    assert weights == [10, 10, 80]
    assert reject == 0


def test_all_unhealthy_each_probes_rest_rejected():
    weights, reject = compute_weights([U, U, U], p=10)
    assert weights == [10, 10, 10]
    assert reject == 70


def test_recovering_bulk_returns_to_primary():
    # Primary healthy again, secondary still unhealthy: primary takes the bulk.
    weights, reject = compute_weights([H, U, H], p=10)
    assert weights == [90, 10, 0]
    assert reject == 0


def test_weights_always_sum_to_100():
    for states in ([H, H], [U, H], [H, U], [U, U]):
        weights, reject = compute_weights(states, p=25)
        assert sum(weights) + reject == 100


def test_swrr_exact_distribution_over_window():
    # [70, 30] over 100 picks must be exactly 70 / 30 — not merely in expectation.
    sel = WeightedSelector([70, 30])
    counts = Counter(sel.pick() for _ in range(100))
    assert counts[0] == 70
    assert counts[1] == 30


def test_swrr_zero_weight_never_picked():
    sel = WeightedSelector([100, 0, 0])
    picks = {sel.pick() for _ in range(50)}
    assert picks == {0}


def test_swrr_is_smooth_not_bursty():
    # SWRR interleaves picks; the minority target should never appear in a long
    # unbroken run. For [70,30], no 3 consecutive picks are all index 1.
    sel = WeightedSelector([70, 30])
    seq = [sel.pick() for _ in range(100)]
    for i in range(len(seq) - 2):
        assert not (seq[i] == seq[i + 1] == seq[i + 2] == 1)


def test_set_weights_resets_epoch():
    sel = WeightedSelector([50, 50])
    sel.pick()
    sel.set_weights([100, 0])
    picks = {sel.pick() for _ in range(20)}
    assert picks == {0}


def test_swrr_three_way_exact():
    sel = WeightedSelector([10, 10, 80])
    counts = Counter(sel.pick() for _ in range(100))
    assert counts[0] == 10
    assert counts[1] == 10
    assert counts[2] == 80
