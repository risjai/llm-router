"""Tests for RouterConfig validation and FailoverRouter behavior."""
from collections import Counter

import pytest

from llm_failover_router.router import FailoverRouter, RouterConfig
from llm_failover_router.provider import MockProvider
from llm_failover_router.errors import ProviderError, NoHealthyProviderError, ConfigError
from llm_failover_router.health import HealthState


def test_valid_config_passes():
    RouterConfig(p=10, x=3, y=2).validate(num_providers=3)  # no raise


@pytest.mark.parametrize(
    "p, x, y, n",
    [
        (0, 3, 2, 3),    # p < 1
        (10, 0, 2, 3),   # x < 1
        (10, 3, 0, 3),   # y < 1
        (10, 3, 2, 0),   # n < 1
        (40, 3, 2, 3),   # p*n = 120 > 100
    ],
)
def test_invalid_config_raises(p, x, y, n):
    with pytest.raises(ConfigError):
        RouterConfig(p=p, x=x, y=y).validate(num_providers=n)


def test_boundary_p_times_n_equals_100_is_valid():
    RouterConfig(p=25, x=1, y=1).validate(num_providers=4)  # 25*4 = 100, ok


def _route_many(router, n):
    """Route n requests, returning a Counter of which provider name served each
    (or 'REJECT' when NoHealthyProviderError was raised)."""
    served = Counter()
    for _ in range(n):
        try:
            served[router.route(None)] += 1
        except NoHealthyProviderError:
            served["REJECT"] += 1
        except ProviderError:
            served["ERROR"] += 1
    return served


def test_all_healthy_routes_everything_to_primary():
    primary = MockProvider("primary", response="primary")
    secondary = MockProvider("secondary", response="secondary")
    router = FailoverRouter([primary, secondary], RouterConfig(p=10, x=2, y=2))
    served = _route_many(router, 50)
    assert served["primary"] == 50
    assert secondary.call_count == 0


def test_traffic_shifts_after_x_failures_then_returns_after_y_successes():
    # Primary fails its first 2 calls (x=2 -> unhealthy), then succeeds forever.
    primary = MockProvider("primary", outcomes=[False, False], response="primary")
    secondary = MockProvider("secondary", response="secondary")
    router = FailoverRouter([primary, secondary], RouterConfig(p=10, x=2, y=2))

    # First two routed requests hit primary and fail (fail-fast: caller sees error).
    with pytest.raises(ProviderError):
        router.route(None)
    with pytest.raises(ProviderError):
        router.route(None)
    assert router.health_snapshot()[0] is HealthState.UNHEALTHY

    # Now degraded: weights [10, 90]. Over 100 requests, ~10 probe the primary
    # (which now succeeds) and drive its recovery after y=2 successes; the rest
    # go to secondary. After recovery, weights snap back to [100, 0].
    served = _route_many(router, 100)
    assert served["secondary"] > 0
    # Primary recovered and is receiving traffic again.
    assert router.health_snapshot()[0] is HealthState.HEALTHY
    assert served["primary"] > 0


def test_all_unhealthy_rejects_leftover_exact_ratio():
    # Both providers permanently fail. x=1 so each trips after one failure.
    # Recovery is impossible (never succeed), so the steady state is fixed:
    # weights [10, 10] + reject 80. SWRR is deterministic, so over any window
    # of 100 the split is EXACT — not approximate.
    primary = MockProvider("primary", outcomes=[False] * 100000)
    secondary = MockProvider("secondary", outcomes=[False] * 100000)
    router = FailoverRouter([primary, secondary], RouterConfig(p=10, x=1, y=5))

    # Trip both to unhealthy (one failing route each) to reach steady state.
    with pytest.raises(ProviderError):
        router.route(None)  # primary probe (index 0) fails -> primary unhealthy
    with pytest.raises(ProviderError):
        router.route(None)  # secondary probe fails -> secondary unhealthy

    # Reset the observable call counters so we measure ONLY the steady state.
    primary.call_count = 0
    secondary.call_count = 0

    served = _route_many(router, 100)
    # Steady weights [10, 10, reject 80]: exactly 10 probes to each provider
    # (all ERROR, since they keep failing) and exactly 80 REJECTs.
    assert served["ERROR"] == 20
    assert served["REJECT"] == 80
    # Each provider was invoked exactly its probe share — proof the probe stream
    # keeps flowing so recovery is always possible.
    assert primary.call_count == 10
    assert secondary.call_count == 10


def test_reject_invokes_no_provider():
    # The reject bucket must invoke NO provider. With both unhealthy and never
    # recovering, exactly p% reaches each provider and the leftover rejects;
    # the total invocations across a 100-window must equal exactly n*p (= 20),
    # never more. This deterministically proves reject touches no provider.
    primary = MockProvider("primary", outcomes=[False] * 100000)
    secondary = MockProvider("secondary", outcomes=[False] * 100000)
    router = FailoverRouter([primary, secondary], RouterConfig(p=10, x=1, y=5))
    with pytest.raises(ProviderError):
        router.route(None)
    with pytest.raises(ProviderError):
        router.route(None)
    primary.call_count = 0
    secondary.call_count = 0
    _route_many(router, 100)
    # n*p = 2*10 = 20 total invocations; the other 80 rejected without any call.
    assert primary.call_count + secondary.call_count == 20
