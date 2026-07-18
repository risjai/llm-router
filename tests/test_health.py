"""Tests for the consecutive-count health state machine."""
import pytest
from llm_failover_router.health import HealthState, ProviderHealth


def test_starts_healthy():
    h = ProviderHealth(x=3, y=2)
    assert h.state is HealthState.HEALTHY
    assert h.is_healthy() is True


def test_trips_unhealthy_after_x_consecutive_failures():
    h = ProviderHealth(x=3, y=2)
    assert h.record_failure() is False  # 1 — no transition
    assert h.record_failure() is False  # 2 — no transition
    assert h.record_failure() is True   # 3 — transition to UNHEALTHY
    assert h.state is HealthState.UNHEALTHY


def test_a_success_resets_the_failure_streak_while_healthy():
    h = ProviderHealth(x=3, y=2)
    h.record_failure()
    h.record_failure()
    assert h.record_success() is False  # streak reset, still HEALTHY
    # Must take 3 fresh consecutive failures again to trip.
    h.record_failure()
    h.record_failure()
    assert h.state is HealthState.HEALTHY
    assert h.record_failure() is True
    assert h.state is HealthState.UNHEALTHY


def test_recovers_after_y_consecutive_successes():
    h = ProviderHealth(x=1, y=2)
    h.record_failure()                  # x=1 -> immediately UNHEALTHY
    assert h.state is HealthState.UNHEALTHY
    assert h.record_success() is False  # 1 success — not yet
    assert h.record_success() is True   # 2 successes — recover
    assert h.state is HealthState.HEALTHY


def test_a_failure_resets_the_success_streak_while_unhealthy():
    h = ProviderHealth(x=1, y=3)
    h.record_failure()                  # UNHEALTHY
    h.record_success()
    h.record_success()
    assert h.record_failure() is False  # still UNHEALTHY, success streak reset
    # Needs 3 fresh consecutive successes now.
    h.record_success()
    h.record_success()
    assert h.state is HealthState.UNHEALTHY
    assert h.record_success() is True
    assert h.state is HealthState.HEALTHY


def test_validates_x_and_y_are_at_least_1():
    with pytest.raises(ValueError, match=r"x and y must be >= 1, got x=0, y=1"):
        ProviderHealth(x=0, y=1)
    with pytest.raises(ValueError, match=r"x and y must be >= 1, got x=1, y=0"):
        ProviderHealth(x=1, y=0)
    with pytest.raises(ValueError, match=r"x and y must be >= 1, got x=0, y=0"):
        ProviderHealth(x=0, y=0)
    with pytest.raises(ValueError, match=r"x and y must be >= 1, got x=-1, y=1"):
        ProviderHealth(x=-1, y=1)
