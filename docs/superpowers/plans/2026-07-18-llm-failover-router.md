# LLM Failover Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thread-safe router that distributes LLM requests across *n* priority-ordered providers, shifting traffic away from unhealthy providers (while keeping a `p%` recovery probe on each) and snapping back to the primary when it recovers.

**Architecture:** Four focused modules — `provider` (base class + mock/adapters), `health` (consecutive-count state machine), `selector` (pure weight function + deterministic Smoothed Weighted Round-Robin engine), `router` (orchestrates pick → invoke → record under a single lock). The whole routing behavior reduces to one pure `compute_weights` function; the selector turns those weights into an exact per-window distribution.

**Tech Stack:** Python 3.9+, standard library only for core logic (`abc`, `enum`, `threading`, `dataclasses`), `pytest` for tests. Provider SDK adapters (`openai`, `anthropic`) are imported lazily and are optional.

## Global Constraints

- Core logic (provider base, health, selector, router) MUST NOT import any third-party package. Only `openai`/`anthropic` adapters may, and only via lazy import inside `__init__`.
- `p` is an integer percent. Config validation: `1 ≤ p`, `n · p ≤ 100`, `x ≥ 1`, `y ≥ 1`, `n ≥ 1`.
- Providers are ordered by priority; index `0` is the highest priority (primary).
- Weights (including the reject bucket) always sum to exactly 100.
- Fail-fast: a failed provider call records the failure and re-raises to the caller; it is NEVER retried on another provider.
- Reject bucket performs no invocation and records no health outcome.
- Package name: `llm_failover_router`. Tests live in `tests/`.
- Every module gets a module docstring; every public class/function gets a docstring; inline comments explain *why*, not *what*.

---

### Task 1: Package scaffold + errors

**Files:**
- Create: `llm_failover_router/__init__.py`
- Create: `llm_failover_router/errors.py`
- Create: `tests/__init__.py`
- Create: `pyproject.toml`
- Test: `tests/test_errors.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ProviderError(Exception)`, `NoHealthyProviderError(ProviderError)`, `ConfigError(ValueError)` in `llm_failover_router.errors`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_errors.py
"""Tests for the exception hierarchy."""
import pytest
from llm_failover_router.errors import (
    ProviderError,
    NoHealthyProviderError,
    ConfigError,
)


def test_no_healthy_provider_is_a_provider_error():
    # Callers should be able to catch every provider-side failure — including
    # the "all providers unhealthy" reject — with a single `except ProviderError`.
    assert issubclass(NoHealthyProviderError, ProviderError)


def test_config_error_is_a_value_error():
    # Bad configuration is a programming/setup error, so it subclasses ValueError.
    assert issubclass(ConfigError, ValueError)


def test_errors_carry_messages():
    err = ProviderError("boom")
    assert str(err) == "boom"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_errors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_failover_router'`.

- [ ] **Step 3: Write minimal implementation**

```python
# pyproject.toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-failover-router"
version = "0.1.0"
description = "Thread-safe failover router across n priority-ordered LLM providers."
requires-python = ">=3.9"

[project.optional-dependencies]
openai = ["openai"]
anthropic = ["anthropic"]
dev = ["pytest"]

[tool.setuptools]
packages = ["llm_failover_router"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# llm_failover_router/errors.py
"""Exception hierarchy for the failover router.

All provider-side failures derive from :class:`ProviderError` so callers can
handle them with a single ``except``. Configuration mistakes are surfaced as
:class:`ConfigError` (a ``ValueError``) at construction time — fail fast, loudly.
"""


class ProviderError(Exception):
    """Raised when a provider invocation fails. Base of all provider errors."""


class NoHealthyProviderError(ProviderError):
    """Raised when a request is routed to the reject bucket.

    This happens only when every provider is unhealthy and the request fell
    into the leftover ``(100 - n*p)%`` that is not covered by any probe stream.
    No provider is invoked in this case.
    """


class ConfigError(ValueError):
    """Raised for invalid router configuration (bad p / x / y / provider set)."""
```

```python
# llm_failover_router/__init__.py
"""Thread-safe failover router across n priority-ordered LLM providers.

Public API is populated as the package is built out. See the design spec at
docs/superpowers/specs/2026-07-18-llm-failover-router-design.md.
"""
from llm_failover_router.errors import (
    ProviderError,
    NoHealthyProviderError,
    ConfigError,
)

__all__ = ["ProviderError", "NoHealthyProviderError", "ConfigError"]
```

```python
# tests/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_errors.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml llm_failover_router/ tests/
git commit -m "feat: package scaffold and error hierarchy"
```

---

### Task 2: Provider base class + MockProvider

**Files:**
- Create: `llm_failover_router/provider.py`
- Modify: `llm_failover_router/__init__.py` (export `Provider`, `MockProvider`)
- Test: `tests/test_provider.py`

**Interfaces:**
- Consumes: `ProviderError` from `llm_failover_router.errors`.
- Produces:
  - `Provider(ABC)` with attribute `name: str` and abstract method `invoke(self, request: Any) -> Any`.
  - `MockProvider(Provider)`: constructor `MockProvider(name: str, outcomes: Optional[Sequence[bool]] = None, response: Any = "ok", latency: float = 0.0)`. `outcomes` is an iterable of booleans consumed one per call (`True` = succeed, `False` = raise `ProviderError`); when exhausted or `None`, defaults to success. `invoke` sleeps `latency` seconds then returns `response` or raises `ProviderError`. Also exposes `call_count: int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provider.py
"""Tests for the Provider base class and the test-only MockProvider."""
import pytest
from llm_failover_router.provider import Provider, MockProvider
from llm_failover_router.errors import ProviderError


def test_provider_is_abstract():
    # The base class defines the contract every provider extends; it cannot
    # be instantiated directly.
    with pytest.raises(TypeError):
        Provider()  # type: ignore[abstract]


def test_mock_defaults_to_success():
    p = MockProvider("openai", response="hello")
    assert p.invoke({"prompt": "hi"}) == "hello"
    assert p.call_count == 1


def test_mock_scripted_outcomes_are_consumed_in_order():
    p = MockProvider("openai", outcomes=[True, False, True])
    assert p.invoke(None) == "ok"          # 1st: success
    with pytest.raises(ProviderError):
        p.invoke(None)                      # 2nd: failure
    assert p.invoke(None) == "ok"          # 3rd: success
    assert p.call_count == 3


def test_mock_defaults_to_success_after_outcomes_exhausted():
    p = MockProvider("openai", outcomes=[False])
    with pytest.raises(ProviderError):
        p.invoke(None)
    # Script exhausted -> default success, so the provider can "recover".
    assert p.invoke(None) == "ok"


def test_mock_has_a_name():
    assert MockProvider("anthropic").name == "anthropic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_failover_router.provider'`.

- [ ] **Step 3: Write minimal implementation**

```python
# llm_failover_router/provider.py
"""Provider abstraction: the base class every LLM provider extends.

The router only ever depends on the :class:`Provider` interface — a name and a
synchronous ``invoke(request)`` that returns a response or raises
:class:`ProviderError`. This keeps the router logic testable without any real
SDK or network access via :class:`MockProvider`.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Optional, Sequence

from llm_failover_router.errors import ProviderError


class Provider(ABC):
    """Base class for all LLM providers.

    Subclasses set ``self.name`` and implement :meth:`invoke`. ``invoke`` must
    raise :class:`ProviderError` (or a subclass) on any failure so the router
    can uniformly record it against the provider's health.
    """

    name: str

    @abstractmethod
    def invoke(self, request: Any) -> Any:
        """Perform one request against the provider and return its response.

        Raises:
            ProviderError: on any provider-side failure.
        """
        raise NotImplementedError


class MockProvider(Provider):
    """Deterministic, offline provider for tests.

    Args:
        name: Provider name.
        outcomes: Optional sequence of booleans consumed one per :meth:`invoke`
            call — ``True`` succeeds, ``False`` raises. When ``None`` or once the
            sequence is exhausted, calls default to success (so a mock can model
            a provider that fails a few times then recovers).
        response: Value returned on success.
        latency: Seconds to sleep before responding (models slow calls; useful
            for concurrency tests).
    """

    def __init__(
        self,
        name: str,
        outcomes: Optional[Sequence[bool]] = None,
        response: Any = "ok",
        latency: float = 0.0,
    ) -> None:
        self.name = name
        self._outcomes = deque(outcomes) if outcomes is not None else deque()
        self._response = response
        self._latency = latency
        self.call_count = 0

    def invoke(self, request: Any) -> Any:
        self.call_count += 1
        if self._latency:
            time.sleep(self._latency)
        succeed = self._outcomes.popleft() if self._outcomes else True
        if not succeed:
            raise ProviderError(f"{self.name} mock failure")
        return self._response
```

```python
# llm_failover_router/__init__.py  (append exports)
from llm_failover_router.provider import Provider, MockProvider

__all__ = [
    "ProviderError",
    "NoHealthyProviderError",
    "ConfigError",
    "Provider",
    "MockProvider",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_provider.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add llm_failover_router/provider.py llm_failover_router/__init__.py tests/test_provider.py
git commit -m "feat: Provider base class and MockProvider"
```

---

### Task 3: Health state machine

**Files:**
- Create: `llm_failover_router/health.py`
- Modify: `llm_failover_router/__init__.py` (export `HealthState`, `ProviderHealth`)
- Test: `tests/test_health.py`

**Interfaces:**
- Consumes: nothing (pure logic).
- Produces:
  - `HealthState(Enum)`: members `HEALTHY`, `UNHEALTHY`.
  - `ProviderHealth`: constructor `ProviderHealth(x: int, y: int)` where `x` = consecutive failures to trip unhealthy, `y` = consecutive successes to recover. Starts `HEALTHY`.
    - `state: HealthState` (property/attribute reflecting current state).
    - `is_healthy() -> bool`.
    - `record_success() -> bool`: returns `True` iff this call caused a state transition.
    - `record_failure() -> bool`: returns `True` iff this call caused a state transition.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_health.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_health.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_failover_router.health'`.

- [ ] **Step 3: Write minimal implementation**

```python
# llm_failover_router/health.py
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
        self.state = HealthState.HEALTHY
        # Only the streak relevant to the current state matters; we keep one
        # counter and reinterpret it per state to avoid stale bookkeeping.
        self._consecutive = 0

    def is_healthy(self) -> bool:
        return self.state is HealthState.HEALTHY

    def record_success(self) -> bool:
        if self.state is HealthState.HEALTHY:
            # Successes while healthy simply clear any partial failure streak.
            self._consecutive = 0
            return False
        # UNHEALTHY: count consecutive successes toward recovery.
        self._consecutive += 1
        if self._consecutive >= self._y:
            self.state = HealthState.HEALTHY
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
            self.state = HealthState.UNHEALTHY
            self._consecutive = 0
            return True
        return False
```

```python
# llm_failover_router/__init__.py  (append exports)
from llm_failover_router.health import HealthState, ProviderHealth

# add "HealthState", "ProviderHealth" to __all__
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_health.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add llm_failover_router/health.py llm_failover_router/__init__.py tests/test_health.py
git commit -m "feat: consecutive-count health state machine"
```

---

### Task 4: `compute_weights` pure function

**Files:**
- Create: `llm_failover_router/selector.py`
- Test: `tests/test_selector.py` (weights portion)

**Interfaces:**
- Consumes: `HealthState` from `llm_failover_router.health`.
- Produces:
  - `compute_weights(states: Sequence[HealthState], p: int) -> Tuple[List[int], int]`. Returns `(weights, reject_weight)` where `weights[i]` is the integer percent for provider `i` (priority order), and `reject_weight` is the leftover. Rules (from spec §4):
    - `weight[i] = p` for every UNHEALTHY provider.
    - `bulk = 100 - p * num_unhealthy`.
    - The highest-priority HEALTHY provider gets `bulk` (its weight, previously 0, becomes `bulk`).
    - If there is NO healthy provider, `reject_weight = bulk` and no provider receives the bulk.
    - Sum of `weights + reject_weight == 100` always.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_selector.py
"""Tests for compute_weights (pure) and the WeightedSelector (SWRR)."""
import pytest
from llm_failover_router.health import HealthState
from llm_failover_router.selector import compute_weights

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_selector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_failover_router.selector'`.

- [ ] **Step 3: Write minimal implementation**

```python
# llm_failover_router/selector.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_selector.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add llm_failover_router/selector.py tests/test_selector.py
git commit -m "feat: compute_weights routing policy"
```

---

### Task 5: WeightedSelector (SWRR engine)

**Files:**
- Modify: `llm_failover_router/selector.py` (add `WeightedSelector`)
- Test: `tests/test_selector.py` (add selector tests)

**Interfaces:**
- Consumes: nothing beyond stdlib.
- Produces:
  - `WeightedSelector`: constructor `WeightedSelector(weights: Sequence[int])`. Methods:
    - `pick() -> int`: returns the index of the next selected target (0..len-1). Implements Smoothed Weighted Round-Robin over the given integer weights.
    - `set_weights(weights: Sequence[int]) -> None`: replaces weights and resets internal `current` accumulators to begin a clean epoch.
  - Note: weights passed here are the *full* target list including the reject bucket as the last entry (the router owns that convention). A weight of 0 means that target is never picked. Total may be any positive integer (the router always passes weights summing to 100).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_selector.py  (append)
from collections import Counter
from llm_failover_router.selector import WeightedSelector


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_selector.py -v`
Expected: FAIL — `ImportError: cannot import name 'WeightedSelector'`.

- [ ] **Step 3: Write minimal implementation**

```python
# llm_failover_router/selector.py  (append)


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_selector.py -v`
Expected: PASS (11 passed total in this file).

- [ ] **Step 5: Commit**

```bash
git add llm_failover_router/selector.py tests/test_selector.py
git commit -m "feat: Smoothed Weighted Round-Robin selector"
```

---

### Task 6: RouterConfig (validation)

**Files:**
- Create: `llm_failover_router/router.py` (config only for now)
- Modify: `llm_failover_router/__init__.py` (export `RouterConfig`)
- Test: `tests/test_router.py` (config portion)

**Interfaces:**
- Consumes: `ConfigError` from `llm_failover_router.errors`.
- Produces:
  - `RouterConfig`: dataclass with fields `p: int`, `x: int`, `y: int`. Method `validate(self, num_providers: int) -> None` raising `ConfigError` unless `1 <= p`, `x >= 1`, `y >= 1`, `num_providers >= 1`, and `p * num_providers <= 100`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_router.py
"""Tests for RouterConfig validation and FailoverRouter behavior."""
import pytest
from llm_failover_router.router import RouterConfig
from llm_failover_router.errors import ConfigError


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_failover_router.router'`.

- [ ] **Step 3: Write minimal implementation**

```python
# llm_failover_router/router.py
"""FailoverRouter: routes each request to a provider, records the outcome, and
recomputes routing weights on any health transition — all thread-safe.

See docs/superpowers/specs/2026-07-18-llm-failover-router-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_failover_router.errors import ConfigError


@dataclass
class RouterConfig:
    """Router tuning parameters.

    Args:
        p: Integer probe percent kept on each unhealthy provider.
        x: Consecutive failures that trip a provider to unhealthy.
        y: Consecutive successes that recover a provider to healthy.
    """

    p: int
    x: int
    y: int

    def validate(self, num_providers: int) -> None:
        """Validate against the provider count. Raises :class:`ConfigError`."""
        if num_providers < 1:
            raise ConfigError("need at least one provider")
        if self.p < 1:
            raise ConfigError("p must be >= 1")
        if self.x < 1:
            raise ConfigError("x must be >= 1")
        if self.y < 1:
            raise ConfigError("y must be >= 1")
        if self.p * num_providers > 100:
            raise ConfigError(
                f"p*n must be <= 100 (got p={self.p} * n={num_providers})"
            )
```

```python
# llm_failover_router/__init__.py  (append exports: RouterConfig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_router.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add llm_failover_router/router.py llm_failover_router/__init__.py tests/test_router.py
git commit -m "feat: RouterConfig with validation"
```

---

### Task 7: FailoverRouter — routing, invocation, health recording

**Files:**
- Modify: `llm_failover_router/router.py` (add `FailoverRouter`)
- Modify: `llm_failover_router/__init__.py` (export `FailoverRouter`)
- Test: `tests/test_router.py` (add router behavior tests)

**Interfaces:**
- Consumes: `Provider`, `ProviderError`, `NoHealthyProviderError`, `ProviderHealth`, `HealthState`, `compute_weights`, `WeightedSelector`, `RouterConfig`.
- Produces:
  - `FailoverRouter`: constructor `FailoverRouter(providers: Sequence[Provider], config: RouterConfig)`. Validates config against provider count. Builds one `ProviderHealth(x, y)` per provider, computes initial weights, and builds a `WeightedSelector` over `provider_weights + [reject_weight]` (reject is the last selector index = `len(providers)`).
  - `route(self, request) -> Any`: picks a target under the lock; if the target is the reject index, raises `NoHealthyProviderError`; otherwise invokes the provider *outside* the lock; records success/failure under the lock; on any health transition, recomputes weights and calls `selector.set_weights`; re-raises `ProviderError` on failure (fail-fast).
  - `health_snapshot(self) -> List[HealthState]`: returns current per-provider states (for tests/observability).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_router.py  (append)
from llm_failover_router.router import FailoverRouter
from llm_failover_router.provider import MockProvider
from llm_failover_router.errors import ProviderError, NoHealthyProviderError
from llm_failover_router.health import HealthState
from collections import Counter


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_router.py -v`
Expected: FAIL — `ImportError: cannot import name 'FailoverRouter'`.

- [ ] **Step 3: Write minimal implementation**

```python
# llm_failover_router/router.py  (append; add imports at top of file)
import threading
from typing import Any, List, Sequence

from llm_failover_router.health import HealthState, ProviderHealth
from llm_failover_router.provider import Provider
from llm_failover_router.errors import ProviderError, NoHealthyProviderError
from llm_failover_router.selector import compute_weights, WeightedSelector


class FailoverRouter:
    """Routes requests across priority-ordered providers with health failover.

    Thread-safe: a single lock guards provider *selection* and health
    *bookkeeping*; the (slow) provider call runs outside the lock so network
    requests execute in parallel across threads.
    """

    def __init__(
        self, providers: Sequence[Provider], config: RouterConfig
    ) -> None:
        config.validate(num_providers=len(providers))
        self._providers = list(providers)
        self._config = config
        self._health = [
            ProviderHealth(x=config.x, y=config.y) for _ in providers
        ]
        # Reentrant lock: recompute path calls back into selection helpers.
        self._lock = threading.RLock()
        # Selector targets are [provider_0 .. provider_{n-1}, reject].
        self._reject_index = len(self._providers)
        self._selector = WeightedSelector(self._current_weights())

    def _current_weights(self) -> List[int]:
        """Full selector weight vector: provider weights followed by reject."""
        states = [h.state for h in self._health]
        weights, reject = compute_weights(states, self._config.p)
        return weights + [reject]

    def _recompute(self) -> None:
        """Reload selector weights from current health (call under the lock)."""
        self._selector.set_weights(self._current_weights())

    def health_snapshot(self) -> List[HealthState]:
        """Return the current health state of each provider (priority order)."""
        with self._lock:
            return [h.state for h in self._health]

    def route(self, request: Any) -> Any:
        """Route one request. Fail-fast: a failed call re-raises to the caller.

        Raises:
            NoHealthyProviderError: the request fell into the reject bucket.
            ProviderError: the selected provider's call failed.
        """
        # --- Critical section 1: pick a target. ---
        with self._lock:
            index = self._selector.pick()
            if index == self._reject_index:
                raise NoHealthyProviderError(
                    "all providers unhealthy; request rejected"
                )
            provider = self._providers[index]

        # --- Slow path: invoke OUTSIDE the lock so calls run in parallel. ---
        try:
            response = provider.invoke(request)
        except ProviderError:
            self._record(index, success=False)
            raise  # fail-fast: propagate to caller, no cross-provider retry
        else:
            self._record(index, success=True)
            return response

    def _record(self, index: int, success: bool) -> None:
        """Record an outcome and recompute weights on a health transition."""
        # --- Critical section 2: health bookkeeping. ---
        with self._lock:
            health = self._health[index]
            transitioned = (
                health.record_success() if success else health.record_failure()
            )
            if transitioned:
                self._recompute()
```

```python
# llm_failover_router/__init__.py  (append export: FailoverRouter)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_router.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_failover_router/router.py llm_failover_router/__init__.py tests/test_router.py
git commit -m "feat: FailoverRouter routing, invocation, and health recording"
```

---

### Task 8: Concurrency stress test

**Files:**
- Test: `tests/test_concurrency.py`

**Interfaces:**
- Consumes: `FailoverRouter`, `RouterConfig`, `MockProvider`, `ProviderError`, `NoHealthyProviderError`.
- Produces: nothing (test-only). Verifies no races and correct aggregate routing under concurrency.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_concurrency.py
"""Concurrency tests: many threads through route() must be race-free and
distribute traffic according to the current weights."""
import threading
from collections import Counter

import pytest

from llm_failover_router.router import FailoverRouter, RouterConfig
from llm_failover_router.provider import MockProvider
from llm_failover_router.errors import ProviderError, NoHealthyProviderError


def test_all_healthy_concurrent_routes_only_to_primary():
    primary = MockProvider("primary", response="primary", latency=0.001)
    secondary = MockProvider("secondary", response="secondary", latency=0.001)
    router = FailoverRouter([primary, secondary], RouterConfig(p=10, x=3, y=3))

    results = Counter()
    lock = threading.Lock()

    def worker():
        for _ in range(100):
            r = router.route(None)
            with lock:
                results[r] += 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 8 threads * 100 = 800 requests, all to the healthy primary. No exceptions,
    # no lost updates.
    assert results["primary"] == 800
    assert secondary.call_count == 0


def test_degraded_concurrent_distribution_matches_weights():
    # Primary is permanently down (x=1). Secondary healthy. Weights settle at
    # [10, 90]: SWRR guarantees the exact ratio even across threads because
    # selection is serialized under the lock.
    primary = MockProvider("primary", outcomes=[False] * 100000)
    secondary = MockProvider("secondary", response="secondary")
    router = FailoverRouter([primary, secondary], RouterConfig(p=10, x=1, y=1000))

    # Trip the primary once to reach the steady [10, 90] state.
    with pytest.raises(ProviderError):
        router.route(None)

    served = Counter()
    lock = threading.Lock()

    def worker():
        for _ in range(100):
            try:
                served[router.route(None)] += 1  # secondary success
            except ProviderError:
                with lock:
                    served["primary_probe_err"] += 1

    # Route 8*100 = 800 requests in steady state.
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = served["secondary"] + served["primary_probe_err"]
    # ~10% probe the (still failing) primary, ~90% go to secondary. Allow a
    # small tolerance for the single warm-up pick already consumed.
    assert 0.85 <= served["secondary"] / total <= 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_concurrency.py -v`
Expected: On a fresh checkout these pass once Task 7 is in; if run before Task 7, FAIL with `ImportError`. (This task adds no new production code — it is a hardening gate. If a race exists, these tests flake/fail.)

- [ ] **Step 3: Confirm implementation (no new production code expected)**

The router's locking from Task 7 should already satisfy these. If a test reveals a race (e.g. a lost health update or a selector picking under concurrent mutation), fix `router.py` by widening/correcting the critical sections — but do NOT move `provider.invoke` inside the lock.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_concurrency.py -v`
Expected: PASS (2 passed).

Also run the full suite: `pytest -v` → all green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_concurrency.py
git commit -m "test: concurrency stress tests for FailoverRouter"
```

---

### Task 9: OpenAI / Anthropic example adapters

**Files:**
- Modify: `llm_failover_router/provider.py` (add `OpenAIProvider`, `AnthropicProvider`)
- Modify: `llm_failover_router/__init__.py` (export both, guarded)
- Test: `tests/test_adapters.py`

**Interfaces:**
- Consumes: `Provider`, `ProviderError`.
- Produces:
  - `OpenAIProvider(Provider)`: `OpenAIProvider(model: str, name: str = "openai", client: Any = None, **client_kwargs)`. Lazily `import openai` inside `__init__` only when `client is None`; wraps `client.chat.completions.create(...)`. `invoke(request: dict)` expects `{"messages": [...]}`; returns the response object; wraps any exception in `ProviderError`.
  - `AnthropicProvider(Provider)`: `AnthropicProvider(model: str, name: str = "anthropic", client: Any = None, max_tokens: int = 1024, **client_kwargs)`. Lazily `import anthropic` inside `__init__` only when `client is None`; wraps `client.messages.create(...)`; same error wrapping.
  - Both accept an injected `client` so tests can pass a fake without importing the SDK.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapters.py
"""Tests for the thin OpenAI/Anthropic adapters using injected fake clients.
No real SDK or network access — the adapters must accept an injected client."""
import pytest
from llm_failover_router.provider import OpenAIProvider, AnthropicProvider
from llm_failover_router.errors import ProviderError


class _FakeOpenAIClient:
    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return {"echo": kwargs}


class _FakeAnthropicClient:
    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        return {"echo": kwargs}


class _BoomClient:
    def __init__(self):
        self.chat = self
        self.completions = self
        self.messages = self

    def create(self, **kwargs):
        raise RuntimeError("network down")


def test_openai_adapter_invokes_client():
    p = OpenAIProvider(model="gpt-x", client=_FakeOpenAIClient())
    out = p.invoke({"messages": [{"role": "user", "content": "hi"}]})
    assert out["echo"]["model"] == "gpt-x"
    assert p.name == "openai"


def test_anthropic_adapter_invokes_client():
    p = AnthropicProvider(model="claude-x", client=_FakeAnthropicClient())
    out = p.invoke({"messages": [{"role": "user", "content": "hi"}]})
    assert out["echo"]["model"] == "claude-x"
    assert p.name == "anthropic"


def test_adapter_wraps_client_errors_in_provider_error():
    p = OpenAIProvider(model="gpt-x", client=_BoomClient())
    with pytest.raises(ProviderError):
        p.invoke({"messages": []})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapters.py -v`
Expected: FAIL — `ImportError: cannot import name 'OpenAIProvider'`.

- [ ] **Step 3: Write minimal implementation**

```python
# llm_failover_router/provider.py  (append)


class OpenAIProvider(Provider):
    """Thin adapter over the OpenAI Chat Completions API.

    The ``openai`` SDK is imported lazily and only when no ``client`` is
    injected, so importing this module never requires the package. Tests inject
    a fake client. ``invoke`` expects ``{"messages": [...]}`` and returns the
    raw SDK response; any SDK exception is wrapped in :class:`ProviderError`.
    """

    def __init__(
        self,
        model: str,
        name: str = "openai",
        client: Any = None,
        **client_kwargs: Any,
    ) -> None:
        self.name = name
        self._model = model
        if client is None:
            import openai  # lazy: only needed for real usage

            client = openai.OpenAI(**client_kwargs)
        self._client = client

    def invoke(self, request: Any) -> Any:
        try:
            return self._client.chat.completions.create(
                model=self._model, messages=request["messages"]
            )
        except Exception as exc:  # wrap ALL SDK/network errors uniformly
            raise ProviderError(f"{self.name} call failed: {exc}") from exc


class AnthropicProvider(Provider):
    """Thin adapter over the Anthropic Messages API.

    The ``anthropic`` SDK is imported lazily and only when no ``client`` is
    injected. ``invoke`` expects ``{"messages": [...]}`` and returns the raw SDK
    response; any SDK exception is wrapped in :class:`ProviderError`.
    """

    def __init__(
        self,
        model: str,
        name: str = "anthropic",
        client: Any = None,
        max_tokens: int = 1024,
        **client_kwargs: Any,
    ) -> None:
        self.name = name
        self._model = model
        self._max_tokens = max_tokens
        if client is None:
            import anthropic  # lazy: only needed for real usage

            client = anthropic.Anthropic(**client_kwargs)
        self._client = client

    def invoke(self, request: Any) -> Any:
        try:
            return self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=request["messages"],
            )
        except Exception as exc:
            raise ProviderError(f"{self.name} call failed: {exc}") from exc
```

```python
# llm_failover_router/__init__.py  (append exports: OpenAIProvider, AnthropicProvider)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapters.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add llm_failover_router/provider.py llm_failover_router/__init__.py tests/test_adapters.py
git commit -m "feat: thin OpenAI and Anthropic provider adapters"
```

---

### Task 10: Full-suite green + README

**Files:**
- Create: `README.md`
- Test: full suite.

**Interfaces:**
- Consumes: the whole package.
- Produces: usage documentation.

- [ ] **Step 1: Run the full suite**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 2: Write the README**

```markdown
# LLM Failover Router

Thread-safe router that distributes requests across *n* priority-ordered LLM
providers, shifts traffic away from unhealthy ones (keeping a `p%` recovery
probe on each), and snaps back to the primary when it recovers.

## Behavior

- Healthy → 100% to the highest-priority provider.
- A provider trips **unhealthy** after `x` consecutive failures; it keeps a
  `p%` probe stream so it can earn `y` consecutive successes and **recover**.
- The highest-priority healthy provider receives the leftover "bulk" traffic.
- If every provider is unhealthy, each still gets its `p%` probe and the
  remaining `(100 − n·p)%` is rejected with `NoHealthyProviderError`.
- Fail-fast: a failed call is recorded and re-raised — never retried elsewhere.

## Usage

​```python
from llm_failover_router import FailoverRouter, RouterConfig, OpenAIProvider, AnthropicProvider

router = FailoverRouter(
    providers=[
        OpenAIProvider(model="gpt-4o"),        # primary  (index 0)
        AnthropicProvider(model="claude-..."), # secondary (index 1)
    ],
    config=RouterConfig(p=10, x=3, y=2),
)

response = router.route({"messages": [{"role": "user", "content": "Hello"}]})
​```

## Development

​```bash
pip install -e ".[dev]"
pytest -v
​```
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with behavior and usage"
```

---

## Self-Review

**Spec coverage:**
- Base class + n providers with priority → Tasks 2, 6/7 (priority = list order). ✓
- Degraded routing / probe on each unhealthy / bulk to top healthy → Task 4 `compute_weights` + Task 5 selector. ✓
- Recovery to primary → Task 4 (healthy provider takes bulk) + Task 7 (recompute on transition). ✓
- Health rules (x failures, y successes) → Task 3. ✓
- All-unhealthy reject of leftover → Tasks 4, 7. ✓
- Multi-threaded correctness → Task 7 locking + Task 8 stress tests. ✓
- Fail-fast (D2) → Task 7 `route`. ✓
- Deterministic exact ratio (D1) → Task 5 SWRR. ✓
- Providers as base + mock + adapters (D4) → Tasks 2, 9. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; no "similar to Task N" references. ✓

**Type consistency:** `compute_weights` returns `(weights, reject)` consistently (Tasks 4, 7). `WeightedSelector.pick/set_weights` consistent (Tasks 5, 7). `ProviderHealth.record_success/record_failure -> bool` consistent (Tasks 3, 7). `RouterConfig(p,x,y).validate(num_providers)` consistent (Tasks 6, 7). Selector target convention (`providers + [reject]`, reject index = `len(providers)`) consistent (Tasks 5, 7). ✓
