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
                r = router.route(None)  # secondary success
                with lock:
                    served[r] += 1
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
