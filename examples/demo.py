"""End-to-end demonstration of the LLM failover router.

Runs four scenarios against scripted mock providers and prints the router's
own log lines so you can see, request by request, where traffic is routed and
when providers change health. Uses p=5 (each unhealthy provider keeps a 5%
recovery probe; the highest-priority healthy provider takes the rest).

Run:  python3 examples/demo.py
"""
from __future__ import annotations

import logging
import sys
from collections import Counter

from llm_failover_router import FailoverRouter, RouterConfig, MockProvider
from llm_failover_router.errors import NoHealthyProviderError, ProviderError

P = 5  # probe percent kept on each unhealthy provider


def _configure_logging() -> None:
    """Send the router's logs to stdout with clear, aligned formatting."""
    # Route logs to stdout (not stderr) so log lines and the narrative print in
    # the correct interleaved order when captured to a file.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("    %(levelname)-7s %(message)s"))
    root = logging.getLogger("llm_failover_router")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def _route_batch(router: FailoverRouter, n: int, *, trace: bool = False) -> Counter:
    """Route n requests; return a tally of served-by (or REJECT / ERROR)."""
    tally: Counter = Counter()
    for i in range(1, n + 1):
        try:
            served = router.route({"messages": [{"role": "user", "content": "hi"}]})
            tally[served] += 1
            if trace:
                print(f"    request #{i:<3d} -> {served}")
        except NoHealthyProviderError:
            tally["REJECT"] += 1
            if trace:
                print(f"    request #{i:<3d} -> REJECT (all providers unhealthy)")
        except ProviderError:
            tally["ERROR"] += 1
            if trace:
                print(f"    request #{i:<3d} -> ERROR (probe hit a failing provider)")
    return tally


def scenario_1_all_healthy() -> None:
    _banner("SCENARIO 1  All healthy -> 100% to primary")
    primary = MockProvider("openai(primary)", response="openai(primary)")
    secondary = MockProvider("anthropic(secondary)", response="anthropic(secondary)")
    router = FailoverRouter([primary, secondary], RouterConfig(p=P, x=3, y=3))
    tally = _route_batch(router, 20, trace=True)
    print(f"\n  RESULT over 20 requests: {dict(tally)}")
    print("  -> Every request went to the primary. Secondary untouched.")


def scenario_2_primary_degrades() -> None:
    _banner("SCENARIO 2  Primary fails x=3 in a row -> trips UNHEALTHY, traffic shifts")
    # Primary fails its first 3 calls, then would succeed. Secondary healthy.
    primary = MockProvider(
        "openai(primary)", outcomes=[False, False, False], response="openai(primary)"
    )
    secondary = MockProvider("anthropic(secondary)", response="anthropic(secondary)")
    router = FailoverRouter([primary, secondary], RouterConfig(p=P, x=3, y=3))

    print("\n  Sending 3 requests while the primary is still the default sink:")
    for i in range(1, 4):
        try:
            router.route({"messages": []})
        except ProviderError:
            print(f"    request #{i} -> ERROR (primary failing; fail-fast to caller)")
    print(f"\n  Health now: {[s.value for s in router.health_snapshot()]}")
    print("  -> After 3 consecutive failures the primary is UNHEALTHY;")
    print("     weights are now [primary=5, secondary=95].")


def scenario_3_degraded_distribution() -> None:
    _banner("SCENARIO 3  Degraded steady state -> 5% probes primary, 95% to secondary")
    # Primary permanently down so the steady state is stable and observable.
    primary = MockProvider("openai(primary)", outcomes=[False] * 100000)
    secondary = MockProvider("anthropic(secondary)", response="anthropic(secondary)")
    router = FailoverRouter([primary, secondary], RouterConfig(p=P, x=1, y=1000))

    # One failing request trips the primary to UNHEALTHY (x=1).
    try:
        router.route({"messages": []})
    except ProviderError:
        pass
    logging.getLogger("llm_failover_router").setLevel(logging.WARNING)  # quiet the per-request lines
    tally = _route_batch(router, 200)
    logging.getLogger("llm_failover_router").setLevel(logging.INFO)
    served_secondary = tally["anthropic(secondary)"]
    print(f"\n  RESULT over 200 requests: {dict(tally)}")
    print(f"  -> secondary served {served_secondary} (~95%), "
          f"primary probed {tally['ERROR']} (~5%). Exact, deterministic split.")


def scenario_4_recovery() -> None:
    _banner("SCENARIO 4  Primary recovers after y=3 successful probes -> snaps back to 100%")
    # Primary fails 3 (trip), then succeeds forever. y=3 successes recover it.
    primary = MockProvider(
        "openai(primary)", outcomes=[False, False, False], response="openai(primary)"
    )
    secondary = MockProvider("anthropic(secondary)", response="anthropic(secondary)")
    router = FailoverRouter([primary, secondary], RouterConfig(p=P, x=3, y=3))

    for _ in range(3):  # trip the primary UNHEALTHY
        try:
            router.route({"messages": []})
        except ProviderError:
            pass
    print(f"\n  After tripping: health = {[s.value for s in router.health_snapshot()]}")
    print("  Now routing 200 requests. The 5% probes hit the (now-recovered)")
    print("  primary; after 3 consecutive successes it flips back to HEALTHY.\n")
    tally = _route_batch(router, 200)
    print(f"\n  RESULT over 200 requests: {dict(tally)}")
    print(f"  Final health = {[s.value for s in router.health_snapshot()]}")
    print("  -> Primary is HEALTHY again and reclaimed the bulk of traffic.")


def main() -> None:
    _configure_logging()
    print("LLM FAILOVER ROUTER — LIVE DEMONSTRATION   (p=5%)")
    print("Providers, by priority: openai(primary) > anthropic(secondary)")
    scenario_1_all_healthy()
    scenario_2_primary_degrades()
    scenario_3_degraded_distribution()
    scenario_4_recovery()
    print("\n" + "=" * 74)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 74)


if __name__ == "__main__":
    main()
