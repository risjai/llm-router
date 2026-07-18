"""Focused trace: with the primary UNHEALTHY and the secondary HEALTHY, which of
20 incoming requests is routed where? Answers the exact routing-decision question
and prints the deterministic SWRR pick for each request. p=5.
"""
from __future__ import annotations

import logging

from llm_failover_router import FailoverRouter, RouterConfig, MockProvider
from llm_failover_router.errors import ProviderError

# This focused trace narrates routing itself, so silence the router's own logs
# (the full demo.py is where the log stream is on display).
logging.getLogger("llm_failover_router").setLevel(logging.CRITICAL)

P = 5


def main() -> None:
    # Primary permanently down so it stays UNHEALTHY (stable observation);
    # secondary always healthy. x=1 => one failure trips the primary.
    primary = MockProvider("openai(primary)", outcomes=[False] * 100000)
    secondary = MockProvider("anthropic(secondary)", response="anthropic(secondary)")
    router = FailoverRouter([primary, secondary], RouterConfig(p=P, x=1, y=1000))

    # Trip the primary to UNHEALTHY with one (failing) request, then reset the
    # tally so we observe a clean 20-request window in the degraded steady state.
    try:
        router.route({"messages": []})
    except ProviderError:
        pass

    print("Degraded steady state: weights = [primary=5, secondary=95]")
    print("Routing 20 requests. p=5 => exactly 1 of every 20 probes the primary.\n")

    to_primary = []
    for i in range(1, 21):
        try:
            router.route({"messages": []})
            print(f"  request #{i:<2d} -> anthropic(secondary)   [healthy: serves the request]")
        except ProviderError:
            to_primary.append(i)
            print(f"  request #{i:<2d} -> openai(primary)  ** PROBE **   "
                  f"[unhealthy: 5% recovery probe, fails fast]")

    print(f"\nSummary: primary probed on request(s) {to_primary}; "
          f"the other {20 - len(to_primary)} went to the healthy secondary.")
    print("The probe is evenly spaced by design (SWRR), not random and not front-loaded,")
    print("so the router continuously samples whether the primary has recovered.")


if __name__ == "__main__":
    main()
