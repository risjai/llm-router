# LLM Failover Router

Thread-safe router that distributes requests across *n* priority-ordered LLM
providers, shifts traffic away from unhealthy ones (keeping a `p%` recovery
probe on each), and snaps back to the primary when it recovers.

## Behavior

- **Healthy** → 100% to the highest-priority provider.
- A provider trips **UNHEALTHY** after `x` consecutive failures; it keeps a
  `p%` probe stream so it can earn `y` consecutive successes and **recover**.
- The highest-priority healthy provider receives the leftover "bulk" traffic.
- If **every** provider is unhealthy, each still gets its `p%` probe and the
  remaining `(100 − n·p)%` is rejected with `NoHealthyProviderError`.
- **Fail-fast:** a failed call is recorded and re-raised — never retried elsewhere.

Routing is realized with **Smoothed Weighted Round-Robin** (the nginx
algorithm), so the split is *exact and deterministic* over any window — e.g. at
`p=5` with a degraded primary, exactly 1 of every 20 requests probes the primary
(landing on request #10), and the other 19 go to the healthy secondary.

## Install / test

```bash
pip install -e ".[dev]"
python3 -m pytest -v          # 41 tests
```

## Usage

```python
from llm_failover_router import (
    FailoverRouter, RouterConfig, OpenAIProvider, AnthropicProvider,
)

router = FailoverRouter(
    providers=[
        OpenAIProvider(model="gpt-4o"),          # primary   (index 0)
        AnthropicProvider(model="claude-..."),   # secondary (index 1)
    ],
    config=RouterConfig(p=5, x=3, y=3),
)

response = router.route({"messages": [{"role": "user", "content": "Hello"}]})
```

## Live demonstration

```bash
PYTHONPATH=. python3 examples/demo.py       # 4 scenarios with live log output
PYTHONPATH=. python3 examples/trace_20.py   # request-by-request routing decision
```

The demo logs every routing decision and every health transition:

```
INFO    router initialised: providers=['openai(primary)', 'anthropic(secondary)'] p=5 x=3 y=3 initial_weights=[100, 0, 0]
INFO    request -> openai(primary) FAILED
INFO    request -> openai(primary) FAILED
INFO    request -> openai(primary) FAILED
WARNING HEALTH TRANSITION: openai(primary) -> UNHEALTHY; new weights=[5, 95, 0]
INFO    request -> anthropic(secondary) OK
...
WARNING HEALTH TRANSITION: openai(primary) -> HEALTHY; new weights=[100, 0, 0]
```

Evidence screenshots live in [`images/`](images/).

## Design

Full spec: [`docs/superpowers/specs/2026-07-18-llm-failover-router-design.md`](docs/superpowers/specs/2026-07-18-llm-failover-router-design.md)
Implementation plan: [`docs/superpowers/plans/2026-07-18-llm-failover-router.md`](docs/superpowers/plans/2026-07-18-llm-failover-router.md)

| Module | Responsibility |
|--------|----------------|
| `provider.py` | `Provider` base class, `MockProvider`, thin OpenAI/Anthropic adapters |
| `health.py` | Consecutive-count health state machine (`x` failures / `y` successes) |
| `selector.py` | `compute_weights` policy + SWRR `WeightedSelector` |
| `router.py` | `RouterConfig` + thread-safe `FailoverRouter` with logging |
