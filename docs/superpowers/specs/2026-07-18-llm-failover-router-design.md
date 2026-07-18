# LLM Failover Router — Design

**Date:** 2026-07-18
**Status:** Approved for implementation
**Language:** Python (production-grade, no hard third-party dependency for core logic)

---

## 1. Problem

Route each LLM request to one of *n* providers ordered by priority (e.g. OpenAI
primary, Anthropic secondary, …). While the primary is healthy, send it 100% of
traffic. When a provider degrades, shift traffic toward lower-priority providers
using a fixed probe fraction `p`, while still sending each unhealthy provider a
small `p%` probe stream so it has the opportunity to recover. When a provider
becomes healthy again, traffic returns to it. The system is multi-threaded:
many requests arrive concurrently.

## 2. Requirements (from the source `requirements` file)

- Providers extend a common base class; design for *n* providers, each with a priority.
- Healthy state → 100% to the highest-priority provider.
- Degraded state → each **unhealthy** provider still receives `p%` (a probe
  stream); the remaining "bulk" traffic goes to the highest-priority **healthy**
  provider.
- Recovery → when a provider is healthy again, traffic returns to it (the
  highest-priority healthy provider always receives the bulk).
- Health detection:
  - `x` consecutive failures ⇒ provider becomes **unhealthy**.
  - `y` consecutive successes ⇒ provider becomes **healthy** again.
- Multi-threaded: concurrent requests must be routed correctly.

## 3. Decisions (locked with the requester)

| # | Decision | Choice |
|---|----------|--------|
| D1 | How the p% / bulk split is realized on the concurrent hot path | **Deterministic exact ratio** — shared state under a lock, exact split even in small windows (contention accepted). |
| D2 | What happens to a request whose chosen provider's call fails | **Fail fast, record only** — the error propagates to the caller; the failure updates health counters; the request is **not** retried on another provider. |
| D3 | Where the leftover `(100 − n·p)%` goes when **all** providers are unhealthy | **Rejected** — every unhealthy provider gets its `p%` probe; the remainder is returned to the client as an error. No "park remainder on primary" special case. |
| D4 | Form of provider implementations | **Abstract base + configurable mock + thin OpenAI/Anthropic example adapters** (no hard SDK dependency). Tests are hermetic (no API keys, offline). |

## 4. Core model — one uniform weight rule

The entire routing behavior reduces to a single pure function over the current
health of the providers. Providers are indexed by priority: index `0` is the
highest priority (primary).

```
Given n providers and probe fraction p (integer percent):

  weight[i] = p                       for every UNHEALTHY provider i
  bulk      = 100 − (p · number_of_unhealthy_providers)

  the highest-priority HEALTHY provider receives `bulk` (added to its weight, which was 0)
  if there is NO healthy provider:
      `bulk` becomes the REJECT weight — those requests fail fast with an error,
      and NO provider is invoked for them.
```

Weights (including the reject bucket) always sum to 100.

### 4.1 Verification against every spec example (n = 3, priority P1 > P2 > P3)

| State | `[P1, P2, P3]` | reject | Matches requirement |
|-------|----------------|--------|---------------------|
| All healthy | `[100, 0, 0]` | 0 | "100% to primary" |
| P1 unhealthy | `[p, 100−p, 0]` | 0 | "(100−p)% to secondary, p% still to primary" |
| P1, P2 unhealthy | `[p, p, 100−2p]` | 0 | "p→P1, p→P2, (100−2p)→next" |
| All unhealthy | `[p, p, p]` | `100−3p` | each unhealthy keeps its p% probe; rest rejected |
| P1 recovering (healthy again) | `[100−p, 0, p]` → `[100, 0, 0]` | 0 | "traffic returns to primary" |

### 4.2 Why the probe stream matters

`p%` to every unhealthy provider is not decoration — it is the recovery
mechanism. An unhealthy provider can only transition back to healthy by
accumulating `y` consecutive **successful** invocations (D2 / health rules). If
it received 0% traffic it could never earn those successes → it would be
permanently stuck unhealthy (starvation). The probe stream guarantees liveness.

Consequence during a **total outage**: exactly `n·p%` of requests are still
*attempted* (the probes) and `(100 − n·p)%` fail fast as rejects. Honest errors,
no starvation.

### 4.3 Config validation

- `1 ≤ p` and `n · p ≤ 100` — keeps every weight (including bulk) non-negative.
- `x ≥ 1` (consecutive failures to trip unhealthy).
- `y ≥ 1` (consecutive successes to recover).
- At least one provider; priorities are the list order (index 0 = primary).

## 5. Modules

Each module has one clear purpose, a small interface, and is independently
testable.

### 5.1 `provider.py`
- `Provider` (ABC): `name: str`, `invoke(request) -> response`. Raises
  `ProviderError` on failure. This is the base class every provider extends.
- `ProviderError(Exception)`: wraps any provider-side failure.
- `MockProvider(Provider)`: scriptable for tests — can be told to fail on
  demand (e.g. a queue/predicate of outcomes) and to inject latency. Enables
  deterministic, offline tests.
- `OpenAIProvider`, `AnthropicProvider`: thin example adapters. The `openai` /
  `anthropic` SDKs are imported lazily inside `__init__` so importing the
  router package never requires those packages to be installed.

### 5.2 `health.py`
- `HealthState` (Enum): `HEALTHY`, `UNHEALTHY`.
- `ProviderHealth`: the per-provider consecutive-outcome state machine.
  - Starts `HEALTHY`.
  - `record_success()`: while `HEALTHY`, resets the failure counter. While
    `UNHEALTHY`, increments the success counter; at `y` consecutive successes →
    `HEALTHY` (counters reset).
  - `record_failure()`: while `UNHEALTHY`, resets the success counter. While
    `HEALTHY`, increments the failure counter; at `x` consecutive failures →
    `UNHEALTHY` (counters reset).
  - Returns whether a state transition occurred (so the router knows to
    recompute weights).
  - Pure in-memory logic; no I/O, no locking (locking is the router's job).

### 5.3 `selector.py`
- `compute_weights(health_states, p) -> (weights, reject_weight)`: the pure
  function from §4. Deterministic; fully unit-tested against the §4.1 table.
- `WeightedSelector`: the deterministic exact-ratio engine, implemented as
  **Smoothed Weighted Round-Robin (SWRR)** — the algorithm nginx uses.

  ```
  pick():
      for each target i:  current[i] += weight[i]
      winner = argmax(current)
      current[winner] -= total_weight        # total_weight = 100
      return winner
  ```

  Properties: over any window the emitted distribution matches the weights
  exactly (not just in expectation), and picks are *smoothly interleaved*
  (e.g. weights `[70,30]` → `A,B,A,A,B,A,A,B,A,A…`, never bursty). When weights
  change, `current[]` is reset to start a clean epoch. The reject bucket is
  just one more weighted target.

### 5.4 `router.py`
- `RouterConfig`: `p`, `x`, `y` (validated per §4.3).
- `FailoverRouter`:
  - Holds the ordered providers, a `ProviderHealth` per provider, a
    `WeightedSelector`, and a single `threading.RLock`.
  - `route(request) -> response`:
    1. **Under the lock**: `pick()` the target. If it's the reject bucket,
       raise `NoHealthyProviderError` (a `ProviderError` subclass).
    2. **Outside the lock**: call `provider.invoke(request)` (the slow network
       call). Success → return the response; failure → capture the
       `ProviderError`.
    3. **Under the lock**: record the outcome into that provider's
       `ProviderHealth`; if a transition occurred, recompute weights and load
       them into the selector.
    4. On failure, re-raise the captured error (fail-fast, D2).

## 6. Concurrency model

- A single `threading.RLock` guards the two short critical sections: **pick**
  (step 1) and **record + maybe-recompute** (step 3).
- The provider `invoke()` — the slow part — runs **outside** the lock, so
  network calls execute fully in parallel across threads. Only microsecond-scale
  selection and health bookkeeping are serialized.
- This delivers D1 (exact ratio even in small windows) because SWRR advances
  under the lock, while keeping the held-lock duration off the network path.
- **Known scaling ceiling** (stated, not hidden): provider *selection* is
  serialized across threads. At typical LLM-router QPS this is negligible; if
  selection ever becomes the bottleneck, the lock is the thing to shard.

## 7. Error handling

- `ProviderError`: base for all provider failures; what health tracking counts.
- `NoHealthyProviderError(ProviderError)`: raised when the reject bucket is
  selected (all providers unhealthy and the request fell in the leftover).
- `ConfigError(ValueError)`: raised by `RouterConfig` on invalid `p/x/y/n`.
- The reject bucket performs **no** provider invocation and records **no**
  health outcome — only real invocations feed the state machine.

## 8. Testing strategy

- **`compute_weights`**: table-test all five §4.1 rows for a representative `p`;
  plus validation-error cases (`p<1`, `n·p>100`).
- **`ProviderHealth`**: outcome-sequence tests for both transitions
  (`x` failures ⇒ unhealthy; `y` successes ⇒ healthy) and counter resets on an
  interrupting opposite outcome.
- **`WeightedSelector`**: assert exact counts over a full window (e.g. 100
  picks for `[70,30]` ⇒ exactly 70/30) and assert smoothness (no long runs).
- **`FailoverRouter` (end-to-end)**: drive a mock primary to fail `x` times →
  assert traffic shifts to secondary at the correct ratio; drive `y` successes
  → assert 100% snaps back to primary. Assert reject behavior when all
  unhealthy.
- **Concurrency stress test**: many threads through `route()` with scripted
  mock outcomes; assert no exceptions/races and that aggregate routing ratios
  match the expected weights.

## 9. Out of scope (YAGNI)

- No same-request cascade/retry across providers (explicitly rejected in D2).
- No time-based/half-open circuit breaking, jitter, or backoff — health is
  purely consecutive-count based, per the requirements.
- No real network retries, streaming, or token accounting inside the adapters —
  the adapters are thin examples; the router logic is the deliverable.
- No persistence of health state across process restarts (in-memory only).

## 10. Deliverable layout

```
llm_failover_router/
  __init__.py        # public exports: FailoverRouter, RouterConfig, Provider, errors
  provider.py        # Provider ABC, ProviderError, MockProvider, OpenAI/Anthropic adapters
  health.py          # HealthState, ProviderHealth
  selector.py        # compute_weights, WeightedSelector (SWRR)
  router.py          # RouterConfig, FailoverRouter, NoHealthyProviderError
tests/
  test_health.py
  test_selector.py
  test_router.py
  test_concurrency.py
```
