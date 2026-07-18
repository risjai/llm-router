"""FailoverRouter: routes each request to a provider, records the outcome, and
recomputes routing weights on any health transition — all thread-safe.

See docs/superpowers/specs/2026-07-18-llm-failover-router-design.md.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, List, Sequence

from llm_failover_router.errors import ConfigError, NoHealthyProviderError, ProviderError
from llm_failover_router.health import HealthState, ProviderHealth
from llm_failover_router.provider import Provider
from llm_failover_router.selector import compute_weights, WeightedSelector

logger = logging.getLogger("llm_failover_router")


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
        logger.info(
            "router initialised: providers=%s p=%d x=%d y=%d initial_weights=%s",
            [pr.name for pr in self._providers],
            config.p,
            config.x,
            config.y,
            self._current_weights(),
        )

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
                logger.warning(
                    "request REJECTED: all providers unhealthy (weights=%s)",
                    self._current_weights(),
                )
                raise NoHealthyProviderError(
                    "all providers unhealthy; request rejected"
                )
            provider = self._providers[index]

        # --- Slow path: invoke OUTSIDE the lock so calls run in parallel. ---
        try:
            response = provider.invoke(request)
        except ProviderError:
            logger.info("request -> %s FAILED", provider.name)
            self._record(index, success=False)
            raise  # fail-fast: propagate to caller, no cross-provider retry
        else:
            logger.info("request -> %s OK", provider.name)
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
                name = self._providers[index].name
                logger.warning(
                    "HEALTH TRANSITION: %s -> %s; new weights=%s",
                    name,
                    health.state.value.upper(),
                    self._current_weights(),
                )
                self._recompute()
