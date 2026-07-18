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
