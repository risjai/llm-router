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
