"""Thread-safe failover router across n priority-ordered LLM providers.

Public API. See the design spec at
docs/superpowers/specs/2026-07-18-llm-failover-router-design.md.
"""
from llm_failover_router.errors import (
    ProviderError,
    NoHealthyProviderError,
    ConfigError,
)
from llm_failover_router.provider import (
    Provider,
    MockProvider,
    OpenAIProvider,
    AnthropicProvider,
)
from llm_failover_router.health import HealthState, ProviderHealth
from llm_failover_router.selector import compute_weights, WeightedSelector
from llm_failover_router.router import FailoverRouter, RouterConfig

__all__ = [
    "ProviderError",
    "NoHealthyProviderError",
    "ConfigError",
    "Provider",
    "MockProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "HealthState",
    "ProviderHealth",
    "compute_weights",
    "WeightedSelector",
    "FailoverRouter",
    "RouterConfig",
]
