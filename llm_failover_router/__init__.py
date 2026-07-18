"""Thread-safe failover router across n priority-ordered LLM providers.

Public API is populated as the package is built out. See the design spec at
docs/superpowers/specs/2026-07-18-llm-failover-router-design.md.
"""
from llm_failover_router.errors import (
    ProviderError,
    NoHealthyProviderError,
    ConfigError,
)
from llm_failover_router.provider import Provider, MockProvider
from llm_failover_router.health import HealthState, ProviderHealth

__all__ = [
    "ProviderError",
    "NoHealthyProviderError",
    "ConfigError",
    "Provider",
    "MockProvider",
    "HealthState",
    "ProviderHealth",
]
