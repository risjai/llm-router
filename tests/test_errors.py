"""Tests for the exception hierarchy."""
import pytest
from llm_failover_router.errors import (
    ProviderError,
    NoHealthyProviderError,
    ConfigError,
)


def test_no_healthy_provider_is_a_provider_error():
    # Callers should be able to catch every provider-side failure — including
    # the "all providers unhealthy" reject — with a single `except ProviderError`.
    assert issubclass(NoHealthyProviderError, ProviderError)


def test_config_error_is_a_value_error():
    # Bad configuration is a programming/setup error, so it subclasses ValueError.
    assert issubclass(ConfigError, ValueError)


def test_errors_carry_messages():
    err = ProviderError("boom")
    assert str(err) == "boom"
