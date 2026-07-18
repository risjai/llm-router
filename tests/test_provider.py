"""Tests for the Provider base class and the test-only MockProvider."""
import pytest
from llm_failover_router.provider import Provider, MockProvider
from llm_failover_router.errors import ProviderError


def test_provider_is_abstract():
    # The base class defines the contract every provider extends; it cannot
    # be instantiated directly.
    with pytest.raises(TypeError):
        Provider()  # type: ignore[abstract]


def test_mock_defaults_to_success():
    p = MockProvider("openai", response="hello")
    assert p.invoke({"prompt": "hi"}) == "hello"
    assert p.call_count == 1


def test_mock_scripted_outcomes_are_consumed_in_order():
    p = MockProvider("openai", outcomes=[True, False, True])
    assert p.invoke(None) == "ok"          # 1st: success
    with pytest.raises(ProviderError):
        p.invoke(None)                      # 2nd: failure
    assert p.invoke(None) == "ok"          # 3rd: success
    assert p.call_count == 3


def test_mock_defaults_to_success_after_outcomes_exhausted():
    p = MockProvider("openai", outcomes=[False])
    with pytest.raises(ProviderError):
        p.invoke(None)
    # Script exhausted -> default success, so the provider can "recover".
    assert p.invoke(None) == "ok"


def test_mock_has_a_name():
    assert MockProvider("anthropic").name == "anthropic"
