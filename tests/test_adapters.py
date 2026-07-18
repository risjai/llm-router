"""Tests for the thin OpenAI/Anthropic adapters using injected fake clients.
No real SDK or network access — the adapters must accept an injected client."""
import pytest

from llm_failover_router.provider import OpenAIProvider, AnthropicProvider
from llm_failover_router.errors import ProviderError


class _FakeOpenAIClient:
    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return {"echo": kwargs}


class _FakeAnthropicClient:
    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        return {"echo": kwargs}


class _BoomClient:
    def __init__(self):
        self.chat = self
        self.completions = self
        self.messages = self

    def create(self, **kwargs):
        raise RuntimeError("network down")


def test_openai_adapter_invokes_client():
    p = OpenAIProvider(model="gpt-x", client=_FakeOpenAIClient())
    out = p.invoke({"messages": [{"role": "user", "content": "hi"}]})
    assert out["echo"]["model"] == "gpt-x"
    assert p.name == "openai"


def test_anthropic_adapter_invokes_client():
    p = AnthropicProvider(model="claude-x", client=_FakeAnthropicClient())
    out = p.invoke({"messages": [{"role": "user", "content": "hi"}]})
    assert out["echo"]["model"] == "claude-x"
    assert p.name == "anthropic"


def test_adapter_wraps_client_errors_in_provider_error():
    p = OpenAIProvider(model="gpt-x", client=_BoomClient())
    with pytest.raises(ProviderError):
        p.invoke({"messages": []})
