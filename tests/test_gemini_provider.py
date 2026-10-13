"""Gemini provider tests use the official SDK types with mocked clients."""

from types import SimpleNamespace

import pytest

from google import genai
from core.llm_provider import GeminiProvider, LLMProviderError

_UNSET = object()


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, models):
        self.models = models
        self.closed = False

    def close(self):
        self.closed = True


def _provider(monkeypatch, *, response=_UNSET, error=None, init_error=None):
    if response is _UNSET:
        response = SimpleNamespace(text='{"result":"ok"}')
    models = FakeModels(response=response, error=error)
    client = FakeClient(models)

    def make_client(**kwargs):
        if init_error:
            raise init_error
        assert kwargs == {"api_key": "test-secret"}
        return client

    monkeypatch.setattr(genai, "Client", make_client)
    return GeminiProvider("test-secret", "gemini-test"), client, models


def _generate(provider):
    return provider.generate(
        '{"request":"do a test"}',
        instructions="Interpret safely",
        schema={"type": "object", "properties": {"result": {"type": "string"}}},
        timeout=4.5,
        max_output_tokens=32,
    )


def test_missing_key_and_model_are_rejected():
    with pytest.raises(LLMProviderError, match="missing_api_key"):
        GeminiProvider("", "gemini-test")
    with pytest.raises(LLMProviderError, match="invalid_configuration"):
        GeminiProvider("key", " ")


def test_generate_uses_official_sdk_structured_json_config(monkeypatch):
    provider, client, models = _provider(monkeypatch)
    assert _generate(provider) == '{"result":"ok"}'
    assert client.closed
    call = models.kwargs
    assert call["model"] == "gemini-test"
    assert call["contents"] == '{"request":"do a test"}'
    config = call["config"]
    assert config.system_instruction == "Interpret safely"
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema["type"] == "object"
    assert config.max_output_tokens == 32
    assert config.temperature == 0
    assert config.http_options.timeout == 4500
    assert config.http_options.retry_options.attempts == 1
    assert "test-secret" not in call["contents"]


@pytest.mark.parametrize(("code", "reason"), [
    (400, "invalid_request"),
    (401, "authentication"),
    (403, "authentication"),
    (408, "timeout"),
    (429, "rate_limited"),
    (503, "provider_error"),
    ("RESOURCE_EXHAUSTED", "rate_limited"),
    ("UNAUTHENTICATED", "authentication"),
    ("DEADLINE_EXCEEDED", "timeout"),
])
def test_sdk_http_errors_are_safely_categorized(monkeypatch, code, reason):
    error = RuntimeError("private error body")
    error.code = code
    provider, client, _ = _provider(monkeypatch, error=error)
    with pytest.raises(LLMProviderError) as caught:
        _generate(provider)
    assert str(caught.value) == reason
    assert "private" not in str(caught.value)
    assert client.closed


def test_timeout_and_network_errors_are_categorized(monkeypatch):
    for error, reason in (
        (TimeoutError("private details"), "timeout"),
        (ConnectionError("private details"), "network_error"),
    ):
        provider, client, _ = _provider(monkeypatch, error=error)
        with pytest.raises(LLMProviderError, match=reason):
            _generate(provider)
        assert client.closed


@pytest.mark.parametrize("response", [None, SimpleNamespace(text=None), SimpleNamespace(text=" ")])
def test_empty_or_malformed_sdk_response_is_rejected(monkeypatch, response):
    provider, client, _ = _provider(monkeypatch, response=response)
    with pytest.raises(LLMProviderError, match="malformed_response"):
        _generate(provider)
    assert client.closed


def test_sdk_initialization_failure_is_sanitized(monkeypatch):
    error = RuntimeError("private key must not leak")
    error.code = 401
    provider, _, _ = _provider(monkeypatch, init_error=error)
    with pytest.raises(LLMProviderError, match="authentication") as caught:
        _generate(provider)
    assert "private" not in str(caught.value)
