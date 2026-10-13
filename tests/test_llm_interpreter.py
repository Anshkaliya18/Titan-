"""Tests for schema, confidence, size, privacy, and fallback handling."""

import json

from core.command_registry import get_all_commands
from core.config_manager import DEFAULT_CONFIG, _deep_merge, _validate_and_fix
from core.llm_interpreter import LLMInterpreter, command_schema
from core.llm_provider import LLMProviderError


class FakeProvider:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def generate(self, prompt, *, instructions, schema, timeout, max_output_tokens):
        self.calls.append((prompt, schema, timeout, max_output_tokens))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def _result(intent="OPEN_APPLICATION", target="chrome", confidence=0.95,
            query=None, **extra):
    return json.dumps({
        "intent": intent,
        "target": target,
        "arguments": {"query": query},
        "confidence": confidence,
        **extra,
    })


def _enable_llm(monkeypatch, **overrides):
    values = {
        "enabled": True,
        "provider": "gemini",
        "model": "test-model",
        "timeout_seconds": 4,
        "max_input_length": 2000,
        "max_output_length": 2000,
        "confidence_threshold": 0.80,
        "dangerous_confidence_threshold": 0.95,
    }
    values.update(overrides)
    monkeypatch.setattr(
        "core.llm_interpreter.get_value",
        lambda section, key, default=None: values.get(key, default)
        if section == "llm" else default,
    )


def test_schema_uses_command_registry_intents():
    schema_intents = command_schema()["properties"]["intent"]["enum"]
    assert set(schema_intents) == set(get_all_commands())


def test_invalid_llm_config_defaults_disabled_and_keeps_safety_floor():
    config = _deep_merge(DEFAULT_CONFIG, {
        "llm": {
            "enabled": "yes", "dangerous_confidence_threshold": -10,
            "timeout_seconds": 999,
        }
    })
    fixed = _validate_and_fix(config)["llm"]
    assert fixed["enabled"] is False
    assert fixed["dangerous_confidence_threshold"] >= 0.95
    assert fixed["timeout_seconds"] <= 30


def test_valid_application_and_search_interpretations(monkeypatch):
    _enable_llm(monkeypatch)
    app_provider = FakeProvider(_result())
    app = LLMInterpreter(app_provider).interpret("Could you launch Chrome?")
    assert app.success and app.intent == "OPEN_APPLICATION"
    assert app.target == "chrome"

    search_provider = FakeProvider(_result(
        "SEARCH_WEB", None, 0.92, query="Python tutorials"
    ))
    search = LLMInterpreter(search_provider).interpret(
        "I want to look up Python tutorials."
    )
    assert search.success and search.intent == "SEARCH_WEB"
    assert search.target == "Python tutorials"


def test_disabled_and_provider_failure_fall_back(monkeypatch):
    _enable_llm(monkeypatch, enabled=False)
    disabled_provider = FakeProvider(_result())
    disabled = LLMInterpreter(disabled_provider).interpret("open chrome")
    assert not disabled.success and disabled.reason == "disabled"
    assert disabled_provider.calls == []

    _enable_llm(monkeypatch)
    failed = LLMInterpreter(FakeProvider(LLMProviderError("timeout"))).interpret(
        "open chrome"
    )
    assert not failed.success and failed.reason == "timeout"


def test_confirmation_phrases_are_never_sent_to_provider(monkeypatch):
    _enable_llm(monkeypatch)
    provider = FakeProvider(_result())
    result = LLMInterpreter(provider).interpret("yes")
    assert not result.success and result.reason == "confirmation_input"
    result = LLMInterpreter(provider).interpret("yes, do it")
    assert not result.success and result.reason == "confirmation_input"
    assert provider.calls == []


def test_enabled_without_api_key_falls_back(monkeypatch):
    _enable_llm(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = LLMInterpreter().interpret("Could you launch Chrome?")
    assert not result.success and result.reason == "unavailable"


def test_provider_payload_is_bounded_and_context_is_minimal(monkeypatch):
    _enable_llm(monkeypatch)
    provider = FakeProvider(_result())
    result = LLMInterpreter(provider).interpret(
        "open it", context={
            "last_application": "calculator",
            "command_history": ["private"],
            "api_key": "should never be included",
        },
    )
    assert result.success
    payload = json.loads(provider.calls[0][0])
    assert payload == {
        "request": "open it",
        "context": {"last_application": "calculator"},
    }
    assert provider.calls[0][2] == 4


def test_sensitive_or_oversized_input_is_not_sent(monkeypatch):
    _enable_llm(monkeypatch, max_input_length=20)
    provider = FakeProvider(_result())
    interpreter = LLMInterpreter(provider)
    assert interpreter.interpret("password: x").reason == "sensitive_input"
    assert interpreter.interpret("x" * 21).reason == "input_too_long"
    assert provider.calls == []


def test_low_confidence_and_malformed_outputs_rejected(monkeypatch):
    _enable_llm(monkeypatch)
    cases = [
        "not json", "[]", '"hello"', '{"target":"chrome"}',
        _result("UNKNOWN_COMMAND"),
        _result(confidence="high"), _result(confidence=5),
        _result(target={"execute": "something"}),
        _result(confidence=0.5),
        _result(confirmed=True),
        _result("OPEN_APPLICATION", r"C:\\malicious.exe"),
    ]
    for output in cases:
        result = LLMInterpreter(FakeProvider(output)).interpret("interpret request")
        assert not result.success, output


def test_dangerous_intents_need_high_confidence(monkeypatch):
    _enable_llm(monkeypatch)
    low = LLMInterpreter(FakeProvider(_result(
        "SHUTDOWN_COMPUTER", None, 0.50
    ))).interpret("please shut down my computer")
    high = LLMInterpreter(FakeProvider(_result(
        "SHUTDOWN_COMPUTER", None, 0.99
    ))).interpret("please shut down my computer")
    assert not low.success
    assert high.success


def test_contextual_close_can_leave_target_to_context_manager(monkeypatch):
    _enable_llm(monkeypatch)
    result = LLMInterpreter(FakeProvider(_result(
        "CLOSE_APPLICATION", None, 0.93
    ))).interpret(
        "Could you close the application I just opened?",
        context={"last_application": "chrome"},
    )
    assert result.success and result.target is None
