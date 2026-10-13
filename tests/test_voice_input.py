"""Tests for voice mode configuration and shared input handling."""

from unittest.mock import patch

from core.speech_recognizer import RecognitionResult
from core.voice_input import VoiceInput
from core.config_manager import DEFAULT_CONFIG, _deep_merge, _validate_and_fix


class FakeRecognizer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def listen(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _settings(monkeypatch, **overrides):
    values = {
        "enabled": True,
        "provider": "windows",
        "language": "en-US",
        "listen_timeout_seconds": 5,
        "phrase_time_limit_seconds": 10,
        "device": None,
    }
    values.update(overrides)
    monkeypatch.setattr(
        "core.voice_input.get_value",
        lambda section, key, default=None: values.get(key, default)
        if section == "voice" else default,
    )


def test_voice_disabled_never_initializes_or_calls_recognizer(monkeypatch):
    _settings(monkeypatch, enabled=False)
    fake = FakeRecognizer(RecognitionResult(True, "open chrome"))
    assert not VoiceInput(fake).listen().success
    assert fake.calls == []


def test_voice_uses_bounded_config_and_returns_transcription(monkeypatch):
    _settings(
        monkeypatch,
        listen_timeout_seconds=999,
        phrase_time_limit_seconds=999,
    )
    fake = FakeRecognizer(RecognitionResult(True, "open chrome"))
    result = VoiceInput(fake).listen()
    assert result.success and result.text == "open chrome"
    assert fake.calls == [{
        "language": "en-US", "listen_timeout": 30,
        "phrase_time_limit": 30, "device": None,
    }]


def test_unknown_provider_fails_without_initializing_recognizer(monkeypatch):
    _settings(monkeypatch, provider="unknown")
    fake = FakeRecognizer(RecognitionResult(True, "open chrome"))
    result = VoiceInput(fake).listen()
    assert not result.success and result.error == "INVALID_CONFIGURATION"
    assert fake.calls == []


def test_config_defaults_voice_disabled_and_caps_timeout_values():
    assert DEFAULT_CONFIG["voice"]["enabled"] is False
    assert DEFAULT_CONFIG["voice"]["provider"] == "windows"
    config = _deep_merge(DEFAULT_CONFIG, {
        "voice": {
            "enabled": True,
            "listen_timeout_seconds": 999,
            "phrase_time_limit_seconds": 999,
        }
    })
    fixed = _validate_and_fix(config)["voice"]
    assert fixed["enabled"] is True
    assert fixed["listen_timeout_seconds"] == 30
    assert fixed["phrase_time_limit_seconds"] == 30


def test_legacy_whisper_config_migrates_without_executable_or_model():
    config = _deep_merge(DEFAULT_CONFIG, {
        "voice": {
            "enabled": True,
            "provider": "whisper_cpp",
            "language": "en",
            "executable_path": "whisper-cli",
            "model_path": "models/whisper/ggml-base.en.bin",
        }
    })
    voice = _validate_and_fix(config)["voice"]
    assert voice["enabled"] is True
    assert voice["provider"] == "windows"
    assert voice["language"] == "en-US"
    assert voice["device"] is None
    assert "executable_path" not in voice and "model_path" not in voice


def test_malformed_windows_voice_device_disables_voice():
    config = _deep_merge(DEFAULT_CONFIG, {"voice": {"enabled": True, "device": "default"}})
    assert _validate_and_fix(config)["voice"]["enabled"] is False


def test_typed_voice_trigger_returns_only_transcribed_text(monkeypatch, capsys):
    from core import input as input_module

    _settings(monkeypatch)
    fake = FakeRecognizer(RecognitionResult(True, "open chrome"))
    monkeypatch.setattr(input_module, "_voice_input", VoiceInput(fake))
    with patch("builtins.input", side_effect=["voice", "y"]):
        value = input_module.get_input()
    assert value == "open chrome"
    assert "You said: open chrome" in capsys.readouterr().out


def test_voice_transcription_is_discarded_without_explicit_approval(monkeypatch, capsys):
    from core import input as input_module

    _settings(monkeypatch)
    fake = FakeRecognizer(RecognitionResult(True, "the room ."))
    monkeypatch.setattr(input_module, "_voice_input", VoiceInput(fake))
    with patch("builtins.input", side_effect=["voice", "n"]):
        value = input_module.get_input()
    assert value == ""
    assert "You said: the room ." in capsys.readouterr().out


def test_voice_failure_returns_to_typed_input(monkeypatch, capsys):
    from core import input as input_module

    _settings(monkeypatch)
    fake = FakeRecognizer(RecognitionResult(False, error="NO_SPEECH"))
    monkeypatch.setattr(input_module, "_voice_input", VoiceInput(fake))
    with patch("builtins.input", side_effect=["voice", "open chrome"]):
        failed = input_module.get_input()
        typed = input_module.get_input()
    assert failed == ""
    assert typed == "open chrome"
    assert "No speech detected." in capsys.readouterr().out


def test_voice_disabled_trigger_never_touches_microphone(monkeypatch, capsys):
    from core import input as input_module

    _settings(monkeypatch, enabled=False)
    fake = FakeRecognizer(RecognitionResult(True, "open chrome"))
    monkeypatch.setattr(input_module, "_voice_input", VoiceInput(fake))
    with patch("builtins.input", return_value="voice"):
        assert input_module.get_input() == ""
    assert fake.calls == []
    assert "Voice input is disabled" in capsys.readouterr().out
