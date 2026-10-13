"""TTS stays local, lazy, output-only, and isolated from command flow."""

from types import SimpleNamespace
from unittest.mock import Mock

from core.config_manager import DEFAULT_CONFIG, _deep_merge, _validate_and_fix
from core.tts_manager import TTSManager, sanitize_speech_text


class FakeEngine:
    def __init__(self):
        self.properties = {}
        self.spoken = []

    def setProperty(self, name, value):
        self.properties[name] = value

    def getProperty(self, name):
        if name == "voices":
            return [SimpleNamespace(id="voice-id", name="English Voice")]
        return self.properties.get(name)

    def say(self, text):
        self.spoken.append(text)

    def runAndWait(self):
        pass


def test_default_tts_is_disabled():
    assert DEFAULT_CONFIG["tts"] == {
        "enabled": False, "provider": "local", "voice": None,
        "rate": 175, "volume": 1.0,
    }
    loaded = _validate_and_fix(_deep_merge(DEFAULT_CONFIG, {}))
    assert loaded["tts"] is not DEFAULT_CONFIG["tts"]


def test_disabled_or_empty_never_initializes_engine(monkeypatch):
    factory = Mock()
    manager = TTSManager(factory)
    monkeypatch.setattr("core.tts_manager.get_value", lambda section, key, default=None: False if key == "enabled" else default)
    assert manager.speak("Hello")['error'] == "DISABLED"
    monkeypatch.setattr("core.tts_manager.get_value", lambda section, key, default=None: True if key == "enabled" else default)
    assert manager.speak("  ")['error'] == "EMPTY_TEXT"
    assert factory.call_count == 0


def test_enabled_speech_uses_local_engine_and_optional_voice(monkeypatch):
    engine = FakeEngine()
    settings = {"enabled": True, "rate": 190, "volume": 0.5, "voice": "English Voice"}
    monkeypatch.setattr("core.tts_manager.get_value", lambda section, key, default=None: settings.get(key, default))
    manager = TTSManager(lambda: engine)
    result = manager.speak("Opening Google Chrome.")
    assert result["success"] is True
    assert engine.spoken == ["Opening Google Chrome."]
    assert engine.properties == {"rate": 190, "volume": 0.5, "voice": "voice-id"}


def test_sanitization_avoids_controls_secrets_paths_and_json():
    assert sanitize_speech_text("\x1b[31mOpening Chrome.\x1b[0m") == "Opening Chrome."
    cleaned = sanitize_speech_text("Key GEMINI_API_KEY=abc123. Open C:\\internal\\secret.txt now.")
    assert "abc123" not in cleaned
    assert "C:\\internal" not in cleaned
    assert "Open" in cleaned and "now." in cleaned
    assert "AIza" not in sanitize_speech_text("Credential AIza1234567890123456789012345678901234")
    assert sanitize_speech_text('{"intent":"OPEN_APPLICATION"}') == ""
    assert sanitize_speech_text(None) == ""


def test_tts_failure_is_reported_without_raising(monkeypatch):
    class BrokenEngine(FakeEngine):
        def runAndWait(self):
            raise RuntimeError("device detail must not leak")

    monkeypatch.setattr("core.tts_manager.get_value", lambda section, key, default=None: True if key == "enabled" else default)
    result = TTSManager(BrokenEngine).speak("Still printed.")
    assert result == {
        "success": False,
        "message": "Unable to speak response.",
        "error": "TTS_UNAVAILABLE",
    }


def test_missing_configured_voice_fails_safely(monkeypatch):
    settings = {"enabled": True, "rate": 175, "volume": 1.0, "voice": "Not Installed"}
    monkeypatch.setattr("core.tts_manager.get_value", lambda section, key, default=None: settings.get(key, default))
    result = TTSManager(FakeEngine).speak("Hello.")
    assert result["error"] == "TTS_UNAVAILABLE"


def test_malformed_tts_configuration_disables_speech():
    for tts in (
        {"enabled": "true"},
        {"enabled": True, "provider": "cloud"},
        {"enabled": True, "rate": 500},
        {"enabled": True, "volume": 2.0},
        "malformed",
    ):
        config = _validate_and_fix(_deep_merge(DEFAULT_CONFIG, {"tts": tts}))
        assert config["tts"]["enabled"] is False


def test_display_response_prints_even_if_tts_fails(monkeypatch, capsys):
    import main

    speak = Mock(side_effect=RuntimeError("simulated failure"))
    monkeypatch.setattr(main._tts_manager, "speak", speak)
    main._display_response("Opening Chrome.")
    assert "Opening Chrome." in capsys.readouterr().out
    speak.assert_called_once_with("Opening Chrome.")


def test_dangerous_command_prompt_is_spoken_without_execution(monkeypatch, capsys):
    import main
    from core.session_state import SessionState

    monkeypatch.setattr(main, "session", SessionState())
    monkeypatch.setattr(main, "get_input", lambda: "shutdown computer")
    speak = Mock()
    monkeypatch.setattr(main._tts_manager, "speak", speak)
    execute = Mock()
    monkeypatch.setattr(main, "execute_command", execute)

    main._process_once()

    status, pending = main.session.resolve_pending_confirmation()
    assert status == "active"
    assert pending.intent == "shutdown_computer"
    execute.assert_not_called()
    prompt = "Are you sure you want to shut down the computer? (yes/no):"
    speak.assert_called_once_with(prompt)
    assert prompt in capsys.readouterr().out
