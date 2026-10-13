"""Wake mode stays local, bounded, lazy, and separate from command execution."""

from unittest.mock import Mock

import numpy as np

from core.config_manager import DEFAULT_CONFIG, _deep_merge, _validate_and_fix
from core.wake_word import FRAME_SAMPLES, WAKE_MODEL_KEY, WakeWordDetector, WakeWordResult


def _settings(monkeypatch, **overrides):
    settings = {
        "wake_word": {
            "enabled": True,
            "provider": "local",
            "phrase": "hey jarvis",
            "sensitivity": 0.5,
            "sample_rate": 16000,
            "channels": 1,
        },
        "voice": {"enabled": True},
    }
    for key, value in overrides.items():
        section, name = key.split(".", 1)
        settings.setdefault(section, {})[name] = value

    def get_value(section, key, default=None):
        return settings.get(section, {}).get(key, default)

    monkeypatch.setattr("core.wake_word.get_value", get_value)
    return settings


class FakeStream:
    def __init__(self):
        self.reads = 0
        self.closed = False

    def read(self, size):
        assert size == FRAME_SAMPLES
        self.reads += 1
        return np.zeros(size, dtype=np.int16).tobytes()


class FakeMicrophone:
    def __init__(self, **kwargs):
        assert kwargs == {"sample_rate": 16000, "chunk_size": FRAME_SAMPLES}
        self.stream = FakeStream()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True
        self.stream.closed = True


def test_defaults_are_safe_and_config_is_validated():
    assert DEFAULT_CONFIG["wake_word"] == {
        "enabled": False,
        "provider": "local",
        "phrase": "hey jarvis",
        "sensitivity": 0.5,
        "sample_rate": 16000,
        "channels": 1,
    }
    invalid = _validate_and_fix(_deep_merge(DEFAULT_CONFIG, {
        "wake_word": {"enabled": True, "phrase": "computer", "sensitivity": 2.0}
    }))
    assert invalid["wake_word"]["enabled"] is False
    assert invalid["wake_word"]["phrase"] == "hey jarvis"
    malformed_number = _validate_and_fix(_deep_merge(DEFAULT_CONFIG, {
        "wake_word": {"enabled": True, "sensitivity": 1}
    }))
    assert malformed_number["wake_word"]["enabled"] is False


def test_disabled_start_never_loads_model_or_opens_microphone(monkeypatch):
    _settings(monkeypatch, **{"wake_word.enabled": False})
    model, microphone = Mock(), Mock()
    detector = WakeWordDetector(model, microphone)
    result = detector.start()
    assert result.error == "DISABLED"
    model.assert_not_called()
    microphone.assert_not_called()


def test_start_is_idempotent_and_stop_releases_model(monkeypatch):
    _settings(monkeypatch)
    model_factory = Mock(return_value=Mock())
    detector = WakeWordDetector(model_factory)
    assert detector.start().detected
    assert detector.start().detected
    assert model_factory.call_count == 1
    detector.stop()
    detector.stop()
    assert not detector.is_running()
    assert detector.wait_for_wake().error == "NOT_RUNNING"


def test_detection_uses_local_pcm_and_releases_microphone_before_return(monkeypatch):
    _settings(monkeypatch)
    scores = iter((0.2, 0.83))

    class FakeModel:
        def __init__(self, **kwargs):
            assert kwargs == {
                "wakeword_models": [WAKE_MODEL_KEY],
                "inference_framework": "onnx",
            }
            self.frames = []

        def predict(self, frame, **kwargs):
            assert frame.dtype == np.int16 and frame.size == FRAME_SAMPLES
            assert kwargs["debounce_time"] == 1.0
            self.frames.append(frame)
            return {WAKE_MODEL_KEY: next(scores)}

    microphones = []

    def microphone_factory(**kwargs):
        microphone = FakeMicrophone(**kwargs)
        microphones.append(microphone)
        return microphone

    detector = WakeWordDetector(FakeModel, microphone_factory)
    assert detector.start().detected
    result = detector.wait_for_wake()
    assert result == WakeWordResult(True, confidence=0.83)
    assert microphones[0].closed
    detector.stop()


def test_missing_voice_input_prevents_start_without_microphone(monkeypatch):
    _settings(monkeypatch, **{"voice.enabled": False})
    microphone = Mock()
    detector = WakeWordDetector(lambda **_: Mock(), microphone)
    assert detector.start().error == "VOICE_DISABLED"
    microphone.assert_not_called()


def test_wake_transcript_only_returns_text_after_existing_approval(monkeypatch, capsys):
    from core import input as input_module

    class FakeDetector:
        def is_running(self):
            return True

        def wait_for_wake(self):
            return WakeWordResult(True, confidence=0.9)

        def stop(self):
            pass

    class FakeVoice:
        def listen(self):
            return type("Result", (), {"success": True, "text": "open chrome"})()

    monkeypatch.setattr(input_module, "_wake_detector", FakeDetector())
    monkeypatch.setattr(input_module, "_voice_input", FakeVoice())
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert input_module.get_input() == "open chrome"
    output = capsys.readouterr().out
    assert "Wake detected." in output
    assert "You said: open chrome" in output
    assert "hey jarvis" not in output.casefold()
