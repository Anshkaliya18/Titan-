"""Voice transcriptions use the same untrusted text processing pipeline."""

from unittest.mock import patch

from core.speech_recognizer import RecognitionResult
from core.session_state import SessionState
from core.voice_input import VoiceInput


class FakeRecognizer:
    def __init__(self, *transcripts):
        self.transcripts = list(transcripts)
        self.calls = 0

    def listen(self, **_kwargs):
        self.calls += 1
        text = self.transcripts.pop(0)
        return RecognitionResult(True, text)


def _enable_voice(monkeypatch):
    settings = {
        "enabled": True,
        "provider": "windows",
        "language": "en-US",
        "listen_timeout_seconds": 5,
        "phrase_time_limit_seconds": 10,
        "device": None,
    }
    monkeypatch.setattr(
        "core.voice_input.get_value",
        lambda section, key, default=None: settings.get(key, default)
        if section == "voice" else default,
    )


def _set_voice_input(monkeypatch, *transcripts):
    from core import input as input_module
    recognizer = FakeRecognizer(*transcripts)
    monkeypatch.setattr(input_module, "_voice_input", VoiceInput(recognizer))
    return recognizer


def _voice_turn(monkeypatch, transcript):
    import main

    _enable_voice(monkeypatch)
    recognizer = _set_voice_input(monkeypatch, transcript)
    monkeypatch.setattr(main, "session", SessionState())
    inputs = iter(["voice", "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    return main, recognizer


def test_transcribed_command_reaches_same_single_command_executor(monkeypatch):
    import main

    voice_main, _ = _voice_turn(monkeypatch, "open chrome")
    with patch("main.execute_command", return_value={
        "success": True, "message": "opened", "data": {},
    }) as voice_execute:
        voice_main._process_once()
    voice_command = voice_execute.call_args.args[0]
    assert voice_main.session.get("last_input") == "open chrome"

    with patch("main.execute_command", return_value={
        "success": True, "message": "opened", "data": {},
    }) as typed_execute:
        main._process_single("open chrome")
    assert voice_command == typed_execute.call_args.args[0]


def test_malicious_transcription_is_not_directly_executed(monkeypatch):
    main, _ = _voice_turn(
        monkeypatch, "powershell Remove-Item C:\\important.txt"
    )
    with patch("main.execute_command") as execute:
        main._process_once()
    execute.assert_not_called()


def test_spoken_context_reference_uses_session_context(monkeypatch):
    import main

    voice_main, _ = _voice_turn(monkeypatch, "close it")
    voice_main.session.context.record_success(
        {"intent": "OPEN_APPLICATION", "target": "chrome"},
        {"success": True},
    )
    with patch("main.execute_command", return_value={
        "success": True, "message": "closed", "data": {},
    }) as execute:
        voice_main._process_once()
    assert execute.call_args.args[0]["intent"] == "close_application"
    assert execute.call_args.args[0]["target"] == "chrome"


def test_spoken_multi_command_uses_existing_splitter(monkeypatch):
    main, _ = _voice_turn(monkeypatch, "open chrome and open calculator")
    calls = []

    def successful(command):
        calls.append(command)
        return {"success": True, "message": "ok", "data": {}}

    with patch("core.multi_command_executor.execute_command", successful):
        main._process_once()
    assert [command["target"] for command in calls] == ["chrome", "calculator"]


def test_spoken_shutdown_keeps_existing_typed_safety_confirmation(monkeypatch):
    import main

    _enable_voice(monkeypatch)
    recognizer = _set_voice_input(monkeypatch, "shutdown computer", "yes")
    monkeypatch.setattr(main, "session", SessionState())
    inputs = iter(["voice", "y", "yes"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    with patch("main.execute_command", return_value={
        "success": True, "message": "confirmed", "data": {},
    }) as execute:
        main._process_once()
        assert main.session.get_pending_confirmation() is not None
        execute.assert_not_called()
        main._process_once()
    execute.assert_called_once()
    assert recognizer.calls == 1


def test_voice_layer_does_not_interpret_or_bypass_llm(monkeypatch):
    import main
    from core.hybrid_intent import HybridIntentResult
    from core.intent import Intent
    from core.llm_interpreter import LLMInterpretation

    voice_main, _ = _voice_turn(monkeypatch, "please launch my web browser")
    monkeypatch.setattr(
        "main._natural_language.interpret",
        lambda *args, **kwargs: type("Result", (), {
            "success": False, "intent": "UNKNOWN", "target": None,
            "arguments": {}, "normalized_text": "please launch my web browser",
        })(),
    )
    monkeypatch.setattr(
        "main.detect_hybrid_intent",
        lambda **kwargs: HybridIntentResult(Intent.UNKNOWN, "ml", confidence=0),
    )
    monkeypatch.setattr(
        "main._llm_interpreter",
        type("FakeLLM", (), {
            "interpret": lambda self, text, context=None: LLMInterpretation(
                True, "OPEN_BROWSER", None, {}, 0.97, "accepted"
            )
        })(),
    )
    with patch("main.execute_command", return_value={
        "success": True, "message": "browser opened", "data": {},
    }) as execute:
        voice_main._process_once()
    assert execute.call_args.args[0]["intent"] == "open_browser"
