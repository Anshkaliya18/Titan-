import main
from gui.app import _redact_secrets


def test_gui_text_uses_main_turn_pipeline(monkeypatch):
    seen = []
    monkeypatch.setattr(main, "_process_once", lambda text: (seen.append(text), main._display_response("safe response")))
    monkeypatch.setattr(main._tts_manager, "speak", lambda _text: {"success": True})
    responses = []
    pending = main.process_gui_text("  open calculator  ", responses.append)
    assert seen == ["  open calculator  "]
    assert responses == ["safe response"]
    assert pending is False


def test_empty_gui_text_never_invokes_pipeline(monkeypatch):
    monkeypatch.setattr(main, "_process_once", lambda _text: (_ for _ in ()).throw(AssertionError("executed")))
    assert main.process_gui_text("  ", lambda _response: None) is False


def test_gui_history_masks_credential_shaped_text():
    secret = "AIza" + "A" * 35
    displayed = _redact_secrets(f"GEMINI_API_KEY={secret}")
    assert secret not in displayed
    assert "[redacted]" in displayed
