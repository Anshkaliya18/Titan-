from concurrent.futures import wait
import threading

from gui.controller import GUIController


def test_controller_serializes_turns_and_delivers_results():
    active = 0
    peak = 0
    lock = threading.Lock()
    seen = []

    def process(text, callback):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        callback(f"response:{text}")
        with lock:
            active -= 1
        return text == "dangerous"

    controller = GUIController(process)
    first = []
    second = []
    done = []
    f1 = controller.submit("first", first.append, lambda error, pending: done.append((error, pending)))
    f2 = controller.submit("dangerous", second.append, lambda error, pending: done.append((error, pending)))
    wait([f1, f2])
    controller.close()
    assert first == ["response:first"]
    assert second == ["response:dangerous"]
    assert sorted(done, key=lambda value: value[1]) == [(None, False), (None, True)]
    assert peak == 1


def test_controller_hides_exception_details_from_ui():
    errors = []
    controller = GUIController(lambda *_args: (_ for _ in ()).throw(ValueError("secret detail")))
    future = controller.submit("request", lambda _response: None,
                               lambda error, _pending: errors.append(error))
    future.result()
    controller.close()
    assert errors == ["ValueError"]
    assert "secret detail" not in str(errors)


def test_voice_uses_injected_existing_recognizer_result():
    from core.speech_recognizer import RecognitionResult

    class FakeVoice:
        def listen(self):
            return RecognitionResult(True, text="open calculator")

    transcripts = []
    failures = []
    controller = GUIController(lambda *_args: False)
    controller.submit_voice(FakeVoice(), transcripts.append, failures.append).result()
    controller.close()
    assert transcripts == ["open calculator"]
    assert failures == []
