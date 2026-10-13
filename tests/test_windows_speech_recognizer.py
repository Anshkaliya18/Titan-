"""SAPI provider tests use fake COM objects; they never open real audio."""

from types import SimpleNamespace

import pytest

from core.speech_recognizer import RecognitionResult
from core.windows_speech_recognizer import WindowsSpeechRecognizer


class FakeToken:
    Id = "SAPI:en-US"

    @staticmethod
    def GetAttribute(_key):
        return "409"

    class Attributes:
        @staticmethod
        def Item(_key):
            return "409"


class FakeCategory:
    def SetId(self, _path):
        self.path = _path

    def EnumerateTokens(self):
        return [FakeToken()]


class FakeGrammar:
    def __init__(self):
        self.states = []

    def DictationLoad(self, *_args):
        pass

    def DictationSetState(self, state):
        self.states.append(state)


class FakeContext:
    def __init__(self):
        self.EventInterests = 0
        self.grammar = FakeGrammar()

    def CreateGrammar(self, _id):
        return self.grammar


class FakeRecognizer:
    def __init__(self):
        self.context = FakeContext()

    def CreateRecoContext(self):
        return self.context


class FakeAudioInput:
    def __init__(self):
        self.DeviceId = None
        self.closed = False

    def SetState(self, state):
        self.closed = state == 0


class FakeBackend:
    def __init__(self, event="recognition", text="open chrome", tokens=None):
        self.event, self.text = event, text
        self.tokens = [FakeToken()] if tokens is None else tokens
        self.audio = FakeAudioInput()
        self.recognizer = FakeRecognizer()
        self.initialized = self.uninitialized = 0
        self.sink = None
        self.client = SimpleNamespace(Dispatch=self.dispatch, WithEvents=self.with_events)
        self.pythoncom = SimpleNamespace(
            CoInitialize=self.initialize,
            CoUninitialize=self.uninitialize,
            PumpWaitingMessages=self.pump,
        )

    def initialize(self):
        self.initialized += 1

    def uninitialize(self):
        self.uninitialized += 1

    def dispatch(self, name):
        if name == "SAPI.SpObjectTokenCategory":
            return FakeCategoryWithTokens(self.tokens)
        if name == "SAPI.SpInProcRecognizer":
            return self.recognizer
        if name == "SAPI.SpMMAudioIn":
            return self.audio
        raise AssertionError(name)

    def with_events(self, _context, handler_type):
        self.sink = handler_type()
        self.sink.closed = False
        self.sink.close = lambda: setattr(self.sink, "closed", True)
        return self.sink

    def pump(self):
        if self.event == "recognition" and not self.sink.text:
            self.sink.OnRecognition(
                1, 0, 0,
                SimpleNamespace(PhraseInfo=SimpleNamespace(GetText=lambda: self.text)),
            )
        elif self.event == "timeout" and self.sink.speech_started_at is None:
            self.sink.OnHypothesis(1, 0, None)


class FakeCategoryWithTokens(FakeCategory):
    def __init__(self, tokens):
        self.tokens = tokens

    def EnumerateTokens(self):
        return self.tokens


@pytest.fixture(autouse=True)
def _windows(monkeypatch):
    monkeypatch.setattr("core.windows_speech_recognizer.sys.platform", "win32")


def test_basic_recognition_is_local_and_releases_com_and_microphone():
    backend = FakeBackend(text="open chrome")
    result = WindowsSpeechRecognizer(backend).listen(
        language="en-US", listen_timeout=5, phrase_time_limit=10,
    )
    assert result == RecognitionResult(True, "open chrome")
    assert backend.audio.closed
    assert backend.initialized == backend.uninitialized == 1
    assert backend.sink.closed
    assert backend.recognizer.context.grammar.states == [1, 0]


def test_selected_microphone_id_is_applied():
    backend = FakeBackend()
    result = WindowsSpeechRecognizer(backend).listen(
        language="en-US", listen_timeout=5, phrase_time_limit=10, device=2,
    )
    assert result.success and backend.audio.DeviceId == 2


def test_no_speech_returns_no_speech_and_closes_input(monkeypatch):
    backend = FakeBackend(event="none")
    ticks = iter([0, 0, 0.1, 0.2, 0.3, 0.4, 5.1, 5.2, 5.3])
    monkeypatch.setattr("core.windows_speech_recognizer.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("core.windows_speech_recognizer.time.sleep", lambda _seconds: None)
    result = WindowsSpeechRecognizer(backend).listen(
        language="en-US", listen_timeout=5, phrase_time_limit=10,
    )
    assert result.error == "NO_SPEECH"
    assert backend.audio.closed and backend.uninitialized == 1


def test_speech_timeout_and_invalid_configuration(monkeypatch):
    backend = FakeBackend(event="timeout")
    clock = [0.0]
    def advance(_seconds):
        clock[0] += 0.2
    monkeypatch.setattr("core.windows_speech_recognizer.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("core.windows_speech_recognizer.time.sleep", advance)
    provider = WindowsSpeechRecognizer(backend)
    timed_out = provider.listen(
        language="en-US", listen_timeout=5, phrase_time_limit=1,
    )
    assert timed_out.error == "TIMEOUT" and backend.audio.closed

    backend = FakeBackend()
    provider = WindowsSpeechRecognizer(backend)
    assert provider.listen(
        language="en-US", listen_timeout=0, phrase_time_limit=10,
    ).error == "INVALID_CONFIGURATION"
    assert backend.initialized == 0
    assert provider.listen(
        language="en-US", listen_timeout=5, phrase_time_limit=10, device=-1,
    ).error == "INVALID_CONFIGURATION"


def test_missing_recognizer_and_unsupported_language_are_controlled():
    missing = WindowsSpeechRecognizer(FakeBackend(tokens=[])).listen(
        language="en-US", listen_timeout=5, phrase_time_limit=10,
    )
    unsupported = WindowsSpeechRecognizer(FakeBackend()).listen(
        language="xx-ZZ", listen_timeout=5, phrase_time_limit=10,
    )
    assert missing.error == "SPEECH_UNAVAILABLE"
    assert unsupported.error == "UNSUPPORTED_LANGUAGE"


def test_device_activation_failure_is_structured():
    backend = FakeBackend()
    backend.audio.SetState = lambda _state: None
    backend.recognizer.CreateRecoContext = lambda: (_ for _ in ()).throw(OSError("device"))
    result = WindowsSpeechRecognizer(backend).listen(
        language="en-US", listen_timeout=5, phrase_time_limit=10,
    )
    assert result.error == "MICROPHONE_UNAVAILABLE"
    assert backend.uninitialized == 1


def test_non_windows_platform_is_rejected_without_loading_com(monkeypatch):
    monkeypatch.setattr("core.windows_speech_recognizer.sys.platform", "linux")
    provider = WindowsSpeechRecognizer()
    result = provider.listen(
        language="en-US", listen_timeout=5, phrase_time_limit=10,
    )
    assert result.error == "WINDOWS_SPEECH_REQUIRES_WINDOWS"

