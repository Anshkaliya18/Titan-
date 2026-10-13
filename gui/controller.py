"""Serialized background bridge from the desktop UI to the TITAN core."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable
from utils.logger import get_logger

logger = get_logger(__name__)


class GUIController:
    """Run core turns on one worker; UI callbacks are supplied by the view."""

    def __init__(self, process_turn: Callable | None = None) -> None:
        self._process_turn = process_turn or self._default_process_turn
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="titan-ui")
        self._closed = False

    @staticmethod
    def _default_process_turn(text: str, callback: Callable[[str], None]) -> bool:
        import main

        return main.process_gui_text(text, callback)

    def submit(self, text: str, on_response: Callable[[str], None], on_done: Callable | None = None) -> Future:
        if self._closed:
            raise RuntimeError("TITAN is shutting down")
        return self._worker.submit(self._run_turn, text, on_response, on_done)

    def _run_turn(self, text: str, on_response: Callable[[str], None], on_done: Callable | None) -> None:
        try:
            pending = self._process_turn(text, on_response)
            if on_done:
                on_done(None, pending)
        except Exception as exc:
            # Never pass exception text, which may contain sensitive data, to UI/logs.
            logger.warning("GUI turn failed (%s)", type(exc).__name__)
            if on_done:
                on_done(type(exc).__name__, False)

    def submit_voice(self, voice_input, on_transcript: Callable[[str], None], on_error: Callable[[str], None]) -> Future:
        if self._closed:
            raise RuntimeError("TITAN is shutting down")
        return self._worker.submit(self._listen, voice_input, on_transcript, on_error)

    @staticmethod
    def _listen(voice_input, on_transcript, on_error) -> None:
        result = voice_input.listen()
        if getattr(result, "success", False) and isinstance(getattr(result, "text", None), str):
            on_transcript(result.text)
        else:
            on_error(getattr(result, "error", None) or "SPEECH_UNAVAILABLE")

    def close(self) -> None:
        self._closed = True
        self._worker.shutdown(wait=False, cancel_futures=True)
