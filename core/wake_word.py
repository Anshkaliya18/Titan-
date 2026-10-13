"""Local, output-free wake-word detection using openWakeWord."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.config_manager import get_value
from utils.logger import get_logger

logger = get_logger(__name__)

WAKE_MODEL_KEY = "hey_jarvis"
WAKE_MODEL_NAME = "hey_jarvis_v0.1"
FRAME_SAMPLES = 1280  # 80 ms of 16 kHz, mono, signed 16-bit PCM


@dataclass(frozen=True)
class WakeWordResult:
    detected: bool
    confidence: float = 0.0
    error: str | None = None


class WakeWordDetector:
    """Own the local wake model and open the shared mic only while waiting.

    A successful detection returns after the microphone context closes. The
    caller can then start Windows speech's one-shot capture without two
    components competing for the microphone. This class has no command,
    parser, executor, LLM, or TTS access.
    """

    def __init__(
        self,
        model_factory: Callable | None = None,
        microphone_factory: Callable | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._microphone_factory = microphone_factory
        self._model = None
        self._running = False
        self._sensitivity = 0.5

    def is_running(self) -> bool:
        return self._running

    def start(self) -> WakeWordResult:
        """Validate opt-in settings and initialize one on-device model."""
        if self._running:
            return WakeWordResult(True)
        if get_value("wake_word", "enabled", False) is not True:
            return WakeWordResult(False, error="DISABLED")
        if get_value("voice", "enabled", False) is not True:
            return WakeWordResult(False, error="VOICE_DISABLED")

        provider = get_value("wake_word", "provider", "local")
        phrase = get_value("wake_word", "phrase", "hey jarvis")
        sensitivity = get_value("wake_word", "sensitivity", 0.5)
        if (
            provider != "local"
            or not isinstance(phrase, str)
            or " ".join(phrase.casefold().split()) != "hey jarvis"
            or not isinstance(sensitivity, (int, float))
            or isinstance(sensitivity, bool)
            or not 0.1 <= sensitivity <= 0.95
            or get_value("wake_word", "sample_rate", 16000) != 16000
            or get_value("wake_word", "channels", 1) != 1
        ):
            return WakeWordResult(False, error="INVALID_CONFIGURATION")

        self._sensitivity = float(sensitivity)
        try:
            if self._model_factory is None:
                import openwakeword
                from openwakeword.model import Model

                model_paths = openwakeword.get_pretrained_model_paths("onnx")
                model_path = next(
                    (Path(path) for path in model_paths
                     if Path(path).name.casefold() == f"{WAKE_MODEL_NAME}.onnx"),
                    None,
                )
                resources = Path(openwakeword.__file__).parent / "resources" / "models"
                if (
                    model_path is None or not model_path.is_file()
                    or not (resources / "melspectrogram.onnx").is_file()
                    or not (resources / "embedding_model.onnx").is_file()
                ):
                    return WakeWordResult(False, error="MODEL_NOT_INSTALLED")
                self._model_factory = Model

            self._model = self._model_factory(
                wakeword_models=[WAKE_MODEL_KEY],
                inference_framework="onnx",
            )
            self._running = True
            logger.info("Local wake-word detector initialized")
            return WakeWordResult(True)
        except KeyboardInterrupt:
            self.stop()
            raise
        except Exception as exc:
            self._model = None
            self._running = False
            logger.warning("Wake-word initialization failed (%s)", type(exc).__name__)
            return WakeWordResult(False, error="INITIALIZATION_FAILED")

    def wait_for_wake(self) -> WakeWordResult:
        """Block on one 80 ms microphone frame at a time until wake/error."""
        model = self._model
        if not self._running or model is None:
            return WakeWordResult(False, error="NOT_RUNNING")

        try:
            import numpy as np
            if self._microphone_factory is None:
                import speech_recognition as sr

                microphone_factory = sr.Microphone
            else:
                microphone_factory = self._microphone_factory

            sample_rate = get_value("wake_word", "sample_rate", 16000)
            microphone = microphone_factory(
                sample_rate=sample_rate,
                chunk_size=FRAME_SAMPLES,
            )
            with microphone as source:
                stream = getattr(source, "stream", None)
                if stream is None:
                    return WakeWordResult(False, error="AUDIO_DEVICE_UNAVAILABLE")
                while self._running:
                    frame_bytes = stream.read(FRAME_SAMPLES)
                    frame = np.frombuffer(frame_bytes, dtype=np.int16)
                    if frame.size != FRAME_SAMPLES:
                        continue
                    scores = model.predict(
                        frame,
                        threshold={WAKE_MODEL_KEY: self._sensitivity},
                        debounce_time=1.0,
                    )
                    confidence = float(scores.get(WAKE_MODEL_KEY, 0.0))
                    if confidence >= self._sensitivity:
                        logger.info("Wake phrase detected")
                        return WakeWordResult(True, confidence=confidence)
            return WakeWordResult(False, error="STOPPED")
        except KeyboardInterrupt:
            raise
        except (AssertionError, AttributeError, OSError) as exc:
            logger.warning("Wake microphone unavailable (%s)", type(exc).__name__)
            return WakeWordResult(False, error="AUDIO_DEVICE_UNAVAILABLE")
        except Exception as exc:
            logger.warning("Wake detection failed (%s)", type(exc).__name__)
            return WakeWordResult(False, error="DETECTOR_ERROR")

    def stop(self) -> None:
        """Stop mode and release model resources; safe to call repeatedly."""
        was_running = self._running
        self._running = False
        self._model = None
        if was_running:
            logger.info("Wake-word detector stopped")
