"""Download openWakeWord's pretrained Hey Jarvis model for offline use.

Run this explicit setup step once after installing requirements-voice.txt.
The wake detector never downloads model files during Jarvis startup.
"""

from pathlib import Path


def main() -> int:
    try:
        import openwakeword
        from openwakeword.utils import download_models
    except ImportError:
        print("Install optional voice dependencies first: python -m pip install -r requirements-voice.txt")
        return 1

    target = Path(openwakeword.__file__).parent / "resources" / "models"
    try:
        download_models(
            model_names=["hey_jarvis_v0.1"],
            target_directory=str(target),
        )
    except Exception as exc:
        print(f"Wake-word model setup failed ({type(exc).__name__}).")
        return 1

    required = (
        target / "hey_jarvis_v0.1.onnx",
        target / "melspectrogram.onnx",
        target / "embedding_model.onnx",
    )
    missing = [
        path.name for path in required
        if not path.is_file() or path.stat().st_size < 100_000
    ]
    if missing:
        print("Model setup did not create required files: " + ", ".join(missing))
        return 1

    print("The local Hey Jarvis wake model is ready for offline use.")
    print("The pretrained wake-word model is licensed CC BY-NC-SA 4.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
