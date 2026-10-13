"""Send one isolated Gemini API diagnostic request; execute no Jarvis command."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from utils.paths import data_root

load_dotenv(data_root() / ".env", override=False)

from core.config_manager import get_value
from core.llm_provider import GeminiProvider, LLMProviderError


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Gemini test failed: GEMINI_API_KEY is not configured.")
        print("Add it to the project .env file, then rerun this script.")
        return 1

    model = get_value("llm", "model", "gemini-3.5-flash-lite")
    provider = GeminiProvider(api_key, model)
    try:
        response = provider.generate(
            json.dumps({"request": "reply with ok", "context": {}}),
            instructions=(
                "This is a connectivity diagnostic. Return one JSON object "
                "with a string field named result. Do not execute commands."
            ),
            schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            },
            timeout=10,
            max_output_tokens=32,
        )
        data = json.loads(response)
        if not isinstance(data, dict) or not isinstance(data.get("result"), str):
            print("Gemini test failed: response did not match the test schema.")
            return 1
    except LLMProviderError as exc:
        print(f"Gemini test failed: {exc}")
        return 1
    except (ValueError, TypeError):
        print("Gemini test failed: response was not valid JSON.")
        return 1

    print(f"Gemini API request succeeded with {model}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
