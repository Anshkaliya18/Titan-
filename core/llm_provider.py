"""Provider boundary for optional LLM command interpretation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProviderError(Exception):
    """Safe, controlled provider failure category."""


class LLMProvider(ABC):
    """Provider-independent structured text generation interface."""

    @abstractmethod
    def generate(self, prompt: str, *, instructions: str,
                 schema: dict[str, Any], timeout: float,
                 max_output_tokens: int) -> str:
        """Return generated text; callers must validate it locally."""


class GeminiProvider(LLMProvider):
    """Google Gemini text generation adapter using the official SDK."""

    def __init__(self, api_key: str, model: str) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise LLMProviderError("missing_api_key")
        if not isinstance(model, str) or not model.strip():
            raise LLMProviderError("invalid_configuration")
        self._api_key = api_key.strip()
        self._model = model.strip()

    def generate(self, prompt: str, *, instructions: str,
                 schema: dict[str, Any], timeout: float,
                 max_output_tokens: int) -> str:
        if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
                or timeout <= 0):
            raise LLMProviderError("invalid_configuration")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMProviderError("provider_unavailable") from exc

        client = None
        try:
            client = genai.Client(api_key=self._api_key)
            config = types.GenerateContentConfig(
                system_instruction=instructions,
                response_mime_type="application/json",
                response_json_schema=schema,
                temperature=0,
                max_output_tokens=max_output_tokens,
                # One attempt avoids SDK retry bursts on quota errors. Jarvis
                # immediately falls back to its deterministic pipeline.
                http_options=types.HttpOptions(
                    timeout=max(1, int(timeout * 1000)),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
            content = getattr(response, "text", None)
            if not isinstance(content, str) or not content.strip():
                raise LLMProviderError("malformed_response")
            return content
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(_error_category(exc)) from None
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass


def _error_category(exc: Exception) -> str:
    """Map SDK/network failures to safe categories without exposing details."""
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        normalized = code.upper()
        if normalized in {"RESOURCE_EXHAUSTED", "RATE_LIMITED", "TOO_MANY_REQUESTS"}:
            return "rate_limited"
        if normalized in {"UNAUTHENTICATED", "PERMISSION_DENIED"}:
            return "authentication"
        if normalized in {"DEADLINE_EXCEEDED", "TIMEOUT"}:
            return "timeout"
        if normalized == "INVALID_ARGUMENT":
            return "invalid_request"
    try:
        status = int(code)
    except (TypeError, ValueError):
        status = None
    if status in (401, 403):
        return "authentication"
    if status == 429:
        return "rate_limited"
    if status in (408, 504):
        return "timeout"
    if status == 400:
        return "invalid_request"
    if status is not None and status >= 500:
        return "provider_error"

    name = type(exc).__name__.casefold()
    if "timeout" in name or isinstance(exc, TimeoutError):
        return "timeout"
    if any(term in name for term in ("connection", "network", "transport")):
        return "network_error"
    if isinstance(exc, ImportError):
        return "provider_unavailable"
    return "provider_error"
