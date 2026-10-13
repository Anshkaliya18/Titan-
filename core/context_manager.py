"""Bounded, deterministic context for the current Jarvis session.

This module stores only typed entities from successful structured commands.
It never executes commands, reads the filesystem, or persists context.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass
from typing import Any

from core.application_manager import get_application
from core.config_manager import get_value
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_ITEMS = 10
DEFAULT_MAX_AGE_COMMANDS = 10
HARD_MAX_ITEMS = 50
HARD_MAX_AGE_COMMANDS = 100


@dataclass
class ContextItem:
    entity_type: str
    name: str | None
    intent: str
    active: bool
    command_number: int


@dataclass(frozen=True)
class ContextResolution:
    intent: str
    target: str | None
    arguments: dict[str, Any]
    canonical_text: str
    entity_type: str


def _positive_limit(key: str, default: int, maximum: int) -> int:
    value = get_value("context", key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return min(value, maximum)


class ContextManager:
    """Session-only typed entity context with bounded age and size."""

    def __init__(self, items: list[ContextItem] | None = None, command_number: int = 0):
        self.max_items = _positive_limit("max_items", DEFAULT_MAX_ITEMS, HARD_MAX_ITEMS)
        self.max_age_commands = _positive_limit(
            "max_age_commands", DEFAULT_MAX_AGE_COMMANDS, HARD_MAX_AGE_COMMANDS
        )
        self._items = copy.deepcopy(items or [])
        self._command_number = command_number

    def fork(self) -> "ContextManager":
        """Return an isolated preview copy, used while prevalidating a chain."""
        return ContextManager(self._items, self._command_number)

    def advance(self) -> None:
        """Advance command age and discard expired entries."""
        self._command_number += 1
        before = len(self._items)
        self._items = [
            item for item in self._items
            if self._command_number - item.command_number < self.max_age_commands
        ]
        if len(self._items) != before:
            logger.debug("Expired %d context item(s)", before - len(self._items))

    def record_success(self, command: dict, result: dict) -> None:
        """Record or update a typed entity only after successful execution."""
        if get_value("context", "enabled", True) is False:
            return
        if not isinstance(result, dict) or result.get("success") is not True:
            return
        intent = str(command.get("intent", "")).upper()
        target = command.get("target")
        if isinstance(target, str):
            target = target.strip()
        if target == "":
            target = None

        entity_type: str | None = None
        name: str | None = target if isinstance(target, str) else None
        active = True
        if intent in {"OPEN_APPLICATION", "CLOSE_APPLICATION"} and name:
            app = get_application(name)
            if app is None:
                return
            name = app.name
            entity_type = "browser" if app.is_browser else "application"
            active = intent == "OPEN_APPLICATION"
        elif intent == "OPEN_BROWSER":
            entity_type, name = "browser", None  # default browser; identity is unknown
        elif intent == "OPEN_URL" and name:
            entity_type = "url"
        elif intent in {"OPEN_FILE", "DELETE_FILE"} and name:
            if self._looks_sensitive(name):
                return
            entity_type = "file"
            active = intent == "OPEN_FILE"
        elif intent == "CREATE_FOLDER" and name:
            entity_type = "folder"
        elif intent == "SEARCH_WEB" and name:
            if self._looks_sensitive(name):
                return
            entity_type = "search"
        else:
            return

        normalized_name = name.casefold() if name is not None else None
        self._items = [
            item for item in self._items
            if not (item.entity_type == entity_type and
                    (item.name.casefold() if item.name else None) == normalized_name)
        ]
        self._items.insert(0, ContextItem(
            entity_type=entity_type,
            name=name,
            intent=intent,
            active=active,
            command_number=self._command_number,
        ))
        self._items = self._items[:self.max_items]
        logger.debug("Context updated (type=%s, intent=%s)", entity_type, intent)

    @staticmethod
    def _looks_sensitive(value: str) -> bool:
        return bool(re.search(
            r"\b(password|passwd|pwd|api[_ -]?key|token|secret|credential|cookie)\b\s*(?:=|:|is|was)",
            value, re.IGNORECASE,
        ))

    def resolve_reference(self, text: str) -> ContextResolution | None:
        """Resolve only explicit, typed references; never guesses from prose."""
        if get_value("context", "enabled", True) is False or not isinstance(text, str):
            return None
        phrase = " ".join(text.strip().lower().rstrip("?!.,;").split())

        if phrase in {"search again", "search that again", "search the same thing again"}:
            item = self._find({"search"})
            if item and item.name:
                return ContextResolution("SEARCH_WEB", item.name, {"query": item.name},
                                         f"search web {item.name}", "search")
            return None

        patterns = (
            (r"^(?:close|quit|exit|stop)\s+(?:it|that|this|the app|the application|that app|this app|that application|this application|the browser|that browser|this browser)$",
             "CLOSE_APPLICATION", {"application", "browser"}),
            (r"^delete\s+(?:it|that|this|the file|that file|this file)$",
             "DELETE_FILE", {"file"}),
            (r"^open\s+(?:it|that|this)(?:\s+again)?$",
             "OPEN_REFERENCE", {"application", "browser", "file", "folder", "url"}),
            (r"^open\s+(?:the|that|this)\s+(?:app|application)(?:\s+again)?$",
             "OPEN_TYPED_REFERENCE", {"application", "browser"}),
            (r"^open\s+(?:the|that|this)\s+file(?:\s+again)?$",
             "OPEN_TYPED_REFERENCE", {"file"}),
            (r"^open\s+(?:the|that|this)\s+browser(?:\s+again)?$",
             "OPEN_TYPED_REFERENCE", {"browser"}),
            (r"^(?:open|launch|start)\s+(?:it|that|this)\s+again$",
             "OPEN_REFERENCE", {"application", "browser", "file", "url"}),
        )
        for pattern, intent, allowed in patterns:
            if re.fullmatch(pattern, phrase):
                may_reopen = intent.startswith("OPEN")
                item = self._find(allowed, include_inactive=may_reopen)
                if item is None:
                    return None
                resolved_intent = intent
                if intent.startswith("OPEN"):
                    if item.entity_type in {"file", "folder"}:
                        resolved_intent = "OPEN_FILE"
                    elif item.entity_type == "folder":
                        resolved_intent = "OPEN_FILE"
                    elif item.entity_type == "url":
                        resolved_intent = "OPEN_URL"
                    elif item.entity_type == "browser" and item.name is None:
                        resolved_intent = "OPEN_BROWSER"
                    else:
                        resolved_intent = "OPEN_APPLICATION"
                if resolved_intent == "CLOSE_APPLICATION" and item.entity_type not in {"application", "browser"}:
                    return None
                if resolved_intent == "DELETE_FILE" and item.entity_type != "file":
                    return None
                if resolved_intent in {"OPEN_FILE", "OPEN_URL", "OPEN_APPLICATION", "CLOSE_APPLICATION"} and not item.name:
                    return None
                return ContextResolution(
                    resolved_intent, item.name, {},
                    self._canonical(resolved_intent, item.name), item.entity_type,
                )
        return None

    def resolve_for_intent(self, intent: str) -> ContextResolution | None:
        """Resolve a missing target using only a compatible recent entity."""
        if get_value("context", "enabled", True) is False:
            return None
        normalized = str(intent).upper()
        allowed_by_intent = {
            "CLOSE_APPLICATION": {"application", "browser"},
            "OPEN_APPLICATION": {"application", "browser"},
            "OPEN_FILE": {"file", "folder", "url"},
            "OPEN_URL": {"url"},
        }
        allowed = allowed_by_intent.get(normalized)
        if not allowed:
            return None
        item = self._find(allowed, include_inactive=normalized.startswith("OPEN"))
        if item is None or item.name is None:
            return None
        return ContextResolution(
            normalized, item.name, {}, self._canonical(normalized, item.name),
            item.entity_type,
        )

    def llm_context(self) -> dict[str, str]:
        """Expose only a minimal active application label for interpretation."""
        item = self._find({"application"})
        if item is None or not item.name:
            return {}
        return {"last_application": item.name}

    def _find(self, allowed: set[str], include_inactive: bool = False) -> ContextItem | None:
        for item in self._items:
            if item.entity_type not in allowed:
                continue
            if not include_inactive and item.entity_type in {"application", "browser", "file"} and not item.active:
                # Inactive apps/files may be reopened, but never closed or deleted again.
                continue
            return item
        return None

    @staticmethod
    def _canonical(intent: str, target: str | None) -> str:
        prefix = {
            "CLOSE_APPLICATION": "close", "DELETE_FILE": "delete",
            "OPEN_FILE": "open", "OPEN_URL": "open",
            "OPEN_APPLICATION": "open", "OPEN_BROWSER": "open browser",
        }.get(intent, "")
        return f"{prefix} {target or ''}".strip()

    def clear(self) -> None:
        self._items.clear()
        self._command_number = 0
        logger.debug("Context cleared")

    def snapshot(self) -> list[dict[str, Any]]:
        """Return safe copies; callers cannot mutate internal context."""
        return [asdict(item) for item in copy.deepcopy(self._items)]
