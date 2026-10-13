"""
Configuration Manager Module

Centralized configuration for TITAN. Settings are loaded from
config/settings.json once and kept in memory; get_value() reads
from the loaded configuration, never from disk on every call.

Safety rules:
- Missing or invalid settings.json falls back to safe defaults.
- Safety-sensitive settings always default to the safe value (True).
- A wrong-typed or malformed safety value is treated as True.
- Secrets (API keys, passwords, tokens) are never stored here.

This module only manages configuration data. It never executes
commands, evaluates configuration values, or touches the shell.
"""

import copy
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from core.exceptions import ConfigurationError
from utils import paths
from utils.logger import get_logger

logger = get_logger(__name__)

# Built-in safe defaults — used when settings.json is missing or invalid.
DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "general": {
        "assistant_name": "TITAN",
        "debug": False,
    },
    "browser": {
        "default_search_engine": "https://www.google.com/search?q={query}",
    },
    "filesystem": {
        "workspace": "TITAN_workspace",
    },
    "safety": {
        "require_confirmation_for_shutdown": True,
        "require_confirmation_for_restart": True,
        "require_confirmation_for_delete": True,
        "confirmation_timeout_seconds": 60,
    },
    "history": {
        "max_entries": 100,
    },
    "intent_detection": {
        "ml_enabled": True,
        "ml_confidence_threshold": 0.50,
        "ml_dangerous_confidence_threshold": 0.85,
    },
    "spell_correction": {
        "enabled": True,
        "confidence_threshold": 0.75,
        "dangerous_confidence_threshold": 0.90,
        "context_aware": True,
    },
    "natural_language": {
        "enabled": True,
        "confidence_threshold": 0.75,
        "dangerous_confidence_threshold": 0.90,
    },
    "context": {
        "enabled": True,
        "max_items": 10,
        "max_age_commands": 10,
    },
    "llm": {
        "enabled": False,
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "timeout_seconds": 10,
        "max_input_length": 2000,
        "max_output_length": 2000,
        "confidence_threshold": 0.80,
        "dangerous_confidence_threshold": 0.95,
    },
    "voice": {
        "enabled": False,
        "provider": "windows",
        "language": "en-US",
        "listen_timeout_seconds": 5,
        "phrase_time_limit_seconds": 10,
        "device": None,
    },
    "tts": {
        "enabled": False,
        "provider": "local",
        "voice": None,
        "rate": 175,
        "volume": 1.0,
    },
    "wake_word": {
        "enabled": False,
        "provider": "local",
        "phrase": "hey TITAN",
        "sensitivity": 0.5,
        "sample_rate": 16000,
        "channels": 1,
    },
    "task_planning": {
        "enabled": False,
        "max_steps": 10,
        "max_plan_input_length": 2000,
        "allow_llm_planning": False,
        "require_confirmation_for_plan": False,
    },
    "automation": {
        "enabled": False,
        "max_workflows": 50,
        "max_steps_per_workflow": 10,
        "max_workflow_name_length": 100,
        "max_description_length": 500,
        "max_runs_per_hour": 10,
        "scheduler_enabled": False,
        "skip_missed_runs": True,
    },
    "applications": {
        "discovery_enabled": True,
        "cache_enabled": True,
        "max_results": 25,
        "fuzzy_matching_enabled": True,
        "fuzzy_threshold": 0.90,
    },
}

# Expected types for known settings (used for validation).
_SETTINGS_SCHEMA: dict[tuple[str, str], type] = {
    ("general", "assistant_name"): str,
    ("general", "debug"): bool,
    ("browser", "default_search_engine"): str,
    ("filesystem", "workspace"): str,
    ("safety", "require_confirmation_for_shutdown"): bool,
    ("safety", "require_confirmation_for_restart"): bool,
    ("safety", "require_confirmation_for_delete"): bool,
    ("safety", "confirmation_timeout_seconds"): int,
    ("history", "max_entries"): int,
    ("intent_detection", "ml_enabled"): bool,
    ("intent_detection", "ml_confidence_threshold"): float,
    ("intent_detection", "ml_dangerous_confidence_threshold"): float,
    ("spell_correction", "enabled"): bool,
    ("spell_correction", "confidence_threshold"): float,
    ("spell_correction", "dangerous_confidence_threshold"): float,
    ("spell_correction", "context_aware"): bool,
    ("natural_language", "enabled"): bool,
    ("natural_language", "confidence_threshold"): float,
    ("natural_language", "dangerous_confidence_threshold"): float,
    ("context", "enabled"): bool,
    ("context", "max_items"): int,
    ("context", "max_age_commands"): int,
    ("llm", "enabled"): bool,
    ("llm", "provider"): str,
    ("llm", "model"): str,
    ("llm", "timeout_seconds"): int,
    ("llm", "max_input_length"): int,
    ("llm", "max_output_length"): int,
    ("llm", "confidence_threshold"): float,
    ("llm", "dangerous_confidence_threshold"): float,
    ("voice", "enabled"): bool,
    ("voice", "provider"): str,
    ("voice", "language"): str,
    ("voice", "listen_timeout_seconds"): int,
    ("voice", "phrase_time_limit_seconds"): int,
    ("tts", "enabled"): bool,
    ("tts", "provider"): str,
    ("tts", "rate"): int,
    ("wake_word", "enabled"): bool,
    ("wake_word", "provider"): str,
    ("wake_word", "phrase"): str,
    ("wake_word", "sensitivity"): float,
    ("wake_word", "sample_rate"): int,
    ("wake_word", "channels"): int,
    ("task_planning", "enabled"): bool,
    ("task_planning", "max_steps"): int,
    ("task_planning", "max_plan_input_length"): int,
    ("task_planning", "allow_llm_planning"): bool,
    ("task_planning", "require_confirmation_for_plan"): bool,
    ("automation", "enabled"): bool,
    ("automation", "scheduler_enabled"): bool,
    ("automation", "skip_missed_runs"): bool,
    ("automation", "max_workflows"): int,
    ("automation", "max_steps_per_workflow"): int,
    ("automation", "max_workflow_name_length"): int,
    ("automation", "max_description_length"): int,
    ("automation", "max_runs_per_hour"): int,
    ("applications", "discovery_enabled"): bool,
    ("applications", "cache_enabled"): bool,
    ("applications", "max_results"): int,
    ("applications", "fuzzy_matching_enabled"): bool,
    ("applications", "fuzzy_threshold"): float,
}

# Safety-sensitive keys: a wrong-typed or malformed value is always
# treated as the safe default (True). These can never be disabled
# through a malformed configuration.
_SAFETY_KEYS: set[tuple[str, str]] = {
    ("safety", "require_confirmation_for_shutdown"),
    ("safety", "require_confirmation_for_restart"),
    ("safety", "require_confirmation_for_delete"),
}

# Integers that must be a genuine positive number (bools are
# excluded — isinstance(True, int) is True in Python). A malformed
# value falls back to the safe default, so a bad setting can never
# make confirmations immortal, vanish instantly, or allow a
# section (like command history) to grow without bound.
_POSITIVE_INT_KEYS: set[tuple[str, str]] = {
    ("safety", "confirmation_timeout_seconds"),
    ("history", "max_entries"),
    ("context", "max_items"),
    ("context", "max_age_commands"),
    ("llm", "timeout_seconds"),
    ("llm", "max_input_length"),
    ("llm", "max_output_length"),
    ("voice", "listen_timeout_seconds"),
    ("voice", "phrase_time_limit_seconds"),
    ("task_planning", "max_steps"),
    ("task_planning", "max_plan_input_length"),
    ("automation", "max_workflows"),
    ("automation", "max_steps_per_workflow"),
    ("automation", "max_workflow_name_length"),
    ("automation", "max_description_length"),
    ("automation", "max_runs_per_hour"),
}

# Confidence thresholds must be valid floats between 0 and 1
_CONFIDENCE_THRESHOLD_KEYS: set[tuple[str, str]] = {
    ("intent_detection", "ml_confidence_threshold"),
    ("intent_detection", "ml_dangerous_confidence_threshold"),
    ("spell_correction", "confidence_threshold"),
    ("spell_correction", "dangerous_confidence_threshold"),
    ("natural_language", "confidence_threshold"),
    ("natural_language", "dangerous_confidence_threshold"),
    ("llm", "confidence_threshold"),
    ("llm", "dangerous_confidence_threshold"),
    ("applications", "fuzzy_threshold"),
}


def _settings_path() -> Path:
    """
    Return the fixed path to settings.json.

    The location comes from utils.paths (Step 23): the project's
    config/settings.json in source mode, the writable per-user
    %APPDATA%\\TITAN\\config\\settings.json when packaged (or the
    folder named by TITAN_DATA_DIR when that is set). It is always an
    absolute path computed from the installation layout — never from
    the working directory and never from user input — so configuration
    path traversal is impossible.
    """
    return paths.settings_path()


def _load_environment() -> None:
    """Load optional secrets from the fixed writable data root."""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=paths.data_root() / ".env", override=False)
    except ImportError:
        # Keep startup usable if dependencies have not yet been installed.
        logger.warning("python-dotenv is unavailable; skipping .env loading")
    except Exception as exc:
        # Avoid logging environment contents or exception messages from parsers.
        logger.warning(f"Could not load .env ({type(exc).__name__})")


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base.

    Nested dictionaries are merged key-by-key; all other values are
    replaced. Neither input is mutated.
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _validate_and_fix(config: dict) -> dict:
    """
    Validate configuration value types and repair unsafe values.

    - Known settings with the wrong type are reset to the safe default.
    - Safety-sensitive keys that are missing or not a bool are forced
      to True so a malformed configuration can never disable a
      protection.
    - Safety-sensitive integers that are not a genuine positive number
      (including bools) fall back to the safe default.
    """
    config["wake_word"] = copy.deepcopy(
        config.get("wake_word", DEFAULT_CONFIG["wake_word"])
    )
    raw_wake = config["wake_word"]
    wake_invalid = not isinstance(raw_wake, dict)
    if isinstance(raw_wake, dict):
        phrase = raw_wake.get("phrase", "hey TITAN")
        sensitivity = raw_wake.get("sensitivity", 0.5)
        wake_invalid = (
            not isinstance(raw_wake.get("enabled", False), bool)
            or raw_wake.get("provider", "local") != "local"
            or not isinstance(phrase, str)
            or " ".join(phrase.casefold().split()) != "hey TITAN"
            or not isinstance(sensitivity, float)
            or isinstance(sensitivity, bool)
            or not 0.1 <= sensitivity <= 0.95
            or type(raw_wake.get("sample_rate", 16000)) is not int
            or raw_wake.get("sample_rate", 16000) != 16000
            or type(raw_wake.get("channels", 1)) is not int
            or raw_wake.get("channels", 1) != 1
        )

    config["tts"] = copy.deepcopy(
        config.get("tts", DEFAULT_CONFIG["tts"])
    )
    raw_tts = config.get("tts", DEFAULT_CONFIG["tts"])
    tts_invalid = not isinstance(raw_tts, dict)
    if isinstance(raw_tts, dict):
        tts_invalid = (
            not isinstance(raw_tts.get("enabled", False), bool)
        ) or (
            raw_tts.get("provider", "local") != "local"
        ) or (
            raw_tts.get("voice") is not None
            and (not isinstance(raw_tts.get("voice"), str)
                 or not raw_tts.get("voice", "").strip())
        ) or (
            not isinstance(raw_tts.get("rate", 175), int)
            or isinstance(raw_tts.get("rate", 175), bool)
            or not 80 <= raw_tts.get("rate", 175) <= 300
        ) or (
            not isinstance(raw_tts.get("volume", 1.0), (int, float))
            or isinstance(raw_tts.get("volume", 1.0), bool)
            or not 0.0 <= raw_tts.get("volume", 1.0) <= 1.0
        )

    for (section, key), expected_type in _SETTINGS_SCHEMA.items():
        section_data = config.get(section)
        if not isinstance(section_data, dict):
            continue
        if key in section_data and not isinstance(section_data[key], expected_type):
            logger.warning(
                f"Config {section}.{key} has wrong type "
                f"({type(section_data[key]).__name__}), using safe default"
            )
            section_data[key] = DEFAULT_CONFIG[section][key]

    # Force safety keys to a valid bool, defaulting to True.
    for section, key in _SAFETY_KEYS:
        section_data = config.setdefault(section, {})
        if not isinstance(section_data.get(key), bool):
            if key in section_data:
                logger.warning(
                    f"Safety setting {section}.{key} is not a bool, "
                    f"forcing to True"
                )
            section_data[key] = True

    # Force positive-integer settings to a genuine positive number.
    for section, key in _POSITIVE_INT_KEYS:
        section_data = config.setdefault(section, {})
        value = section_data.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            if key in section_data:
                logger.warning(
                    f"Setting {section}.{key} must be a positive "
                    f"int, using safe default"
                )
            section_data[key] = DEFAULT_CONFIG[section][key]

    # Task planning has hard upper bounds independent of user config.
    if not isinstance(config.get("task_planning"), dict):
        logger.warning("Invalid task_planning section; using safe defaults")
        config["task_planning"] = copy.deepcopy(DEFAULT_CONFIG["task_planning"])
    task_planning = config.setdefault("task_planning", {})
    for key, maximum in (("max_steps", 10), ("max_plan_input_length", 2000)):
        value = task_planning.get(key, DEFAULT_CONFIG["task_planning"][key])
        if type(value) is not int or not 1 <= value <= maximum:
            logger.warning("Invalid task_planning.%s; using safe default", key)
            task_planning[key] = DEFAULT_CONFIG["task_planning"][key]

    if not isinstance(config.get("automation"), dict):
        logger.warning("Invalid automation section; using safe defaults")
        config["automation"] = copy.deepcopy(DEFAULT_CONFIG["automation"])
    automation = config["automation"]
    for key, maximum in (("max_workflows", 50), ("max_steps_per_workflow", 10),
                         ("max_workflow_name_length", 100),
                         ("max_description_length", 500),
                         ("max_runs_per_hour", 10)):
        value = automation.get(key, DEFAULT_CONFIG["automation"][key])
        if type(value) is not int or not 1 <= value <= maximum:
            logger.warning("Invalid automation.%s; using safe default", key)
            automation[key] = DEFAULT_CONFIG["automation"][key]
    for key in ("enabled", "scheduler_enabled", "skip_missed_runs"):
        if type(automation.get(key, DEFAULT_CONFIG["automation"][key])) is not bool:
            logger.warning("Invalid automation.%s; using safe default", key)
            automation[key] = DEFAULT_CONFIG["automation"][key]
    # No scheduler is provided in this release. A malformed or edited setting
    # cannot turn on a scheduler implementation that does not exist.
    automation["scheduler_enabled"] = False

    applications = config.get("applications")
    if not isinstance(applications, dict):
        logger.warning("Invalid applications section; using safe defaults")
        applications = copy.deepcopy(DEFAULT_CONFIG["applications"])
        config["applications"] = applications
    for key in ("discovery_enabled", "cache_enabled", "fuzzy_matching_enabled"):
        if type(applications.get(key, DEFAULT_CONFIG["applications"][key])) is not bool:
            logger.warning("Invalid applications.%s; using safe default", key)
            applications[key] = DEFAULT_CONFIG["applications"][key]
    max_results = applications.get("max_results", 25)
    if type(max_results) is not int or not 1 <= max_results <= 100:
        logger.warning("Invalid applications.max_results; using safe default")
        applications["max_results"] = 25
    threshold = applications.get("fuzzy_threshold", .90)
    if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
            or not .90 <= threshold <= 1.0):
        logger.warning("Invalid applications.fuzzy_threshold; using safe default")
        applications["fuzzy_threshold"] = .90
    # This product feature has a security floor irrespective of config.
    applications["fuzzy_threshold"] = max(.90, float(applications["fuzzy_threshold"]))

    # Validate confidence thresholds: must be float between 0 and 1
    for section, key in _CONFIDENCE_THRESHOLD_KEYS:
        section_data = config.setdefault(section, {})
        value = section_data.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            if key in section_data:
                logger.warning(
                    f"Setting {section}.{key} must be a float, using safe default"
                )
            section_data[key] = DEFAULT_CONFIG[section][key]
        else:
            # Clamp to [0, 1] range
            if value < 0.0 or value > 1.0:
                logger.warning(
                    f"Setting {section}.{key} must be between 0 and 1, "
                    f"clamping to safe range"
                )
                section_data[key] = max(0.0, min(1.0, float(value)))

    # LLM thresholds can never be configured below the security floor
    # for dangerous actions. Even an accidental low setting is repaired.
    llm = config.setdefault("llm", {})
    if llm.get("dangerous_confidence_threshold", 0) < 0.95:
        llm["dangerous_confidence_threshold"] = 0.95

    # Bound remote request size and duration independently of settings.
    for key, maximum in (("timeout_seconds", 30),
                         ("max_input_length", 10000),
                         ("max_output_length", 4000)):
        value = llm.get(key, DEFAULT_CONFIG["llm"][key])
        llm[key] = min(value, maximum)

    if not llm.get("provider", "").strip() or not llm.get("model", "").strip():
        llm["enabled"] = False

    # Migrate the previous Whisper settings to the Windows provider. Voice
    # remains opt-in and is still entered only by the explicit `voice` command.
    voice = config.get("voice")
    if not isinstance(voice, dict):
        voice = copy.deepcopy(DEFAULT_CONFIG["voice"])
        config["voice"] = voice
        voice["enabled"] = False
    if voice.get("provider") == "whisper_cpp":
        voice["provider"] = "windows"
        if voice.get("language", "en") == "en":
            voice["language"] = "en-US"
        voice.pop("executable_path", None)
        voice.pop("model_path", None)
    voice.setdefault("provider", "windows")
    voice.setdefault("language", "en-US")
    voice.setdefault("device", None)
    voice.setdefault("listen_timeout_seconds", DEFAULT_CONFIG["voice"]["listen_timeout_seconds"])
    voice.setdefault("phrase_time_limit_seconds", DEFAULT_CONFIG["voice"]["phrase_time_limit_seconds"])
    for key, maximum in (("listen_timeout_seconds", 30),
                         ("phrase_time_limit_seconds", 30)):
        voice[key] = min(voice[key], maximum)
    if (
        voice.get("provider") != "windows"
        or not isinstance(voice.get("language"), str)
        or not re.fullmatch(r"[A-Za-z]{2,3}-[A-Za-z]{2,4}", voice["language"])
        or (
            voice["device"] is not None
            and (type(voice["device"]) is not int or not 0 <= voice["device"] <= 255)
        )
    ):
        voice["enabled"] = False
        voice.update(copy.deepcopy(DEFAULT_CONFIG["voice"]))
        voice["enabled"] = False

    # TTS is output-only and opt-in. Any malformed or unsupported setting
    # disables it rather than affecting command handling.
    tts = config.get("tts")
    if not isinstance(tts, dict):
        tts = copy.deepcopy(DEFAULT_CONFIG["tts"])
        config["tts"] = tts
        tts_invalid = True
    if tts.get("provider") != "local":
        tts["enabled"] = False
    if not isinstance(tts.get("voice"), (str, type(None))) or (
        isinstance(tts.get("voice"), str) and not tts["voice"].strip()
    ):
        tts["enabled"] = False
        tts["voice"] = None
    rate = tts.get("rate")
    volume = tts.get("volume")
    if (
        not isinstance(rate, int) or isinstance(rate, bool)
        or not 80 <= rate <= 300
        or not isinstance(volume, (int, float)) or isinstance(volume, bool)
        or not 0.0 <= volume <= 1.0
    ):
        logger.warning("Invalid TTS settings; disabling speech")
        tts["enabled"] = False
        tts.update(DEFAULT_CONFIG["tts"])
    if tts_invalid:
        logger.warning("Malformed TTS configuration; speech disabled")
        tts["enabled"] = False

    wake = config.get("wake_word")
    if not isinstance(wake, dict):
        wake = copy.deepcopy(DEFAULT_CONFIG["wake_word"])
        config["wake_word"] = wake
        wake_invalid = True
    if wake_invalid:
        logger.warning("Malformed wake-word configuration; mode disabled")
        wake.update(DEFAULT_CONFIG["wake_word"])
        wake["enabled"] = False

    return config


def _write_defaults_file(path: Path) -> None:
    """Write the default configuration to settings.json (best effort)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        logger.info("Created default settings.json")
    except OSError as e:
        logger.error(f"Could not create settings.json: {e}")


def _seed_from_bundled_defaults(path: Path) -> dict | None:
    """
    First launch of a packaged build: seed the writable settings file
    from the read-only default bundled with the application (Step 23).

    Only used when running frozen, where the bundled template and the
    user's writable file are different locations. Every failure — a
    missing template, malformed JSON, a non-object template, an
    unwritable target — returns None so the caller falls back to the
    built-in safe defaults, which keeps protections intact.

    Args:
        path: The writable settings location to create.

    Returns:
        The parsed template dictionary, or None when seeding was not
        possible (the caller then uses DEFAULT_CONFIG).
    """
    template = paths.default_settings_path()
    try:
        if template.resolve() == path.resolve():
            return None
        with open(template, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info("Created settings.json from the bundled defaults")
        return data
    except (OSError, ValueError) as e:
        logger.warning(f"Could not seed settings.json from bundle: {e}")
        return None


def _load_config() -> dict:
    """
    Load configuration from settings.json, falling back to defaults.

    - Missing file: use defaults and create the file as a template.
    - Invalid JSON: log the error and use defaults.
    - Valid JSON: deep-merged over defaults, then type-validated.
    """
    path = _settings_path()

    if not path.exists():
        logger.info("settings.json not found, using defaults")
        # Packaged first launch: seed the writable file from the
        # bundled default template, and use it immediately. In source
        # mode this is skipped — there the project file is the source
        # of truth and the built-in defaults apply.
        if paths.is_frozen():
            seeded = _seed_from_bundled_defaults(path)
            if seeded is not None:
                return _validate_and_fix(_deep_merge(DEFAULT_CONFIG, seeded))
        _write_defaults_file(path)
        return _deep_merge(DEFAULT_CONFIG, {})

    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in settings.json: {e}, using defaults")
        return _deep_merge(DEFAULT_CONFIG, {})
    except OSError as e:
        logger.error(f"Could not read settings.json: {e}, using defaults")
        return _deep_merge(DEFAULT_CONFIG, {})

    if not isinstance(loaded, dict):
        logger.error("settings.json must contain a JSON object, using defaults")
        return _deep_merge(DEFAULT_CONFIG, {})

    config = _deep_merge(DEFAULT_CONFIG, loaded)
    config = _validate_and_fix(config)
    logger.info("Configuration loaded from settings.json")
    return config


# Load secrets before providers read environment variables.
_load_environment()

# Module-level configuration, loaded once at import.
_CONFIG: dict[str, dict[str, Any]] = _load_config()


def reload_config() -> dict:
    """
    Reload configuration from settings.json into memory.

    Returns:
        dict: The freshly loaded configuration.
    """
    global _CONFIG
    _CONFIG = _load_config()
    return _CONFIG


def get_config() -> dict:
    """
    Return a deep copy of the full configuration.

    Returns:
        dict: The complete configuration.
    """
    return copy.deepcopy(_CONFIG)


def get_value(section: str, key: str, default: Any = None) -> Any:
    """
    Get a single configuration value.

    Reads from the in-memory configuration loaded at startup.

    Args:
        section: The configuration section, e.g. "general".
        key: The setting name, e.g. "debug".
        default: Value returned when the setting is absent.

    Returns:
        The configured value, or default.
    """
    return _CONFIG.get(section, {}).get(key, default)


def get_section(section: str) -> dict:
    """
    Return a copy of a configuration section.

    Args:
        section: The configuration section, e.g. "browser".

    Returns:
        dict: A copy of the section, or an empty dict if absent.
    """
    return dict(_CONFIG.get(section, {}))


def set_value(section: str, key: str, value: Any) -> None:
    """
    Set a configuration value and persist it atomically.

    The value is validated against the known schema before saving.
    Safety-sensitive keys are never allowed to disable a protection
    through this method — they are forced to True.

    Args:
        section: The configuration section.
        key: The setting name.
        value: The new value.

    Raises:
        TypeError: If the value's type does not match the schema.
    """
    schema_key = (section, key)

    if schema_key in _SETTINGS_SCHEMA:
        expected = _SETTINGS_SCHEMA[schema_key]
        if not isinstance(value, expected):
            raise TypeError(
                f"{section}.{key} must be {expected.__name__}, "
                f"got {type(value).__name__}"
            )

    # Safety settings can only ever be True — never disable a protection.
    if schema_key in _SAFETY_KEYS and value is not True:
        logger.warning(
            f"Refusing to disable safety setting {section}.{key}; "
            f"forcing to True"
        )
        value = True

    # Safety-sensitive integers must be genuine positive numbers.
    if schema_key in _POSITIVE_INT_KEYS and (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
    ):
        raise ValueError(
            f"{section}.{key} must be a positive integer, "
            f"got {value!r}"
        )

    if section == "task_planning" and key in {
        "max_steps", "max_plan_input_length"
    }:
        maximum = 10 if key == "max_steps" else 2000
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"task_planning.{key} must be between 1 and {maximum}")

    if section == "wake_word":
        if key == "provider" and value != "local":
            raise ValueError("wake_word.provider only supports 'local'")
        if key == "phrase" and (
            not isinstance(value, str)
            or " ".join(value.casefold().split()) != "hey TITAN"
        ):
            raise ValueError("The installed wake model only supports 'hey TITAN'")
        if key == "sensitivity" and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.1 <= value <= 0.95
        ):
            raise ValueError("wake_word.sensitivity must be between 0.1 and 0.95")
        if key == "sample_rate" and (type(value) is not int or value != 16000):
            raise ValueError("openWakeWord requires a 16000 Hz sample rate")
        if key == "channels" and (type(value) is not int or value != 1):
            raise ValueError("openWakeWord requires one microphone channel")

    if section == "voice":
        if key == "provider" and value != "windows":
            raise ValueError("voice.provider only supports 'windows'")
        if key == "language" and (
            not isinstance(value, str)
            or not re.fullmatch(r"[A-Za-z]{2,3}-[A-Za-z]{2,4}", value)
        ):
            raise ValueError("voice.language must be a locale such as 'en-US'")
        if key == "device" and (
            value is not None
            and (type(value) is not int or not 0 <= value <= 255)
        ):
            raise ValueError("voice.device must be null or a Windows input device ID")

    if section not in _CONFIG:
        _CONFIG[section] = {}
    _CONFIG[section][key] = value
    _save_config()
    logger.info(f"Configuration updated: {section}.{key}")


def _save_config() -> None:
    """
    Persist the in-memory configuration to settings.json atomically.

    Writes to a temporary file in the same directory, then replaces
    the original, so an interrupted write cannot corrupt the file.

    Raises:
        ConfigurationError: If the configuration cannot be written.
    """
    path = _settings_path()
    tmp_name = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=".settings-", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_CONFIG, f, indent=4)
        os.replace(tmp_name, path)
    except Exception as e:
        # Clean up the temporary file on any failure
        if tmp_name is not None:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
        logger.error("Failed to persist configuration", exc_info=True)
        raise ConfigurationError(
            f"Could not save settings.json: {e}"
        ) from e
