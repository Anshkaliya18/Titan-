# Changelog

All notable changes to TITAN are documented in this file.

## Unreleased — Windows application discovery

- Added lazy, bounded discovery from user/all-users Start Menu shortcuts and read-only Windows uninstall registry display metadata.
- Added deterministic name and alias resolution, ambiguity reporting, and explicit `find installed apps`, filtered `find applications containing ...`, and `refresh applications` commands.
- Discovered launch targets are constrained to validated `.exe` files under common application roots. Validated Start Menu links are rechecked and activated through Windows so installed shortcut settings work; fallback direct launches use the Application Manager's structured `shell=False` process call. AppX activation without a supported shortcut, arbitrary executable paths, shell commands, and user-supplied app arguments remain unsupported.
- Added application discovery settings and Windows-specific discovery, resolution, and security coverage.

## Unreleased — Step 35 automation and workflows

- Added opt-in, persistent workflows that contain only locally validated
  commands registered by TITAN. Workflow storage uses bounded, atomic JSON
  writes and rejects malformed or tampered records when read.
- Added explicit create, list, inspect, run, enable, disable, and delete CLI
  requests. Workflow runs are rebuilt as fresh TaskPlans and use the existing
  structured execution, confirmation, command history, and manager pipeline.
- Dangerous commands retain one-shot confirmation; a workflow stops at the
  confirmation and never resumes later steps automatically.
- Added safe automation limits and documentation. Scheduling is intentionally
  not implemented; no background persistence, Windows Task Scheduler, shell,
  script execution, autonomous agents, retry, rollback, or parallel execution.

## Unreleased — Step 34 task planning

- Added opt-in, bounded task plans containing only commands registered by TITAN.
- Plans use the existing normalization, natural-language interpretation,
  context, command validation, sequential executor, history, and confirmation
  layers. A plan stops on failure or at a dangerous command; confirmation runs
  only that exact command and does not resume later steps.
- Added plan preview, status, cancellation, dependency-cycle and schema checks,
  safe configuration limits, and tests for malicious/untrusted plan data.
- Planning does not add autonomous agents, retries, rollback, parallel work,
  browser/GUI automation, or arbitrary code execution.

## Unreleased — Step 30 Windows-native speech replacement

- Replaced the active Whisper.cpp transcription provider with bounded,
  local Windows SAPI dictation behind the existing speech-recognizer interface.
- Added locale/device configuration and safe recognition errors. Voice remains
  explicit and transcripts continue through the existing approval and command
  validation pipeline; no audio is saved or sent to a cloud speech API.
- Removed the Whisper setup flow and model requirement while preserving the
  wake-word microphone-to-voice handoff.

## Unreleased — Step 32 local wake word

- Added optional local openWakeWord detection for the pretrained Hey Jarvis
  phrase, with configurable sensitivity and safe-off defaults.
- Integrated wake activation with the existing bounded Windows speech input and
  approval flow; microphone capture closes between wake detection, SAPI,
  command processing, and TTS playback.
- Added explicit model setup, lifecycle/error handling, wake-mode privacy and
  licensing documentation, and detector security/regression tests.

## Unreleased — Step 31 local text-to-speech

- Added opt-in local `pyttsx3`/Windows SAPI speech for final Response Manager
  text, with voice, rate, and volume configuration.
- Kept speech lazy and output-only; TTS failures do not interrupt command
  execution or printed responses, and no permanent audio is stored.
- Added response sanitization, safe configuration fallback, packaging imports,
  and TTS isolation tests.

## Unreleased — Gemini LLM provider

- Replaced the prior LLM adapter with Google's official `google-genai` SDK
  and stable `gemini-3.5-flash-lite` as the configurable Flash-Lite default.
- Added safe authentication, timeout, network, invalid-request, service, and
  rate-limit categorization. Quota errors do not trigger automatic retries;
  TITAN falls back to its deterministic command pipeline.
- Preserved deterministic-first dispatch, registry-derived structured JSON,
  local schema/target checks, validator enforcement, and dangerous-command
  confirmation. Added an isolated manual API diagnostic with no command access.
- Added mocked Gemini provider and updated interpreter/security coverage.

## Unreleased — optional LLM and local Whisper.cpp (Step 31)

- Added an optional hosted LLM adapter with an environment-based API key and
  configurable model. Structured output continued through local schema,
  registry, validator, and confirmation checks.
- Replaced online voice transcription with one-shot local Whisper.cpp CLI
  transcription. Microphone audio is written to a temporary WAV and removed
  after the process; no audio is uploaded or added to TITAN history.
- Added configurable executable/model paths, language, timeout handling, and
  safe missing executable/model errors. Voice and LLM remain disabled by
  default; typed commands work without either provider.
- Added mocked LLM and Whisper.cpp provider tests, including failure,
  cleanup, privacy, and untrusted-output cases. No live API, microphone, or
  system command is used by tests.

## Unreleased (Step 29 — Optional LLM Interpretation)

- Added a provider interface and optional OpenAI Responses API adapter;
  LLM support is disabled by default and requires `OPENAI_API_KEY` only
  when explicitly enabled.
- Added strict registry-derived structured output, local schema and target
  checks, configurable input/output limits, timeouts, and ordinary/dangerous
  confidence thresholds.
- Integrated LLM fallback only after deterministic, hybrid, and explicit
  context resolution miss. Accepted data continues through the existing
  parser, validator, executor, and confirmation flow.
- Provider errors, missing credentials, malformed output, prompt injection,
  and low confidence fail closed to deterministic handling. No function
  calling, shell access, or model confirmation authority was added.
- Added mocked provider, interpreter, integration, and security tests.

## Unreleased (Step 25 — Hybrid Intent Detection)

- Added local ML intent classifier (TF-IDF + Logistic Regression,
  scikit-learn) as an interpretation layer behind the deterministic
  rule-based detector.
- Hybrid decision: rules win when they match; ML fills UNKNOWN gaps;
  low-confidence predictions become UNKNOWN; dangerous intents need
  higher confidence (0.85 vs 0.50).
- New `intent_detection` settings (`ml_enabled`,
  `ml_confidence_threshold`, `ml_dangerous_confidence_threshold`) with
  safe defaults; `ml_enabled: false` restores pure deterministic
  behavior.
- Training dataset (`data/intent_dataset.csv`, 249 examples, 20
  classes) and training script (`scripts/train_intent_model.py`).
- No execution changes: parser, validator, confirmation, and all
  managers operate exactly as before.

## 1.0.0

Initial stable Windows CLI release.

### Application launching
- Open registered applications by name or alias (Chrome, Firefox, Edge,
  Notepad, Calculator, File Explorer, ...)
- Close running applications
- Exact-match application allowlist — no fuzzy executable matching

### File operations
- Create folders, delete files, open files, search files
- All operations restricted to configured safe directories
- Resolved-path containment checks (no string-prefix security)
- Deletion requires explicit confirmation

### Browser operations
- Open the default browser
- Web search via the configured search engine
- Open a specific URL
- Only http/https URL schemes allowed

### Windows system controls
- Lock the computer
- Shutdown / restart (both require explicit confirmation)

### Command pipeline
- Input normalization
- NLTK tokenization
- Deterministic edit-distance spell correction (no ML)
- Rule-based intent detection
- Command parsing and validation
- Multi-command input ("open chrome and open calculator")
- Command history (bounded, in-memory, never replays commands)
- Help and command discovery

### Safety
- Confirmation-protected dangerous operations
- Confirmation authorizes only the exact pending command
- Application allowlist enforced
- Filesystem path containment enforced
- URL scheme validation enforced
- No arbitrary shell execution (no shell=True, no os.system)

### Configuration & logging
- JSON configuration with safe defaults
- Malformed configuration never disables safety controls
- Rotating file logging with secret redaction
- Working-directory independence

### Windows startup (optional)
- Enable/disable starting TITAN with Windows
- User-level HKCU Run key registration
- Confirmation-gated, reversible, no administrator rights

### Packaging
- PyInstaller onedir build (TITAN.exe)
- Bundled NLTK tokenizer data (offline, no downloads)
- Portable distribution — copy the folder, run the executable

# Step 36 — TITAN Desktop GUI

- Added an optional dark Tkinter command center with conversation display,
  keyboard shortcuts, voice input, TTS/wake controls, settings, and optional
  Windows system tray integration.
- GUI text and accepted voice transcripts use the existing single-turn core
  pipeline, including command validation, pending confirmation, execution,
  response formatting, and local TTS.
- Added `python main.py --gui`; the existing CLI remains the default and is
  available explicitly with `--cli`.
- Branded the new desktop interface as TITAN while retaining legacy CLI
  configuration and data-directory identifiers for compatibility.
- Added optional tray dependencies in `requirements-gui.txt`.
