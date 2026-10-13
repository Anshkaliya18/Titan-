# Jarvis

A deterministic-first Windows CLI assistant written in Python, with
optional Google Gemini interpretation and one-shot local Windows SAPI
speech-to-text. Both provider features are disabled by default.

## Features

- **Application launching** — open/close registered apps (Chrome, Edge,
  Notepad, Calculator, File Explorer, ...) by name or alias
- **File operations** — create folders, delete files, open files, search
  files (restricted to safe directories)
- **Browser operations** — open the default browser, web search, open a
  URL
- **Windows system controls** — lock, shutdown, restart
- **Command history** — bounded, in-memory, never replays commands
- **Help & discovery** — `help`, `commands`, per-command help
- **Multi-command input** — "open chrome and open calculator"
- **Task planning (Step 34)** — an opt-in, bounded plan of structured Jarvis
  commands; execution still uses the existing validator, executor, history,
  context, and confirmation flow
- **Spell correction** — conservative, context-aware command and
  application typo correction from the command and application registries;
  preserves search text, URLs, paths, extensions, and uncertain words
- **Natural-language commands** — deterministic local patterns support
  polite phrasing and known command synonyms, then pass structured intents
  and targets through the existing parser, validator, and confirmation flow
- **Short-term context** — recent successful app, file, URL, browser, and web
  search commands can resolve explicit references such as `close it` or
  `search again`; context stays bounded and in memory for the current session
- **Configuration** — JSON settings with safe defaults
- **Optional LLM interpretation** — disabled by default; strict structured
  output is checked locally and then continues through Jarvis's normal
  parser, validator, executor, and confirmation flow
- **Optional voice input** — explicit one-shot microphone capture converts
  speech to text locally with Windows SAPI, then uses the same command pipeline
  as typed input
- **Optional wake word** — local “Hey Jarvis” detection activates bounded
  Windows speech input only after the user starts wake mode
- **Optional local text-to-speech** — speaks final responses through the
  installed Windows SAPI voices; disabled by default
- **Logging** — rotating file log with secret redaction
- **Windows startup** — optional start-with-Windows support

## Task Planning

Task planning is disabled by default. When enabled, Jarvis can represent a
request such as “create a folder called project and open it” as ordered
`CREATE_FOLDER` and `OPEN_FILE` steps. `plan this task but don't execute it:
open chrome and open github` displays a validated preview without executing it.
The supported bare-site shortcut `open github` resolves to
`https://github.com`.

Each plan has a session-only ID, ordered steps, dependencies, status, and
per-step results. Plans are bounded to ten steps and 2,000 input characters.
They stop on validation or execution failure. A dangerous command enters the
existing confirmation flow; confirming executes only that exact stored
command, then the plan stops without running later steps. History continues to
record the actual user request and each command outcome. Context may resolve
typed references, but neither context nor optional Gemini interpretation can
authorize a command.

Configuration defaults:

```json
"task_planning": {
  "enabled": false,
  "max_steps": 10,
  "max_plan_input_length": 2000,
  "allow_llm_planning": false,
  "require_confirmation_for_plan": false
}
```

When `allow_llm_planning` is enabled, the existing Gemini interpreter may
interpret individual steps only when Gemini is separately enabled. Its output
still passes local command and plan validation. There is no task-level remote
plan generator. Setting `require_confirmation_for_plan` to `true` blocks plan
execution; Jarvis does not let one `yes` approve every command in a plan.
Dangerous steps use their existing exact-command confirmation flow.

Task Planning does not provide autonomous agents, long-running background
agents, automatic retries, rollback, parallel execution, browser or GUI
automation, unrestricted tool calling, or arbitrary code execution. It uses
only capabilities already registered in Jarvis.

---

## 1. Run from source

Requirements: Python 3.11+ on Windows.

```bat
python -m pip install -r requirements.txt
python main.py
```

The first interactive prompt appears after a short initialization
(~0.4 s on a typical machine). Type `help` at the `Jarvis >` prompt to
see available commands.

## 2. Install dependencies

Core runtime dependencies are listed in `requirements.txt`. Gemini support
uses Google's official `google-genai` Python SDK and is only requested when
local interpretation cannot confidently understand a command.

```bat
python -m pip install -r requirements.txt
```

NLTK data needed at runtime (source mode only — the packaged build
bundles it, see §5):

```bat
python -m nltk.downloader punkt_tab
```

## Environment Variables

Jarvis reads `GEMINI_API_KEY` from the process environment or a local `.env`
file during configuration initialization. In source mode, `.env` belongs in
the project root. In a packaged build, put it in `%APPDATA%\Jarvis`. Start
with `.env.example`; `.env` is ignored by Git. Jarvis never writes or logs the
key, and process environment values take precedence over `.env` values.

## Gemini Setup

Gemini is optional and disabled by default in built-in configuration. Install
dependencies with:

```powershell
python -m pip install -r requirements.txt
```

Add your Google AI Studio API key to the ignored `.env` file:

```env
GEMINI_API_KEY=your_key_here
```

Enable it in `config/settings.json` when ready:

```json
{
  "llm": {
    "enabled": true,
    "provider": "gemini",
    "model": "gemini-3.5-flash-lite"
  }
}
```

`gemini-3.5-flash-lite` is Google's current stable Flash-Lite model for
high-throughput, low-cost requests. Change `llm.model` to another supported
model if needed. To test API access without running any Jarvis commands:

```powershell
python scripts/test_gemini.py
```

Jarvis calls Gemini only after deterministic, natural-language, hybrid-intent,
and context handling cannot resolve a single command. Google API/network
failures, missing credentials, malformed output, and rate limits return to the
deterministic pipeline; requests are not automatically retried on quota errors.
Check current project limits in Google AI Studio if requests are rate-limited.

Gemini receives only the request and minimal relevant application context. It
does not receive history, files, logs, environment variables, or configuration.
It returns schema-constrained JSON only; Jarvis treats that data as untrusted
and checks it against the Command Registry and existing validator. Gemini has
no executor, manager, shell, subprocess, or file access. Dangerous commands
still require Jarvis's local confirmation flow. No API request is made while
`llm.enabled` is false.

## Windows Speech Input

Voice input is optional and off in the built-in configuration. It uses the
installed Windows SAPI dictation recognizer locally; no Whisper executable,
model download, API key, or cloud speech service is used. Windows 10/11 must
have a SAPI speech recognizer installed for the configured locale (the current
configuration uses `en-US`). Jarvis reports a clear error if the recognizer or
locale is unavailable.

Install the optional Windows speech bridge and the microphone package used by
wake-word detection:

```powershell
python -m pip install -r requirements-voice.txt
```

Relevant settings in `config/settings.json` are:

```json
{
  "voice": {
    "enabled": false,
    "provider": "windows",
    "language": "en-US",
    "listen_timeout_seconds": 5,
    "phrase_time_limit_seconds": 10,
    "device": null
  }
}
```

Set `voice.enabled` to `true`, then type `voice` to start one bounded capture.
`device` is `null` for the Windows default microphone or a Windows multimedia
input-device ID. Language must be an installed Windows locale such as `en-US`.
Microphone privacy permission and the selected input device are controlled in
Windows Settings. No microphone is opened at startup or while Jarvis waits for
typed input. Captured audio is processed by SAPI in memory and is not saved or
passed to Gemini.

Jarvis displays the recognized transcript and asks you to type `y` before it
continues. Recognized text is untrusted and enters the same normalization,
spell correction, intent/context/Gemini handling, parser, validator, executor,
confirmation, history, response, and TTS pipeline as typed input. Speaking a
command never confirms a dangerous operation. Wake-word mode uses the same
Windows speech provider after the local wake detector releases its microphone.

If recognition is unavailable, install the Windows speech language through
Settings > Time & language > Speech, confirm microphone access, and restart
Jarvis. Dictation uses the installed local SAPI recognizer; no cloud speech API
is configured by Jarvis.

## Wake Word

Wake detection uses [openWakeWord 0.6.0](https://github.com/w-marco/openwakeword)
with its pretrained `hey_jarvis_v0.1` ONNX model and ONNX Runtime. Detection is
local. The code is Apache-2.0, while the pretrained model is CC BY-NC-SA 4.0;
that model license is non-commercial and share-alike. The included pretrained
model should not be used in a commercial distribution without an appropriately
licensed alternative.

Install the optional microphone and wake detector packages, then download the
model assets once while online:

```powershell
python -m pip install -r requirements-voice.txt
python scripts/setup_wake_word.py
```

The download is an explicit setup action. Jarvis does not download models or
contact a wake-word service at runtime. The following settings keep wake mode
off until both opt-in settings are enabled:

```json
{
  "voice": { "enabled": true },
  "wake_word": {
    "enabled": true,
    "provider": "local",
    "phrase": "hey jarvis",
    "sensitivity": 0.5,
    "sample_rate": 16000,
    "channels": 1
  }
}
```

The shipped model recognizes only “Hey Jarvis”; the configured phrase is
validated against that model, and custom phrases are not supported. Sensitivity
is adjustable from 0.1 to 0.95 (default 0.5). Audio is sampled as single-channel
16 kHz PCM in memory. Type `wake` to enter visible wake mode; say “Hey Jarvis,”
then give one command (Mode A). Wake-plus-command in one utterance is not
supported. Windows speech recognition starts only after detection and uses the bounded
voice timeouts, and still asks you to approve its transcript by typing `y`.
That transcript then follows the normal validator and confirmation flow.

While waiting, Jarvis opens the microphone for local 80 ms frames. It closes
that stream as soon as the wake phrase is detected, before SAPI opens its
one-shot capture. The stream is also closed on errors and Ctrl+C; Ctrl+C stops
wake mode and returns to typed input. Since the microphone is closed during
command processing and TTS playback, Jarvis cannot hear its own spoken reply as
a new wake. If the model is missing, rerun the setup script. If the microphone
cannot open at 16 kHz, check the selected Windows input device. Wake mode adds
local ONNX inference while it is active; no transcription or Gemini work runs just to
wait for the phrase.

## Text-to-Speech

Jarvis can speak its final, formatted responses using local Windows SAPI
voices through `pyttsx3`. Speech is disabled by default; typed and voice input
continue working normally while it is off. Install dependencies with
`python -m pip install -r requirements.txt`, then configure
`config/settings.json`:

```json
{
  "tts": {
    "enabled": true,
    "provider": "local",
    "voice": null,
    "rate": 175,
    "volume": 1.0
  }
}
```

Set `enabled` to `false` to disable speech. `voice` may be `null` to use the
Windows default, or the exact name/ID of an installed voice. Jarvis does not
assume a particular voice is installed. `rate` accepts 80–300 words per
minute; `volume` accepts 0.0–1.0. Invalid TTS settings disable speech safely.
If speech initialization or playback fails, Jarvis logs a safe diagnostic and
continues to display the response and process commands normally. TTS speaks
only the Response Manager's final response; debug output and command structures
are excluded. Speech is synthesized locally: response text is not sent to a
speech service, and no audio files are retained. TTS has no command execution
access; dangerous commands still require the existing confirmation step.

Development/test dependencies (optional, for contributors):

```bat
python -m pip install -r requirements-dev.txt
```

## 3. Build the Windows executable

Build dependency (build machine only — not needed to run Jarvis):

```bat
python -m pip install "pyinstaller>=6"
python build.py
```

`build.py` runs the complete test suite first, then builds with
PyInstaller, verifies the result, and prints the output location.
Use `python build.py --skip-tests` to skip the tests during quick
iteration.

## 4. Build output

```
dist\Jarvis\Jarvis.exe        the packaged application
dist\Jarvis\_internal\        bundled libraries and data files
```

To build directly without the script:

```bat
python -m PyInstaller jarvis.spec
```

## 5. Run the packaged version

```bat
"dist\Jarvis\Jarvis.exe"
```

The packaged build is a portable one-directory distribution: copy the
whole `dist\Jarvis` folder anywhere and run `Jarvis.exe`. It needs no
Python installation and downloads nothing at startup (the required
NLTK tokenizer data is bundled).

Flags (print and exit, do not start the interactive loop):

```bat
Jarvis.exe --version
Jarvis.exe --help
```

## 6. Enable Windows startup

At the Jarvis prompt:

```
Jarvis > enable startup
```

Jarvis asks for confirmation, then registers **this installation** in
the per-user Run key:

```
HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run
```

The entry is named `Jarvis` and points at the running executable (or
`python main.py` in source mode). No administrator rights are required,
and no other startup entries are touched.

Aliases: `enable startup`, `start with windows`, `enable windows startup`.

## 7. Disable Windows startup

```
Jarvis > disable startup
```

Removes only the `Jarvis` Run value; your configuration, logs, and all
other startup entries are left untouched. Disabling when nothing is
registered is a safe no-op.

Aliases: `disable startup`, `stop startup`, `disable windows startup`.

Check the current state at any time:

```
Jarvis > startup status
```

Aliases: `startup status`, `check startup`, `is startup enabled`.

## Windows application discovery

On Windows, Jarvis supplements its trusted built-in application registry with a lazy, in-memory index of applications referenced by the current user's and all users' Start Menu Programs shortcuts and by read-only uninstall registry `DisplayName`/`DisplayIcon` metadata. It does not scan disks or running processes. Shortcuts are resolved through optional Windows COM support when available; discovery gracefully skips shortcuts it cannot resolve. Registry metadata without a safely validated executable is ignored.

Use commands such as `open discord`, `open vscode`, `open visual studio code`, or `find applications containing studio`. `find installed apps` lists a bounded set of results and `refresh applications` rebuilds the index. If a name matches multiple installed applications, Jarvis lists the candidates and asks which one to open. `open visual studio` stays ambiguous when both Visual Studio products are present.

Discovery results are cached in memory and capped. When a validated Start Menu shortcut exists, Jarvis revalidates its location and executable target, then asks Windows to activate that exact registered shortcut so the app's installed launch settings work (including packaged web apps). Jarvis never appends user-supplied arguments. Apps without a usable shortcut launch through the existing Application Manager with a structured executable argument and `shell=False`. Built-in static applications retain priority. Native Windows Store/AppX identifiers without a supported Start Menu shortcut are not supported.

Discovery options are in `config/settings.json` under `applications`: `discovery_enabled`, `cache_enabled`, `max_results` (1–100), `fuzzy_matching_enabled`, and `fuzzy_threshold` (minimum 0.90). Malformed values fall back to safe defaults. Disabling discovery leaves the static application registry available. On non-Windows systems, the discovery module imports safely and returns no Windows-discovered applications.

## 8. Automation & workflows (Step 35)

Automation is opt-in. Set `automation.enabled` to `true` in
`config/settings.json`, then restart Jarvis. Create a workflow with an
explicit request, for example:

```
Jarvis > create a workflow called development setup that opens Chrome, opens GitHub, and opens Notepad
Jarvis > list workflows
Jarvis > show workflow development setup
Jarvis > run workflow development setup
Jarvis > disable workflow development setup
Jarvis > enable workflow development setup
Jarvis > delete workflow development setup
```

Creation shows the validated steps that were saved. Workflows contain only
registered Jarvis commands and are stored as bounded JSON in `data/workflows.json`
(or under the configured per-user data directory in a packaged build). They
are revalidated when loaded, enabled, and run. A run is converted into a fresh
TaskPlan and uses the existing command validator, confirmation flow, executor,
managers, context, and command history. Execution is sequential and stops at
the first failure or confirmation request. Confirmation authorizes only the
exact pending command for that run; remaining steps are not resumed afterward.
Workflow creation uses local deterministic command interpretation and does not
send workflow text to Gemini.

The `automation` defaults are `enabled: false`, at most 50 workflows, 10
steps per workflow, 100-character names, 500-character descriptions, and a
reserved limit of 10 runs per hour for a future scheduler. Scheduling is not
implemented in this release; therefore there are no background jobs, Windows
Task Scheduler entries, hidden services, or missed-run replays. Workflows cannot store or run shell commands, scripts,
Python, PowerShell, arbitrary code, or permanent confirmation state. Jarvis
does not generate workflows autonomously or edit them without an explicit
workflow command.

## 9. Where configuration is stored

| Mode | Settings file |
|---|---|
| Source | `config\settings.json` in the project folder (edited in place) |
| Packaged | `%APPDATA%\Jarvis\config\settings.json` |

The packaged build seeds its settings from the bundled default
template on first launch. Set the `JARVIS_DATA_DIR` environment
variable to relocate the writable data folder in either mode.

## 10. Where logs are stored

| Mode | Log file |
|---|---|
| Source | `logs\jarvis.log` in the project folder |
| Packaged | `%APPDATA%\Jarvis\logs\jarvis.log` |

Logs are append-only, rotated at 5 MB (3 backups), and never written
into the read-only bundle directory.

## 11. Verify the installation

```bat
Jarvis.exe --version        expect: "Jarvis 1.0.0"
Jarvis --version            (same, from source)
```

Then start it and try the safe commands:

```
Jarvis > help
Jarvis > commands
Jarvis > startup status
Jarvis > history
```

See `docs/VALIDATION.md` for the full repeatable checklist.

## 12. Uninstall / clean up

The packaged build is fully portable — there is no installer and no
system-wide change beyond the optional startup entry:

1. If startup is enabled, run `disable startup` (or delete the `Jarvis`
   value from the HKCU Run key in `regedit`).
2. Delete the `dist\Jarvis` folder.
3. (Optional) Delete `%APPDATA%\Jarvis` to remove user settings and
   logs. Your own files are never touched.

## 13. Known packaging limitations

- **Python version:** the build must run on the same major.minor
  Python (3.11) you intend to support; the executable is not
  cross-version portable.
- **Antivirus:** PyInstaller onedir builds are occasionally flagged by
  over-eager heuristics; signing the binary removes most false
  positives.
- **Updates:** replacing `dist\Jarvis` while it runs can fail on Windows
  (locked files); close Jarvis first.
- **Startup registration** writes the path of the running executable —
  move the distribution and re-run `enable startup` afterwards.
- The distribution does not include tests, logs, or developer tools.

---

## Project layout

```
main.py            entry point (CLI flags + interactive loop)
core/              pipeline: input → tokenize → spell → intent →
                   parse → validate → execute → respond
utils/             logger, path resolution, NLTK import shim
config/            settings.json, commands.json (templates)
commands/          Windows command compatibility shims
tests/             1000+ unit/integration/performance tests
build.py           repeatable Windows build (test → build → verify)
jarvis.spec        PyInstaller specification
```

## Tests

```bat
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

1440 tests cover the command pipeline, task planning, automation workflows,
security boundaries, startup manager, and packaging regressions. No test launches real applications
or touches the real user startup entry.

## License

No license has been selected for this project. The licensing status
is currently undefined. Do not assume any license grants or
restrictions.
