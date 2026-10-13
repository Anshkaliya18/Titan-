# Packaged Build Validation (Step 23 — Phase 27)

Repeatable checklist for validating a packaged `Jarvis.exe` build.
Run every step and record the result. Do NOT mark a step passed if it
was not actually executed.

Build first (from the project root):

```bat
python build.py
```

The script already verifies the executable exists, the bundled
resources are present, and `--version` works. The steps below go
further: they exercise the real interactive behavior.

---

## A. Startup

- [ ] A1. `dist\Jarvis\Jarvis.exe --version` prints `Jarvis 0.1.0`
       and exits (no interactive loop).
- [ ] A2. `dist\Jarvis\Jarvis.exe --help` prints usage and exits.
- [ ] A3. Launch `Jarvis.exe` from the project directory — the
       `Jarvis >` prompt appears within a few seconds.
- [ ] A4. Launch it from a different directory (e.g. `C:\` or the
       desktop) — startup still works; no "file not found" errors.
- [ ] A5. Launch it via a double-click in Explorer — a console window
       opens and shows the prompt (console mode, not windowed).

## B. Commands (safe ones only)

Run each at the prompt:

- [ ] B1. `help` — full help text appears, including the startup
       commands under System.
- [ ] B2. `commands` — command list appears.
- [ ] B3. `startup status` — reports the current state (initially
       "does not start with Windows").
- [ ] B4. `history` — empty or lists this session's entries.
- [ ] B5. `open chrome` (or another allowlisted app) — resolves and
       launches; response is correct. (Real launch — only on a
       machine where that is acceptable.)
- [ ] B6. `search web python tutorials` — opens the default browser
       with a search URL. (Real browser launch.)
- [ ] B7. `lock computer` — parks as a confirmation; answer `no` and
       confirm nothing executed. (Do NOT answer `yes` on a shared
       machine.)
- [ ] B8. `shutdown computer` — parks as a confirmation; answer `no`.
       Confirm the prompt appears and nothing shuts down.

## C. Configuration

- [ ] C1. After first launch, `%APPDATA%\Jarvis\config\settings.json`
       exists (packaged mode).
- [ ] C2. Edit `general.debug` to `true`, relaunch — debug lines
       appear in the console; set it back to `false` afterwards.
- [ ] C3. Delete the settings file, relaunch — it is recreated from
       the bundled defaults without crashing.

## D. Logging

- [ ] D1. After running commands, `%APPDATA%\Jarvis\logs\jarvis.log`
       exists and contains the session start/stop records.
- [ ] D2. The log contains no secrets (passwords typed as arguments
       are redacted — try `open file secret_password.txt` and check
       the log shows `[REDACTED]`).
- [ ] D3. No log files were written inside `dist\Jarvis\_internal\`.

## E. NLTK resources (offline behavior)

- [ ] E1. Disconnect the network (or rename the user `nltk_data`
       folder temporarily) and relaunch — tokenization still works
       because `punkt_tab` is bundled.
- [ ] E2. `opn chrokme` (a typo) is corrected to `open chrome` — the
       bundled spell model works offline.

## F. Working-directory independence

- [ ] F1. From `C:\`, run the full path to `Jarvis.exe` — startup,
       help, and history all work.
- [ ] F2. Create a shortcut to `Jarvis.exe` on the desktop, launch
       from it — works, and the console stays visible.

## G. Startup registration (optional feature)

- [ ] G1. `enable startup` → confirm `yes` → `startup status` reports
       enabled, and `regedit` shows the `Jarvis` value under
       `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run`.
- [ ] G2. `enable startup` again → still exactly one entry (no
       duplicates).
- [ ] G3. `disable startup` → confirm → `startup status` reports
       disabled and the registry value is gone.
- [ ] G4. `disable startup` again → safe no-op, no error.
- [ ] G5. The registered command points at this build's own executable
       (never an arbitrary path).

## H. Source-mode parity

- [ ] H1. `python main.py` from the project directory still works.
- [ ] H2. `python main.py --version` prints the same version as the
       packaged build.
- [ ] H3. `python -m pytest tests/ -q` passes (1093 tests).

## Result

- [ ] All applicable steps passed.
- [ ] Any failed step: describe exactly what happened here.

Notes:
- Steps B5–B8 and G1–G5 change real system state (launch apps, write
  the Run key). Run them only on a test machine or a VM, and clean up
  afterwards (disable startup, close launched apps).
- Never run the packaged build with a network-connected production
  profile during validation.
