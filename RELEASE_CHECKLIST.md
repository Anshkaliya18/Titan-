# Jarvis 1.0.0 — Release Checklist

Repeatable checklist for cutting a Jarvis release. Every box must be
checked before the release is published.

## Code
- [ ] Source reviewed (no debug prints, no dead code, no placeholders)
- [ ] No TODO/FIXME/XXX markers in source
- [ ] No hardcoded developer paths or usernames
- [ ] No secrets, credentials, tokens, or API keys in source
- [ ] Version updated in `core/version.py`

## Tests
- [ ] `python -m pytest tests/ -q` passes
- [ ] `python -m pytest tests/ -v` passes
- [ ] Security tests pass (confirmation, path, URL, allowlist)
- [ ] Integration tests pass (pipeline, multi-command, history)
- [ ] Regression tests pass (natural input, misspellings)
- [ ] Release matrix test passes (every registered command)

## Packaging
- [ ] `python build.py` succeeds
- [ ] `dist/Jarvis/Jarvis.exe` exists and launches
- [ ] Bundled resources present (config, spell model, NLTK data)
- [ ] `--version` prints the correct version
- [ ] Works from a different working directory
- [ ] No logs, caches, or test artifacts in `dist/`

## Security
- [ ] No `shell=True` anywhere
- [ ] No `os.system()` anywhere
- [ ] No `eval()`/`exec()` for command execution
- [ ] Application allowlist enforced
- [ ] Filesystem path containment enforced
- [ ] Dangerous URL schemes rejected (file, javascript, data, ...)
- [ ] Shutdown confirmation enforced
- [ ] Restart confirmation enforced
- [ ] Delete confirmation enforced
- [ ] Confirmation cannot be bypassed (exact command only)
- [ ] History cannot execute or replay commands
- [ ] Malformed config cannot disable safety controls
- [ ] Startup registration cannot register arbitrary executables
- [ ] Secrets are not packaged
- [ ] Logs do not expose secrets

## Documentation
- [ ] README complete (features, install, run, commands, safety,
      config, logging, startup, development, packaging, version)
- [ ] CHANGELOG created
- [ ] This checklist created
- [ ] License status reported (none selected)

## Release
- [ ] Release directory created and clean
- [ ] Executable checksum calculated
- [ ] Artifacts verified (launch, resources, commands)
- [ ] No developer-specific paths in artifacts
