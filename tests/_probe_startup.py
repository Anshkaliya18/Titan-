"""Quick manual startup-manager probe (not part of the test suite)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import startup_manager as sm

cmd = sm._resolve_install_command()
print("command:", cmd)
target = sm._target_path(cmd)
print("target:", target, target.is_file())
print("stale:", sm._target_path('"C:\\gone\\x.exe"'),
      sm._target_path('"C:\\gone\\x.exe"').is_file())
print("empty:", sm._target_path(""), sm._target_path('""'))
print("space:", sm._target_path(""))

st = sm.startup_status()
print("status:", st["success"], "|", st["message"])
print("data:", st["data"])
print("enabled:", sm.is_startup_enabled())
print("run key:", sm.RUN_KEY_DISPLAY)

# Non-Windows simulation (registry unavailable) — must fail safe.
sm.winreg = None
print("no-win status:", sm.startup_status())
print("no-win enable:", sm.enable_startup())
print("no-win disable:", sm.disable_startup())
print("no-win is_enabled:", sm.is_startup_enabled())
