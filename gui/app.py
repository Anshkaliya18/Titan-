"""Tkinter desktop command center for TITAN."""
from __future__ import annotations

import queue
import re
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from core.config_manager import get_value, set_value
from gui.controller import GUIController

BG = "#0b1018"
PANEL = "#111a26"
PANEL_ALT = "#172333"
TEXT = "#e6edf5"
MUTED = "#91a1b4"
ACCENT = "#4dd6c8"
ERROR = "#ff7d87"
_SECRET_TEXT = re.compile(
    r"(?i)\b(?:GEMINI_API_KEY|OPENAI_API_KEY|API_KEY|ACCESS_TOKEN|AUTH_TOKEN|PASSWORD|SECRET)\s*[:=]\s*[^\s,;]+|"
    r"\bAIza[0-9A-Za-z_-]{20,}\b|\bsk-[0-9A-Za-z_-]{20,}\b|"
    r"\bgh[pousr]_[0-9A-Za-z]{20,}\b|\bgithub_pat_[0-9A-Za-z_]{20,}\b"
)


def _redact_secrets(text: str) -> str:
    """Mask credential-shaped strings before adding text to GUI history."""
    return _SECRET_TEXT.sub("[redacted]", text)


class TitanApp:
    def __init__(self, root: tk.Tk, controller: GUIController | None = None) -> None:
        self.root = root
        self.controller = controller or GUIController()
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.pending_confirmation = False
        self.closing = False
        self.tray_icon = None
        self.wake_thread = None
        self.wake_stop = threading.Event()
        self.root.title("TITAN — AI Command Center")
        self.root.geometry("980x700")
        self.root.minsize(720, 520)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.hide_or_close)
        self._style()
        self._build()
        self._bind_shortcuts()
        self._refresh_status()
        self.root.after(80, self._drain_events)

    def _style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Brand.TLabel", font=("Segoe UI Semibold", 22), foreground=TEXT)
        style.configure("Accent.TButton", background=ACCENT, foreground="#081318", padding=(16, 10), font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton", background=[("disabled", "#42575a"), ("active", "#70e8dc")])
        style.configure("Tool.TButton", background=PANEL_ALT, foreground=TEXT, padding=(10, 8))

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=24)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 18))
        brand = ttk.Frame(header)
        brand.pack(side="left")
        ttk.Label(brand, text="TITAN", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(brand, text="AI COMMAND CENTER", style="Muted.TLabel").pack(anchor="w")
        self.status = tk.Label(header, text="● READY", fg=ACCENT, bg=BG, font=("Segoe UI Semibold", 10))
        self.status.pack(side="right", anchor="center")

        orb = tk.Frame(outer, bg=PANEL, highlightbackground="#233345", highlightthickness=1)
        orb.pack(fill="x", pady=(0, 16))
        tk.Label(orb, text="◉", bg=PANEL, fg=ACCENT, font=("Segoe UI Light", 42)).pack(pady=(10, 0))
        tk.Label(orb, text="READY FOR COMMAND", bg=PANEL, fg=MUTED, font=("Segoe UI Semibold", 9)).pack(pady=(0, 12))

        chat_panel = tk.Frame(outer, bg=PANEL, highlightbackground="#233345", highlightthickness=1)
        chat_panel.pack(fill="both", expand=True)
        self.chat = tk.Text(chat_panel, wrap="word", bg=PANEL, fg=TEXT, insertbackground=TEXT,
                            relief="flat", padx=18, pady=14, font=("Segoe UI", 10),
                            state="disabled", spacing1=3, spacing3=9)
        scroll = ttk.Scrollbar(chat_panel, command=self.chat.yview)
        self.chat.configure(yscrollcommand=scroll.set)
        self.chat.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.chat.tag_configure("speaker", foreground=ACCENT, font=("Segoe UI Semibold", 9))
        self.chat.tag_configure("user", foreground="#b7c6d8")
        self.chat.tag_configure("assistant", foreground=TEXT)
        self.chat.tag_configure("error", foreground=ERROR)
        self._append("TITAN", "TITAN is ready. Enter a command to get started.")

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(14, 10))
        self.voice_button = ttk.Button(controls, text="🎙  Voice", style="Tool.TButton", command=self.listen)
        self.voice_button.pack(side="left", padx=(0, 8))
        self.entry = tk.Entry(controls, bg=PANEL_ALT, fg=TEXT, insertbackground=TEXT,
                              relief="flat", font=("Segoe UI", 11), highlightthickness=1,
                              highlightbackground="#34465b", highlightcolor=ACCENT)
        self.entry.pack(side="left", fill="x", expand=True, ipady=11, padx=(0, 8))
        self.entry.insert(0, "Command TITAN...")
        self.entry.bind("<FocusIn>", self._clear_placeholder)
        self.entry.bind("<Return>", lambda _event: self.submit())
        self.entry.bind("<Escape>", lambda _event: self.entry.delete(0, "end"))
        self.send_button = ttk.Button(controls, text="Send  ➤", style="Accent.TButton", command=self.submit)
        self.send_button.pack(side="right")

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(4, 0))
        self.voice_var = tk.BooleanVar(value=get_value("voice", "enabled", False))
        self.tts_var = tk.BooleanVar(value=get_value("tts", "enabled", False))
        self.wake_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(footer, text="Voice input", variable=self.voice_var,
                        command=lambda: self._save_toggle("voice", self.voice_var)).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(footer, text="TTS", variable=self.tts_var,
                        command=lambda: self._save_toggle("tts", self.tts_var)).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(footer, text="Wake word", variable=self.wake_var,
                        command=self.toggle_wake).pack(side="left", padx=(0, 14))
        ttk.Button(footer, text="Settings  ⚙", style="Tool.TButton", command=self.open_settings).pack(side="right")
        self.entry.focus_set()

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-l>", lambda _e: self.entry.focus_set())
        self.root.bind("<Control-comma>", lambda _e: self.open_settings())
        self.root.bind("<Control-q>", lambda _e: self.exit())

    def _clear_placeholder(self, _event=None) -> None:
        if self.entry.get() == "Command TITAN...":
            self.entry.delete(0, "end")

    def _append(self, speaker: str, text: str, tag: str = "assistant") -> None:
        text = _redact_secrets(text)
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{speaker.upper()}\n", "speaker")
        self.chat.insert("end", f"{text}\n\n", tag)
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def submit(self) -> None:
        text = self.entry.get().strip()
        if not text or text == "Command TITAN..." or self.busy:
            return
        self.entry.delete(0, "end")
        self._dispatch(text, _redact_secrets(text))

    def _dispatch(self, text: str, display_text: str) -> None:
        self._append("You", display_text, "user")
        self.busy = True
        self._set_state("PROCESSING")
        try:
            self.controller.submit(text, lambda response: self.events.put(("response", response)),
                                   lambda error, pending: self.events.put(("done", error, pending)))
        except RuntimeError:
            self.busy = False

    def listen(self) -> None:
        if self.busy:
            return
        if not self.voice_var.get():
            self._append("TITAN", "Voice input is turned off. Enable it in the status bar first.")
            return
        self.busy = True
        self._set_state("LISTENING")
        from core.voice_input import VoiceInput
        try:
            self.controller.submit_voice(VoiceInput(),
                lambda text: self.events.put(("transcript", text)),
                lambda error: self.events.put(("voice_error", error)))
        except RuntimeError:
            self.busy = False

    def _drain_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "response":
                self._append("TITAN", event[1])
            elif kind == "done":
                self.busy = False
                error, self.pending_confirmation = event[1], event[2]
                self._set_state("WAITING FOR CONFIRMATION" if self.pending_confirmation else ("ERROR" if error else "READY"))
                if error:
                    self._append("TITAN", "I couldn't complete that request. Please try again.", "error")
            elif kind == "transcript":
                self.busy = False
                self._set_state("READY")
                transcript = event[1].strip()
                if transcript and messagebox.askyesno("TITAN voice input", f"Use this recognized command?\n\n{_redact_secrets(transcript)}", parent=self.root):
                    self._dispatch(transcript, _redact_secrets(transcript))
            elif kind == "voice_error":
                self.busy = False
                self._set_state("READY")
                self._append("TITAN", self._voice_error(event[1]))
            elif kind == "wake_detected":
                self.wake_var.set(False)
                self._append("TITAN", "Wake phrase detected. Listening for one command.")
                self.listen()
            elif kind == "wake_error":
                self.wake_var.set(False)
                self._append("TITAN", "Wake-word listening stopped because the microphone or local detector became unavailable.")
        if not self.closing:
            self.root.after(80, self._drain_events)

    @staticmethod
    def _voice_error(code: str) -> str:
        messages = {"DISABLED": "Voice input is turned off in settings.", "DEPENDENCY_MISSING": "Windows speech support is not installed.",
                    "NO_SPEECH": "I didn't hear a command.", "TIMEOUT": "Listening timed out.",
                    "PERMISSION_DENIED": "Microphone permission was denied.", "MICROPHONE_UNAVAILABLE": "I couldn't access the microphone."}
        return messages.get(code, "Windows speech recognition is unavailable.")

    def _set_state(self, value: str) -> None:
        colors = {"READY": ACCENT, "PROCESSING": "#91b8ff", "LISTENING": "#ffcf70",
                  "WAITING FOR CONFIRMATION": "#ffcf70", "ERROR": ERROR, "OFFLINE": MUTED}
        self.status.configure(text=f"● {value}", fg=colors.get(value, MUTED))

    def _refresh_status(self) -> None:
        if not self.busy:
            self._set_state("READY")

    @staticmethod
    def _save_toggle(section: str, variable: tk.BooleanVar) -> None:
        set_value(section, "enabled", bool(variable.get()))

    def toggle_wake(self) -> None:
        from core.input import _wake_detector
        if self.wake_var.get():
            result = _wake_detector.start()
            if not result.success:
                self.wake_var.set(False)
                self._append("TITAN", "Wake word could not start. Check voice settings and local model setup.")
                return
            self.wake_stop.clear()
            self.wake_thread = threading.Thread(target=self._wake_loop, daemon=True, name="titan-wake")
            self.wake_thread.start()
        else:
            self.wake_stop.set()
            _wake_detector.stop()

    def _wake_loop(self) -> None:
        from core.input import _wake_detector
        while not self.wake_stop.is_set():
            result = _wake_detector.wait_for_wake()
            if result.success:
                _wake_detector.stop()
                self.events.put(("wake_detected",))
                return
            elif result.error not in {"STOPPED", "NOT_RUNNING"}:
                _wake_detector.stop()
                self.events.put(("wake_error",))
                break

    def open_settings(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("TITAN Settings")
        window.configure(bg=BG)
        window.resizable(False, False)
        ttk.Label(window, text="TITAN SETTINGS", style="Brand.TLabel").pack(anchor="w", padx=20, pady=(18, 8))
        llm_var = tk.BooleanVar(value=get_value("llm", "enabled", False))
        planning_var = tk.BooleanVar(value=get_value("task_planning", "enabled", False))
        ttk.Checkbutton(window, text="Enable Gemini interpretation", variable=llm_var).pack(anchor="w", padx=20, pady=5)
        ttk.Checkbutton(window, text="Enable task planning", variable=planning_var).pack(anchor="w", padx=20, pady=5)
        ttk.Label(window, text="API keys are read from the environment or local .env file; TITAN never displays them.",
                  style="Muted.TLabel", wraplength=430).pack(anchor="w", padx=20, pady=(8, 18))
        actions = ttk.Frame(window)
        actions.pack(fill="x", padx=20, pady=(0, 18))
        ttk.Button(actions, text="Cancel", command=window.destroy).pack(side="right")
        def save_settings() -> None:
            set_value("llm", "enabled", bool(llm_var.get()))
            set_value("task_planning", "enabled", bool(planning_var.get()))
            window.destroy()
        ttk.Button(actions, text="Save", style="Accent.TButton", command=save_settings).pack(side="right", padx=(0, 8))

    def hide_or_close(self) -> None:
        try:
            import pystray
            from PIL import Image, ImageDraw
            image = Image.new("RGB", (64, 64), BG)
            draw = ImageDraw.Draw(image)
            draw.ellipse((8, 8, 56, 56), fill=ACCENT)
            draw.text((19, 20), "T", fill=BG)
            menu = pystray.Menu(pystray.MenuItem("Show TITAN", lambda _icon, _item: self.root.after(0, self.show)),
                                pystray.MenuItem("Voice input", lambda _icon, _item: self.root.after(0, self._tray_voice)),
                                pystray.MenuItem("TTS", lambda _icon, _item: self.root.after(0, self._tray_tts)),
                                pystray.MenuItem("Wake Word", lambda _icon, _item: self.root.after(0, self._tray_wake)),
                                pystray.MenuItem("Settings", lambda _icon, _item: self.root.after(0, self.open_settings)),
                                pystray.MenuItem("Exit", lambda _icon, _item: self.root.after(0, self.exit)))
            self.tray_icon = pystray.Icon("titan", image, "TITAN", menu)
            threading.Thread(target=self.tray_icon.run, daemon=True, name="titan-tray").start()
            self.root.withdraw()
        except ImportError:
            self.exit()

    def _tray_voice(self) -> None:
        self.voice_var.set(not self.voice_var.get())
        self._save_toggle("voice", self.voice_var)

    def _tray_tts(self) -> None:
        self.tts_var.set(not self.tts_var.get())
        self._save_toggle("tts", self.tts_var)

    def _tray_wake(self) -> None:
        self.wake_var.set(not self.wake_var.get())
        self.toggle_wake()

    def show(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.entry.focus_set()

    def exit(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.wake_stop.set()
        try:
            from core.input import _wake_detector
            _wake_detector.stop()
        except Exception:
            pass
        try:
            import main
            main._tts_manager.stop()
        except Exception:
            pass
        self.controller.close()
        if self.tray_icon is not None:
            try:
                self.tray_icon.stop()
            except Exception:
                pass
        self.root.destroy()


def run_gui() -> None:
    root = tk.Tk()
    TitanApp(root)
    root.mainloop()
