"""Small bilingual image-to-map UI for the distributed CPU client."""
import json
import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import traceback

from PIL import Image, ImageTk

from streetgeo import PROJECT_SHORT_NAME
from streetgeo.core import EmptyPredictionDomainError
from streetgeo.paths import ROOT
from streetgeo.inference.backend import InferenceEngine, read_picture
from streetgeo.maps.widgets import WorldMap
from streetgeo.ui.common import (ACCENT, BG, FONT, INPUT, MUTED, NAV, PANEL,
                                 SURFACE, TEXT, ScrollFrame, SidebarLayout, apply_dark_theme, button, label)

if __package__:
    from .i18n import LANGUAGES, map_text, read_language, text
else:
    from client_i18n import LANGUAGES, map_text, read_language, text


class ClientApp:
    default_sidebar_width = 336

    def __init__(self, root, *, settings_path=None, engine=None, checkpoint=None, language=None, device="cpu"):
        self.root = root
        self.prediction_device = device
        self.settings_path = Path(settings_path) if settings_path else ROOT / ".runtime/settings.json"
        self.language = language if language in LANGUAGES else read_language(self.settings_path)
        self.language_choice = tk.StringVar(value=LANGUAGES[self.language])
        preferences = {}
        try:
            preferences = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if not isinstance(preferences, dict):
                preferences = {}
            saved_checkpoint = preferences.get("checkpoint", "")
            if not isinstance(saved_checkpoint, str):
                saved_checkpoint = ""
        except (OSError, ValueError):
            saved_checkpoint = ""
        default_checkpoint = str(ROOT / "best.pt") if (ROOT / "best.pt").is_file() else ""
        self.checkpoint = tk.StringVar(value=str(checkpoint) if checkpoint is not None else saved_checkpoint or default_checkpoint)
        self.google_coverage_only = tk.BooleanVar(value=preferences.get("google_coverage_only") is True)
        self._sidebar_width = preferences.get("sidebar_width")
        self.engine = engine if engine is not None else InferenceEngine()
        self.images, self.candidates = [], []
        self.result, self.preview_photo = None, None
        self.busy = self.closing = self.closed = False
        self.explanation_window = None
        self.explaining = False
        self.explanation_stop = threading.Event()
        self.selected = 0
        self.events = queue.Queue()
        self._texts = []
        self.status = tk.StringVar()
        self._status_key, self._status_values = "ready", {}
        self._build()
        self.checkpoint.trace_add("write", self._checkpoint_changed)
        if not self.checkpoint.get().strip():
            self._set_status("model_required")
        self._refresh_text()
        self._poll_id = root.after(80, self._poll)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def t(self, key, **values):
        return text(key, self.language, **values)

    @property
    def checkpoint_path(self):
        value = self.checkpoint.get().strip()
        if not value:
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path

    def _text(self, widget, key):
        self._texts.append((widget, key))
        return widget

    def _build(self):
        root = self.root
        apply_dark_theme(root)
        root.geometry(f"{min(1300, int(root.winfo_screenwidth() * .9))}x{min(900, int(root.winfo_screenheight() * .86))}")
        root.minsize(1000, 720)
        header = tk.Frame(root, bg=NAV)
        header.pack(fill="x")
        identity = tk.Frame(header, bg=NAV)
        identity.pack(side="left", padx=22, pady=16)
        label(identity, PROJECT_SHORT_NAME, size=20, bold=True, color="white").pack(anchor="w")
        self._text(label(identity, size=9, color=MUTED), "subtitle").pack(anchor="w", pady=(3, 0))
        languages = tk.Frame(header, bg=NAV)
        languages.pack(side="right", padx=22)
        self._text(label(languages, size=9, color=MUTED), "language").pack(side="left", padx=10)
        self.language_box = ttk.Combobox(languages, textvariable=self.language_choice,
                                         values=tuple(LANGUAGES.values()), state="readonly", width=10)
        self.language_box.pack(side="left")
        self.language_box.bind("<<ComboboxSelected>>", self.change_language)
        models = tk.Frame(root, bg=PANEL)
        models.pack(fill="x", padx=18, pady=(12, 0))
        self._text(label(models, size=10, bold=True), "model").pack(side="left", padx=12, pady=10)
        self.model_button = self._text(button(models, "", self.choose_checkpoint, padx=12, pady=7), "choose_model")
        self.model_button.pack(side="right", padx=10, pady=7)
        self.model_entry = ttk.Entry(models, textvariable=self.checkpoint, font=(FONT, 9))
        self.model_entry.pack(side="left", fill="x", expand=True, pady=7)
        self.model_entry.bind("<FocusOut>", lambda _: self._save_settings())
        self.sidebar_layout = SidebarLayout(root, width=self._sidebar_width, default_width=self.default_sidebar_width,
                                            on_resize=lambda _: self._save_settings())
        self.sidebar_layout.pack(fill="both", expand=True, padx=18, pady=18)
        left = self.sidebar_layout.sidebar
        sidebar = ScrollFrame(left)
        sidebar.pack(fill="both", expand=True)
        fields = sidebar.content
        fields.configure(padx=14, pady=16)
        self._text(label(fields, size=13, bold=True), "images").pack(fill="x")
        self.choose_button = self._text(button(fields, "", self.choose_images), "choose")
        self.choose_button.pack(fill="x", pady=(14, 8))
        self._text(label(fields, size=9, color=MUTED, wraplength=282, justify="left"), "image_hint").pack(fill="x")
        self.preview = tk.Label(fields, bg=INPUT, fg=MUTED, height=8, font=(FONT, 10))
        self.preview.pack(fill="x", pady=(14, 8))
        self.image_names = label(fields, size=8, color=MUTED, wraplength=282, justify="left")
        self.image_names.pack(fill="x")
        self.coverage_check = self._text(ttk.Checkbutton(fields, variable=self.google_coverage_only,
                                                        command=self._coverage_changed), "coverage_only")
        self.coverage_check.pack(fill="x", pady=(14, 5))
        self._text(label(fields, size=8, color=MUTED, wraplength=282, justify="left"), "coverage_hint").pack(fill="x")
        self.predict_button = self._text(button(fields, "", self.start_prediction, primary=True), "predict")
        self.predict_button.pack(fill="x", pady=(14, 18))
        self.explain_button = self._text(button(fields, "", self.open_explanation), "explain")
        self.explain_button.pack(fill="x", pady=(0, 18))
        self._text(label(fields, size=12, bold=True), "results").pack(fill="x")
        self._text(label(fields, size=8, color=MUTED, wraplength=282, justify="left"), "results_hint").pack(fill="x", pady=(7, 10))
        self.candidate_list = tk.Listbox(fields, height=5, bg=INPUT, fg=TEXT, relief="flat",
                                        selectbackground="#245b61", selectforeground="white",
                                        highlightthickness=0, font=("Consolas", 10), exportselection=False)
        self.candidate_list.pack(fill="x")
        self.candidate_list.bind("<<ListboxSelect>>", self.select_candidate)
        self._text(label(fields, size=8, color=MUTED, wraplength=282), "local").pack(side="bottom", fill="x", pady=(12, 0))
        right = tk.Frame(self.sidebar_layout.content, bg=BG)
        right.pack(fill="both", expand=True, padx=(10, 0))
        tools = tk.Frame(right, bg=BG)
        tools.pack(fill="x", pady=(0, 10))
        self._text(label(tools, size=14, bold=True), "results").pack(side="left")
        self._text(button(tools, "", lambda: self.map.show_world(), padx=10, pady=6), "world").pack(side="right")
        button(tools, "+", lambda: self.map.change_zoom(1), padx=12, pady=6).pack(side="right", padx=5)
        button(tools, "−", lambda: self.map.change_zoom(-1), padx=12, pady=6).pack(side="right")
        self.map = WorldMap(right, translate=lambda source: map_text(source, self.language),
                            cache_folder=self.settings_path.parent / "map_tiles")
        self.map.pack(fill="both", expand=True)
        footer = tk.Frame(root, bg=BG)
        footer.pack(fill="x", padx=20, pady=(0, 12))
        tk.Label(footer, textvariable=self.status, bg=BG, fg=ACCENT, font=(FONT, 9), anchor="w").pack(fill="x")
        self._set_busy(False)

    def _set_status(self, key, **values):
        self._status_key, self._status_values = key, values
        self.status.set(self.t(key, **values))

    def _refresh_text(self):
        self.root.title(f"{PROJECT_SHORT_NAME} · {self.t('title')}")
        self._texts = [(widget, key) for widget, key in self._texts if widget.winfo_exists()]
        for widget, key in self._texts:
            widget.configure(text=self.t(key))
        self.status.set(self.t(self._status_key, **self._status_values))
        if not self.images:
            self.preview.configure(text=self.t("placeholder"))
        self.image_names.configure(text=(self.t("selected", count=len(self.images)) + "\n" +
                                        "\n".join(Path(p).name for p in self.images)) if self.images else "")
        if not self.candidates:
            self.candidate_list.delete(0, "end")
            self.candidate_list.insert("end", self.t("no_results"))
        self.map.redraw()
        if self.explanation_window is not None and not self.explanation_window.closed:
            self.explanation_window.set_language(self.language)

    def change_language(self, event=None):
        self.language = next(key for key, value in LANGUAGES.items() if value == self.language_choice.get())
        self._refresh_text()
        self._save_settings()

    def _settings(self):
        return {"language": self.language, "checkpoint": self.checkpoint.get().strip(),
                "google_coverage_only": self.google_coverage_only.get(),
                "sidebar_width": self.sidebar_layout.sidebar_width}

    def _save_settings(self):
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.settings_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self._settings(), ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, self.settings_path)
        except OSError:
            pass

    def error(self, key):
        messagebox.showerror(PROJECT_SHORT_NAME, self.t(key), parent=self.root)

    def choose_checkpoint(self):
        if self.busy or self.closing:
            return
        selected = self.checkpoint_path
        path = filedialog.askopenfilename(parent=self.root, title=self.t("choose_model"),
            initialdir=str(selected.parent) if selected and selected.parent.is_dir() else str(ROOT),
            filetypes=[(self.t("model_files"), "*.pt *.pth *.ckpt"), (self.t("all_files"), "*.*")])
        if path:
            self.checkpoint.set(path)
            self._save_settings()

    def _checkpoint_changed(self, *_):
        self.result, self.candidates, self.selected = None, [], 0
        self.map.candidates = []
        self.map.show_world()
        self._set_status("loaded" if self.images and self.checkpoint.get().strip() else
                         "ready" if self.checkpoint.get().strip() else "model_required")
        self._refresh_text()
        self._set_busy(self.busy)

    def _coverage_changed(self):
        self._checkpoint_changed()
        if self.explanation_window is not None and not self.explanation_window.closed:
            self.explanation_window.close()
        self._save_settings()

    def choose_images(self):
        paths = filedialog.askopenfilenames(parent=self.root, title=self.t("choose"),
            filetypes=[(self.t("image_files"), "*.jpg *.jpeg *.png *.webp *.bmp"), (self.t("all_files"), "*.*")])
        if paths:
            self.set_images(paths)

    def set_images(self, paths):
        if self.busy or self.closing:
            return False
        if not 1 <= len(paths) <= 4:
            self.error("too_many")
            return False
        try:
            canvas = Image.new("RGB", (282, 164), SURFACE)
            for index, path in enumerate(paths):
                picture = read_picture(path)
                cell_w, cell_h = (282, 164) if len(paths) == 1 else (141, 82)
                picture.thumbnail((cell_w - 6, cell_h - 6))
                x, y = (0, 0) if len(paths) == 1 else ((index % 2) * 141, (index // 2) * 82)
                canvas.paste(picture, (x + (cell_w - picture.width) // 2, y + (cell_h - picture.height) // 2))
            photo = ImageTk.PhotoImage(canvas, master=self.root)
        except (OSError, ValueError, Image.DecompressionBombError):
            self.error("invalid_image")
            return False
        self.images = [str(Path(p).resolve()) for p in paths]
        self.preview_photo = photo
        self.preview.configure(image=photo, text="", height=164)
        self.result, self.candidates, self.selected = None, [], 0
        self.map.candidates = []
        self.map.show_world()
        self._set_status("loaded" if self.checkpoint.get().strip() else "model_required")
        self._refresh_text()
        self._set_busy(False)
        return True

    def _set_busy(self, busy):
        self.busy = busy
        self.choose_button.configure(state="disabled" if busy or self.closing else "normal")
        self.model_button.configure(state="disabled" if busy or self.closing else "normal")
        self.model_entry.configure(state="disabled" if busy or self.closing else "normal")
        self.coverage_check.configure(state="disabled" if busy or self.closing else "normal")
        self.predict_button.configure(state="disabled" if busy or self.closing or not self.images or not self.checkpoint.get().strip() else "normal")
        self.explain_button.configure(state=self.predict_button.cget("state"))
        if self.explanation_window is not None and not self.explanation_window.closed:
            self.explanation_window.set_busy(busy or self.closing)
            self.explanation_window.cancel_button.configure(state="normal" if self.explaining and busy else "disabled")

    def open_explanation(self):
        if self.busy or self.closing or not self.images:
            return
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            self.error("missing_model")
            return
        from streetgeo.ui.explanation import ExplanationWindow
        if self.explanation_window is None or self.explanation_window.closed:
            self.explanation_window = ExplanationWindow(self.root, request=self.start_explanation_job,
                                                        cancel=self.cancel_explanation, language=self.language)
        self.explanation_window.lift()
        self.start_explanation_job(self.explanation_window, "attention", None)

    def cancel_explanation(self):
        if not self.explaining:
            return
        self.explanation_stop.set()
        self._set_status("explanation_cancelling")
        if self.explanation_window is not None and not self.explanation_window.closed:
            self.explanation_window.set_status("正在取消 · 等待当前模型加载或预测结束…")
            self.explanation_window.cancel_button.configure(state="disabled")

    def _log_failure(self):
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            with (self.settings_path.parent / "prediction.log").open("a", encoding="utf-8") as log:
                log.write(traceback.format_exc() + "\n")
        except OSError:
            pass

    def start_explanation_job(self, window, action, payload):
        if self.busy or self.closing or window.closed:
            return
        if not self.images or self.checkpoint_path is None:
            window.show_error(self.t("model_required"))
            return
        analysis, occlusion = window.analysis, window.occlusion
        if action != "attention" and analysis is None:
            return
        checkpoint, images, language = self.checkpoint_path, list(self.images), self.language
        coverage_only = self.google_coverage_only.get()
        self.explanation_stop.clear()
        self.explaining = True
        self._set_busy(True)
        self._set_status("explaining")
        window.set_status("正在准备分析…")

        def worker():
            try:
                from streetgeo.inference.explain import run_explanation_job
                data = run_explanation_job(self.engine, action, payload, checkpoint=checkpoint,
                    images=images, device=self.prediction_device, analysis=analysis, occlusion=occlusion, language=language,
                    google_coverage_only=coverage_only,
                    cancelled=self.explanation_stop.is_set,
                    progress=lambda source: self.events.put(("explanation_progress", (window, source))))
                self.events.put(("explanation_result", (window, action, data)))
            except Exception:
                cancelled = self.explanation_stop.is_set()
                if not cancelled:
                    self._log_failure()
                self.events.put(("explanation_cancelled" if cancelled else "explanation_error", (window, action, None)))
        threading.Thread(target=worker, daemon=True, name="svgt-client-explanation").start()

    def start_prediction(self):
        if self.busy or self.closing:
            return
        if not self.images:
            self._set_status("ready")
            return
        checkpoint = self.checkpoint_path
        if checkpoint is None or not checkpoint.is_file():
            self.error("missing_model")
            return
        self._set_busy(True)
        self._set_status("preparing")
        images = list(self.images)
        coverage_only = self.google_coverage_only.get()
        def worker():
            try:
                result = self.engine.predict(checkpoint, images, device=self.prediction_device, top_k=5,
                    google_coverage_only=coverage_only,
                    progress=lambda value: self.events.put(("progress", value)))
                self.events.put(("result", result))
            except Exception as error:
                self._log_failure()
                self.events.put(("error", "coverage_empty" if isinstance(error, EmptyPredictionDomainError) else "failed"))
        threading.Thread(target=worker, daemon=True, name="svgt-client-prediction").start()

    def select_candidate(self, event=None):
        indices = self.candidate_list.curselection()
        if indices and indices[0] < len(self.candidates):
            self.selected = indices[0]
            candidate = self.candidates[self.selected]
            self.map.selected = self.selected
            self.map.focus(candidate["latitude"], candidate["longitude"], zoom=4)

    def _poll(self):
        self._poll_id = None
        for _ in range(30):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if self.handle_event(kind, value):
                continue
            if kind == "progress" and not self.closing:
                self._set_status("loading" if "加载" in value else "analyzing")
            elif kind == "explanation_progress":
                window, source = value
                if not window.closed and not self.explanation_stop.is_set():
                    window.set_status(source)
            elif kind in ("explanation_result", "explanation_error", "explanation_cancelled"):
                window, action, data = value
                self.explaining = False
                self._set_busy(False)
                cancelled = self.explanation_stop.is_set() or kind == "explanation_cancelled"
                status = "explanation_cancelled" if cancelled else "explanation_failed" if kind == "explanation_error" else "explanation_done"
                if not self.closing:
                    self._set_status(status)
                if not window.closed and not self.closing:
                    if cancelled:
                        window.set_status("分析已取消 · 已有结果保留")
                    elif kind == "explanation_error":
                        window.show_error(self.t("explanation_failed"))
                    elif action == "attention":
                        window.set_analysis(data)
                        window.set_busy(False)
                    elif action == "occlusion":
                        window.set_occlusion(data)
                        window.set_busy(False)
                    else:
                        window.set_status("已导出：{path}", path=data)
            elif kind == "result":
                self._set_busy(False)
                if self.closing:
                    continue
                self.show_result(value)
            elif kind == "error":
                self._set_busy(False)
                if not self.closing:
                    self._set_status(value or "failed")
                    self.error(value or "failed")
        if self.closing and not self.busy:
            self._destroy()
        else:
            self._poll_id = self.root.after(80, self._poll)

    def handle_event(self, kind, value):
        return False

    def show_result(self, value):
        self.result = value
        self.candidates = list(value["candidates"])
        self.selected = 0
        self.candidate_list.delete(0, "end")
        for index, candidate in enumerate(self.candidates, 1):
            self.candidate_list.insert("end", f"{index}. {candidate['latitude']:.5f}, {candidate['longitude']:.5f}")
        if self.candidates:
            self.candidate_list.selection_set(0)
        self.map.candidates, self.map.selected = self.candidates, 0
        self.map.show_world()
        self._set_status("done_coverage" if value.get("google_coverage_only") else "done",
                         seconds=value["elapsed_seconds"])

    def close(self):
        if self.closed:
            return
        self._save_settings()
        self.closing = True
        self.cancel_explanation()
        self._set_busy(self.busy)
        if self.busy:
            self._set_status("closing")
        else:
            self._destroy()

    def _destroy(self):
        self.closed = True
        if self.explanation_window is not None and not self.explanation_window.closed:
            self.explanation_window.close()
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        self.map.close()
        self.root.destroy()
