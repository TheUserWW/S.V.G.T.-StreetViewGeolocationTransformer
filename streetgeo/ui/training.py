"""Chinese desktop workbench for OpenStreetView-5M import and resumable training."""

import argparse
from collections import deque
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, ttk

from streetgeo.ui.backend import (DEFAULTS, ROOT, ProcessRunner, build_prepare,
                         build_training, load_settings, resolve_path, save_settings)

from streetgeo.ui.common import (BG, PANEL, NAV, TEXT, MUTED, BORDER, ACCENT, FONT,
                                 SURFACE, INPUT, DISABLED, SELECTION, WARNING, CHART_FILL,
                                 apply_dark_theme, label, button, ScrollFrame)


class LossChart(tk.Canvas):
    def __init__(self, parent):
        super().__init__(parent, height=190, bg=PANEL, highlightthickness=0)
        self.points = deque(maxlen=400)
        self.bind("<Configure>", lambda _: self.draw())

    def add(self, step, loss):
        if math.isfinite(loss):
            self.points.append((step, loss))
            self.draw()

    def reset(self):
        self.points.clear()
        self.draw()

    def draw(self):
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        left, right, top, bottom = 44, max(70, width - 16), 20, max(60, height - 30)
        for fraction in (0, 0.5, 1):
            y = top + fraction * (bottom - top)
            self.create_line(left, y, right, y, fill=BORDER, dash=(3, 4))
        if not self.points:
            self.create_text(width / 2, height / 2, text="开始训练后显示损失曲线", fill=MUTED, font=(FONT, 9))
            return
        x0, x1 = self.points[0][0], self.points[-1][0]
        low, high = min(point[1] for point in self.points), max(point[1] for point in self.points)
        padding = max((high - low) * 0.12, 0.03)
        low, high = max(0, low - padding), high + padding
        positions = [(left + (step - x0) / max(1, x1 - x0) * (right - left),
                      bottom - (loss - low) / max(1e-8, high - low) * (bottom - top))
                     for step, loss in self.points]
        if len(positions) > 1:
            polygon = [positions[0][0], bottom, *[value for point in positions for value in point], positions[-1][0], bottom]
            self.create_polygon(polygon, fill=CHART_FILL, outline="")
            self.create_line(*[value for point in positions for value in point], fill=ACCENT, width=2)
        x, y = positions[-1]
        self.create_oval(x - 3, y - 3, x + 3, y + 3, fill=ACCENT, outline=PANEL, width=1)
        for y, value in ((top, high), (bottom, low)):
            self.create_text(left - 7, y, text=f"{value:.2f}", anchor="e", fill=MUTED, font=(FONT, 8))
        self.create_text(left, height - 12, text=f"step {x0}", anchor="w", fill=MUTED, font=(FONT, 8))
        self.create_text(right, height - 12, text=str(x1), anchor="e", fill=MUTED, font=(FONT, 8))


class TrainingApp:
    def __init__(self, root, *, settings_path=None, probe=True):
        self.root = root
        self.settings_path = Path(settings_path) if settings_path else ROOT / ".gui" / "settings.json"
        self.values = {key: tk.StringVar(value=value) for key, value in load_settings(self.settings_path).items()}
        self.events = queue.Queue()
        self.runner = ProcessRunner(self.events)
        self.probe_runner = ProcessRunner(self.events)
        self.pages, self.nav_buttons, self.editors, self.browse_buttons = {}, {}, {}, {}
        self.action_buttons = []
        self.active_page, self.active_job = "train", None
        self.total_steps = 0
        self.last_checkpoint = ""
        self.closing = False
        self._build()
        self._show_page("train")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll)
        if probe:
            self.root.after(300, self._probe)

    def _build(self):
        root = self.root
        root.title("StreetGeo · 训练工作台")
        apply_dark_theme(root)
        width = min(1460, int(root.winfo_screenwidth() * 0.9))
        height = min(1020, int(root.winfo_screenheight() * 0.9))
        root.geometry(f"{width}x{height}")
        root.minsize(1080, 780)
        sidebar = tk.Frame(root, bg=NAV, width=196)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        brand = tk.Canvas(sidebar, height=94, bg=NAV, highlightthickness=0)
        brand.pack(fill="x", padx=18, pady=(22, 15))
        brand.create_oval(1, 2, 33, 34, outline="#60d4bd", width=2)
        brand.create_oval(10, 2, 24, 34, outline="#60d4bd", width=1)
        brand.create_line(1, 18, 33, 18, fill="#60d4bd")
        brand.create_text(0, 58, text="STREETGEO", anchor="w", fill="white", font=("Segoe UI", 12, "bold"))
        brand.create_text(0, 80, text="TRAINING STUDIO", anchor="w", fill="#8aa4b8", font=("Segoe UI", 8))
        label(sidebar, "工作流程", color="#7f9bad", size=9).pack(fill="x", padx=22, pady=(10, 12))
        for key, text in (("split", "01   导入与划分"), ("train", "02   模型训练")):
            item = tk.Button(sidebar, text=text, anchor="w", command=lambda key=key: self._show_page(key),
                             font=(FONT, 10), bg=NAV, fg="#bdceda", activebackground="#263e50",
                             activeforeground="white", relief="flat", bd=0, padx=18, pady=14, cursor="hand2")
            item.pack(fill="x", padx=10, pady=3)
            self.nav_buttons[key] = item
        tk.Button(sidebar, text="03   预测地图", anchor="w", command=self._open_prediction_map,
                  font=(FONT, 10), bg=NAV, fg="#bdceda", activebackground="#263e50",
                  activeforeground="white", relief="flat", bd=0, padx=18, pady=14,
                  cursor="hand2").pack(fill="x", padx=10, pady=3)
        footer = tk.Frame(sidebar, bg=NAV)
        footer.pack(side="bottom", fill="x", padx=20, pady=22)
        label(footer, "本地运行 · 自动保存", color="#7f9bad", size=9).pack(anchor="w", pady=(0, 12))
        for text, command in (("GPU 运行检查", self._gpu_check), ("打开项目目录", lambda: self._open(ROOT)),
                              ("OSV-5M 训练指南", lambda: self._open(ROOT / "docs" / "STREETGEO_TRAINING.md"))):
            tk.Button(footer, text=text, command=command, anchor="w", bg=NAV, fg="#bdceda",
                      activebackground=NAV, activeforeground="white", font=(FONT, 9),
                      relief="flat", bd=0, cursor="hand2", pady=6).pack(fill="x")

        main = tk.Frame(root, bg=BG)
        main.pack(side="left", fill="both", expand=True, padx=24, pady=(22, 16))
        header = tk.Frame(main, bg=BG)
        header.pack(fill="x", pady=(0, 18))
        title_box = tk.Frame(header, bg=BG)
        title_box.pack(side="left")
        self.page_title = label(title_box, "模型训练", size=22, bold=True)
        self.page_title.pack(anchor="w")
        self.page_subtitle = label(title_box, "设置训练参数，随时保存进度并继续。", color=MUTED, size=10)
        self.page_subtitle.pack(anchor="w", pady=(5, 0))
        self.hardware = label(header, "正在检测训练设备…", size=9, color=MUTED, justify="right")
        self.hardware.pack(side="right", anchor="e", padx=(12, 0))

        panes = tk.PanedWindow(main, orient="vertical", bg=BG, sashwidth=8, bd=0, sashrelief="flat")
        panes.pack(fill="both", expand=True)
        upper = tk.Frame(panes, bg=BG)
        upper.columnconfigure(0, weight=3, minsize=450)
        upper.columnconfigure(1, weight=2, minsize=300)
        upper.rowconfigure(0, weight=1)
        panes.add(upper, stretch="always", minsize=410)
        self.page_host = tk.Frame(upper, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        self.page_host.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        self.page_host.rowconfigure(0, weight=1)
        self.page_host.columnconfigure(0, weight=1)
        self._build_split()
        self._build_train()
        self._build_monitor(upper)

        logs = tk.Frame(panes, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        panes.add(logs, stretch="never", minsize=120, height=210)
        log_header = tk.Frame(logs, bg=PANEL)
        log_header.pack(fill="x", padx=16, pady=(10, 6))
        label(log_header, "运行日志", size=10, bold=True).pack(side="left")
        tk.Button(log_header, text="打开日志目录", command=lambda: self._open(ROOT / ".gui" / "logs"),
                  bg=PANEL, fg=ACCENT, relief="flat", bd=0, font=(FONT, 9), cursor="hand2").pack(side="right")
        text_frame = tk.Frame(logs, bg=PANEL)
        text_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        self.log = tk.Text(text_frame, bg=INPUT, fg=TEXT, font=("Consolas", 9), height=7,
                           wrap="word", bd=0, padx=10, pady=8, state="disabled",
                           insertbackground=TEXT, selectbackground=SELECTION, selectforeground=TEXT, yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=self.log.yview)
        self.footer_text = label(main, "就绪  ·  配置仅保存在本机 · 仅接受 OpenStreetView-5M", color=MUTED, size=9)
        self.footer_text.pack(fill="x", pady=(10, 0))
        self._append_log("欢迎使用 StreetGeo 训练工作台。仅支持 OpenStreetView-5M，自动跳过尚未下载的图片。默认模型约 514MiB，升级架构需新建训练。")

    def _page(self, key, title, description):
        page = tk.Frame(self.page_host, bg=PANEL)
        page.grid(row=0, column=0, sticky="nsew")
        page.rowconfigure(1, weight=1)
        page.columnconfigure(0, weight=1)
        heading = tk.Frame(page, bg=PANEL)
        heading.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 8))
        label(heading, title, size=12, bold=True).pack(anchor="w")
        label(heading, description, color=MUTED, size=9, wraplength=550).pack(anchor="w", pady=(6, 0))
        scroll = ScrollFrame(page)
        scroll.grid(row=1, column=0, sticky="nsew", padx=(18, 6))
        scroll.content.columnconfigure(0, weight=1)
        scroll.content.columnconfigure(1, weight=1)
        actions = tk.Frame(page, bg=PANEL)
        actions.grid(row=2, column=0, sticky="ew", padx=20, pady=16)
        self.pages[key] = page
        return scroll.content, actions

    def _field(self, parent, text, key, row, col=0, span=1, *, choices=None, browse=None):
        frame = tk.Frame(parent, bg=PANEL)
        frame.grid(row=row, column=col, columnspan=span, sticky="ew", padx=(2, 12), pady=(6, 8))
        frame.columnconfigure(0, weight=1)
        label(frame, text, size=9, color=MUTED).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        variable = self.values[key]
        if choices:
            editor = ttk.Combobox(frame, textvariable=variable, values=choices, state="readonly", width=12)
        else:
            editor = tk.Entry(frame, textvariable=variable, font=(FONT, 10), bg=INPUT, fg=TEXT,
                              insertbackground=TEXT, selectbackground=SELECTION, selectforeground=TEXT,
                              relief="flat", bd=0, width=16,
                              highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT,
                              disabledbackground=SURFACE, disabledforeground=DISABLED)
        editor.grid(row=1, column=0, sticky="ew", ipady=6 if not choices else 0)
        self.editors[key] = editor
        if browse:
            chooser = button(frame, "选择", lambda: self._browse(key, browse), padx=10, pady=6)
            chooser.grid(row=1, column=1, padx=(8, 0))
            self.browse_buttons[key] = chooser
        return editor

    def _action(self, parent, text, command, primary=False):
        item = button(parent, text, command, primary=primary)
        item.pack(side="left", padx=(0, 10))
        self.action_buttons.append(item)
        return item

    def _build_split(self):
        fields, actions = self._page("split", "OpenStreetView-5M 导入与划分", "唯一训练数据来源为 OSV-5M；支持只下载部分训练分片。")
        label(fields, "数据来源：OpenStreetView-5M（固定）", color=ACCENT, size=10).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=2, pady=(8, 12))
        self._field(fields, "OSV-5M 根目录 · 含 train.csv 或 train/train.csv", "osv_root", 1, span=2, browse="directory")
        self._field(fields, "划分结果保存目录", "split_output", 2, span=2, browse="directory")
        self._field(fields, "地理分组大小 / 度", "block_degrees", 3)
        self._field(fields, "不同集合间的隔离距离 / 米", "guard_m", 3, 1)
        description = label(fields, "只导入本地已解压的 OSV-5M 训练图片，未下载记录自动跳过并统计。\n按采集序列和地理块整体分组，训练、验证、测试之间隔离近邻。\n官方 test 目录不参与训练；补充图片后使用新的划分输出目录。",
                            size=9, color=MUTED, wraplength=500, justify="left")
        description.grid(row=4, column=0, columnspan=2, sticky="w", padx=2, pady=14)
        self.split_summary = label(fields, "尚未生成数据划分", size=10, color=ACCENT, wraplength=500)
        self.split_summary.grid(row=5, column=0, columnspan=2, sticky="w", padx=2, pady=8)
        self._action(actions, "导入并生成划分", self._prepare, primary=True)
        self._action(actions, "填入训练配置", self._use_splits)

    def _build_train(self):
        fields, actions = self._page("train", "OSV-5M 全球街景训练", "新建默认约 514MiB / 1.35 亿参数；续训沿用权重架构。单视角 · 224×224 · BF16。")
        editor = self._field(fields, "运行方式", "mode", 0, choices=("新建训练", "继续上次训练"))
        editor.bind("<<ComboboxSelected>>", lambda _: self._mode_changed())
        self._field(fields, "训练设备", "device", 0, 1, choices=("cuda", "auto", "cpu"))
        self._field(fields, "恢复进度 / *.last.pt", "resume", 1, span=2, browse="checkpoint")
        self._field(fields, "OSV-5M 训练清单", "train_manifest", 2, span=2, browse="manifest")
        self._field(fields, "OSV-5M 验证清单（可留空）", "val_manifest", 3, span=2, browse="manifest")
        self._field(fields, "最佳模型输出文件", "output", 4, span=2, browse="save_checkpoint")
        self._field(fields, "训练轮数 / epochs", "epochs", 5)
        self._field(fields, "批次大小 / batch size", "batch_size", 5, 1)
        self._field(fields, "训练精度", "amp", 6, choices=("bf16", "fp16", "off", "auto"))
        self._field(fields, "数据读取进程数", "num_workers", 6, 1)
        self._field(fields, "每多少批次保存一次", "save_every_steps", 7)
        self._field(fields, "最长保存间隔 / 秒（0 关闭）", "save_every_seconds", 7, 1)
        self._field(fields, "初始学习率", "learning_rate", 8)
        self._field(fields, "预热步数", "warmup_steps", 8, 1)
        self._field(fields, "自定义模型配置（可选 JSON）", "model_config", 9, span=2, browse="json")
        self._field(fields, "新阶段初始权重（可选）", "init_from", 10, span=2, browse="checkpoint")
        self.train_start = self._action(actions, "开始训练", self._train, primary=True)
        self._action(actions, "读取续训信息", self._inspect_checkpoint)
        self._mode_changed()

    def _build_monitor(self, parent):
        monitor = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        monitor.grid(row=0, column=1, sticky="nsew")
        inner = tk.Frame(monitor, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=20, pady=18)
        top = tk.Frame(inner, bg=PANEL)
        top.pack(fill="x")
        label(top, "任务监控", size=12, bold=True).pack(side="left")
        self.status_badge = label(top, "●  就绪", size=9, color=ACCENT)
        self.status_badge.pack(side="right")
        self.task_label = label(inner, "等待开始", size=11, bold=True)
        self.task_label.pack(fill="x", pady=(20, 6))
        self.detail_label = label(inner, "先准备街景数据，再启动训练。", size=9, color=MUTED, wraplength=350)
        self.detail_label.pack(fill="x", pady=(0, 14))
        self.progress = ttk.Progressbar(inner, mode="determinate", maximum=100)
        self.progress.pack(fill="x", ipady=2)
        metrics = tk.Frame(inner, bg=PANEL)
        metrics.pack(fill="x", pady=(18, 14))
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(1, weight=1)
        self.metric_labels = {}
        for number, (key, title) in enumerate((("step", "已完成批次"), ("epoch", "当前轮次"),
                                              ("loss", "训练损失"), ("distance", "验证中位误差 / km"))):
            cell = tk.Frame(metrics, bg=SURFACE, padx=12, pady=10)
            cell.grid(row=number // 2, column=number % 2, sticky="ew", padx=(0, 8) if number % 2 == 0 else (0, 0), pady=4)
            label(cell, title, size=8, color=MUTED).pack(anchor="w")
            value = label(cell, "—", size=18, bold=True)
            value.pack(anchor="w", pady=(4, 0))
            self.metric_labels[key] = value
        # Reserve the stop action before allocating the chart's flexible area.
        self.stop_button = button(inner, "停止并保存", self._stop)
        self.stop_button.pack(side="bottom", fill="x")
        self.stop_button.configure(state="disabled")
        self.checkpoint_label = label(inner, "最近保存：尚无 checkpoint", size=9, color=MUTED, wraplength=350, justify="left")
        self.checkpoint_label.pack(side="bottom", fill="x", pady=(4, 12))
        label(inner, "LOSS / 训练损失", size=8, color=MUTED, bold=True).pack(anchor="w", pady=(5, 0))
        self.chart = LossChart(inner)
        self.chart.pack(fill="both", expand=True, pady=(2, 8))

    def _mode_changed(self):
        resume = self.values["mode"].get() == "继续上次训练"
        if hasattr(self, "train_start"):
            self.train_start.configure(text="继续训练" if resume else "开始训练")
        keys = ("train_manifest", "val_manifest", "output", "epochs", "batch_size", "amp", "learning_rate",
                "warmup_steps", "model_config", "init_from")
        for key in keys:
            editor = self.editors.get(key)
            if editor is not None:
                editor.configure(state="disabled" if resume else ("readonly" if isinstance(editor, ttk.Combobox) else "normal"))
            if key in self.browse_buttons:
                self.browse_buttons[key].configure(state="disabled" if resume else "normal")
        if "resume" in self.editors:
            self.editors["resume"].configure(state="normal" if resume else "disabled")
            self.browse_buttons["resume"].configure(state="normal" if resume else "disabled")

    def _show_page(self, key):
        self.active_page = key
        self.pages[key].tkraise()
        titles = {"train": ("模型训练", "设置训练参数，随时保存进度并继续。"),
                  "split": ("数据导入与划分", "仅 OpenStreetView-5M · 全球坐标 · 地点隔离。")}
        self.page_title.configure(text=titles[key][0])
        self.page_subtitle.configure(text=titles[key][1])
        for name, item in self.nav_buttons.items():
            item.configure(bg=SELECTION if name == key else NAV, fg="white" if name == key else "#bdceda")

    def _browse(self, key, kind):
        existing = resolve_path(self.values[key].get())
        initial = existing if existing.is_dir() else existing.parent
        options = {"parent": self.root, "initialdir": str(initial if initial.exists() else ROOT)}
        if kind == "directory":
            value = filedialog.askdirectory(**options)
        elif kind == "save_checkpoint":
            value = filedialog.asksaveasfilename(**options, defaultextension=".pt", initialfile=existing.name,
                                                filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("所有文件", "*")])
        else:
            filters = {"manifest": [("JSONL 数据清单", "*.jsonl"), ("所有文件", "*")],
                       "checkpoint": [("PyTorch checkpoint", "*.pt *.pth"), ("所有文件", "*")],
                       "json": [("JSON 配置", "*.json"), ("所有文件", "*")]}
            value = filedialog.askopenfilename(**options, filetypes=filters[kind])
        if value:
            self.values[key].set(value)

    def _settings(self):
        return {key: value.get() for key, value in self.values.items()}

    def _persist(self):
        try:
            save_settings(self.settings_path, self._settings())
        except OSError:
            self._append_log("无法保存界面配置；当前任务仍可正常运行。")

    def _error(self, message):
        if not self.active_job:
            self.status_badge.configure(text="●  请检查配置", fg=WARNING)
            self.detail_label.configure(text=str(message), fg=WARNING)
        self._append_log("[配置] " + str(message))

    def _lock_controls(self, locked):
        for editor in self.editors.values():
            editor.configure(state="disabled" if locked else ("readonly" if isinstance(editor, ttk.Combobox) else "normal"))
        for item in (*self.browse_buttons.values(), *self.action_buttons):
            item.configure(state="disabled" if locked else "normal")
        if not locked:
            self._mode_changed()

    def _start(self, job, module, arguments, *, stop_file=None, extra_env=None):
        if self.runner.busy:
            self._error("已有任务在运行，请先等待完成或停止并保存。")
            return
        try:
            self.runner.start(job, module, arguments, stop_file=stop_file, extra_env=extra_env)
        except (OSError, RuntimeError) as error:
            self._error(str(error))
            return
        self.active_job = job
        self.total_steps = 0
        self._persist()
        self._lock_controls(True)
        self.stop_button.configure(state="normal" if stop_file else "disabled", text="停止并保存")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        names = {"split": "正在导入与划分数据", "train": "训练启动中", "inspect": "正在读取 checkpoint", "gpu": "正在检查 GPU"}
        self.task_label.configure(text=names[job])
        self.status_badge.configure(text="●  运行中", fg=ACCENT)
        self.detail_label.configure(text="后台任务正在运行，日志将在下方显示。", fg=MUTED)

    def _prepare(self):
        try:
            stop = self.runner.new_stop_file()
            self._start("split", "streetgeo.datasets.prepare", build_prepare(self._settings(), stop), stop_file=stop)
        except ValueError as error:
            self._error(error)

    def _use_splits(self):
        output = resolve_path(self.values["split_output"].get())
        if not all((output / name).is_file() for name in ("train.jsonl", "val.jsonl", "test.jsonl")):
            self._error("请先生成完整的训练、验证和测试清单。")
            return
        from streetgeo.datasets.policy import check_osv_manifest_header
        try:
            for name in ("train", "val", "test"):
                check_osv_manifest_header(output / f"{name}.jsonl")
        except (ValueError, OSError) as error:
            self._error(str(error))
            return
        self.values["train_manifest"].set(str(output / "train.jsonl"))
        self.values["val_manifest"].set(str(output / "val.jsonl"))
        self.values["mode"].set("新建训练")
        name = "streetgeo_514m_osv"
        candidate = ROOT / "checkpoints" / name / "best.pt"
        suffix = 2
        while candidate.exists() or candidate.with_name("best.last.pt").exists():
            candidate = ROOT / "checkpoints" / f"{name}_{suffix}" / "best.pt"
            suffix += 1
        self.values["output"].set(str(candidate))
        self.values["resume"].set(str(candidate.with_name("best.last.pt")))
        self.values["init_from"].set("")
        self._mode_changed()
        self._show_page("train")
        self._persist()
        self.detail_label.configure(text="已将划分结果填入训练配置。", fg=ACCENT)

    def _train(self):
        try:
            stop = self.runner.new_stop_file()
            arguments = build_training(self._settings(), stop)
            self.chart.reset()
            for item in self.metric_labels.values():
                item.configure(text="—")
            self._start("train", "streetgeo.training.train", arguments, stop_file=stop)
        except ValueError as error:
            self._error(error)

    def _stop(self):
        if self.runner.request_stop():
            self.stop_button.configure(state="disabled", text="正在等待安全停止…")
            self.status_badge.configure(text="●  正在保存", fg=WARNING)
            self.detail_label.configure(text="当前批次或网络请求结束后保存进度，请稍候。", fg=MUTED)
            self._append_log("[控制] 已请求安全停止，正在等待进度保存。")

    def _probe(self):
        try:
            self.probe_runner.start("hardware", "streetgeo.ui.worker", ["hardware"])
        except OSError:
            self.hardware.configure(text="设备检测不可用，可在训练日志中查看。")

    def _inspect_checkpoint(self):
        try:
            path = self.values["resume"].get()
            if not resolve_path(path).is_file():
                raise ValueError("请先切换为续训，并选择有效的 *.last.pt 文件。")
            self._start("inspect", "streetgeo.ui.worker", ["checkpoint", "--path", str(resolve_path(path))])
        except ValueError as error:
            self._error(error)

    def _gpu_check(self):
        self._start("gpu", "streetgeo.cli.check_gpu", ["--batch-size", "16"])

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        line_count = int(self.log.index("end-1c").split(".")[0])
        if line_count > 2200:
            self.log.delete("1.0", f"{line_count - 2000}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _structured(self, data):
        if data.get("event") == "dataset_import_progress":
            self.progress.stop()
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
            if data["stage"] == "index":
                text = f"OSV-5M：已发现 {data['local_images']:,} 张本地训练图片…"
            else:
                text = (f"OSV-5M：已扫描 {data['metadata_rows']:,} 条坐标，匹配 {data['samples']:,} 张，"
                        f"未下载 {data['missing_images']:,} 条。")
            self.detail_label.configure(text=text, fg=MUTED)
        elif data.get("event") == "dataset_import_complete":
            self.split_summary.configure(text=f"OSV-5M：匹配 {data['samples']:,} 张，跳过未下载 {data['missing_images']:,} 条。")
        if data.get("event") == "dataset_import":
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=100,
                                    value=40 * data["cities_done"] / max(1, data["cities_total"]))
            self.detail_label.configure(text=f"正在导入 {data['city']} · 累计 {data['samples']:,} 张 · "
                                             f"{data['cities_done']} / {data['cities_total']} 城市", fg=MUTED)
        elif data.get("event") == "dataset_splitting":
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
            self.detail_label.configure(text=f"已导入 {data['samples']:,} 张，正在按地点与地理块划分…", fg=MUTED)
        elif data.get("event") == "dataset_ready":
            self.detail_label.configure(text=f"已准备 {data['samples']:,} 张 · " +
                                             " / ".join(f"{key}: {value:,}" for key, value in data["source_counts"].items()), fg=ACCENT)
        event = data.get("event")
        if event == "hardware":
            gpu = data["gpu"].replace("NVIDIA GeForce ", "")
            self.hardware.configure(text=f"{gpu} · {data['vram_gib']:.0f} GB\nPyTorch {data['torch']} · 可用磁盘 {data['disk_free_gib']:.0f} GiB")
            if not data["cuda"] and not self.active_job:
                self.values["device"].set("cpu")
                self.values["amp"].set("off")
            return
        if event == "checkpoint_info":
            if not data["resumable"]:
                self._error("这是推理权重文件。请选用 *.last.pt 或 *.last.previous.pt 续训。")
                return
            for key in ("train_manifest", "val_manifest", "output"):
                if data.get(key):
                    self.values[key].set(data[key])
            for key in ("epochs", "batch_size", "learning_rate", "warmup_steps", "amp"):
                if key in data["run_config"]:
                    self.values[key].set(str(data["run_config"][key]))
            self.metric_labels["step"].configure(text=str(data["step"]))
            self.metric_labels["epoch"].configure(text=str(data["epoch"] + 1))
            self.detail_label.configure(text=f"可恢复：已完成 {data['step']} 批次，下次从第 {data['epoch'] + 1} 轮的批次 {data['next_batch'] + 1} 继续。", fg=ACCENT)
            self._persist()
            return
        if event == "training_plan":
            self.total_steps = data["total_steps"]
            self.values["resume"].set(data["resume_checkpoint"])
            self.last_checkpoint = data["resume_checkpoint"]
            self.task_label.configure(text="模型训练中")
            self.detail_label.configure(text=f"共 {data['epochs']} 轮 · 每轮 {data['batches_per_epoch']} 批次", fg=MUTED)
            self._persist()
        if "total_steps" in data:
            self.total_steps = data["total_steps"]
        if "step" in data:
            self.metric_labels["step"].configure(text=f"{data['step']:,}")
            if self.total_steps:
                self.progress.stop()
                self.progress.configure(mode="determinate", value=min(100, data["step"] / self.total_steps * 100))
        if "epoch" in data:
            self.metric_labels["epoch"].configure(text=str(data["epoch"]))
        loss = data.get("loss", data.get("train_loss"))
        if isinstance(loss, (int, float)):
            self.metric_labels["loss"].configure(text=f"{loss:.4f}")
            self.chart.add(data.get("step", 0), loss)
        if "validation" in data:
            self.metric_labels["distance"].configure(text=f"{data['validation']['median_km']:,.1f}")
        if "checkpoint" in data:
            self.last_checkpoint = data["checkpoint"]
            self.values["resume"].set(data["checkpoint"])
            self.checkpoint_label.configure(text=f"最近保存：{datetime.now():%H:%M:%S}\n{Path(data['checkpoint']).name}")
            self._persist()

    def _finished(self, job, result):
        if job == "hardware":
            if result["returncode"]:
                self.hardware.configure(text="设备检测未完成，请查看日志。")
            return
        self.active_job = None
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self._lock_controls(False)
        self.stop_button.configure(state="disabled", text="停止并保存")
        if result["returncode"] != 0:
            self.status_badge.configure(text="●  执行失败", fg=WARNING)
            self.task_label.configure(text="任务未完成")
            self.detail_label.configure(text="请查看下方日志中的错误信息，已有 checkpoint 和数据划分仍保留。", fg=WARNING)
        elif result["stopped"]:
            self.status_badge.configure(text="●  已安全停止", fg=ACCENT)
            self.task_label.configure(text="进度已保存")
            self.detail_label.configure(text="可切换为“继续上次训练”，或重新执行数据准备。", fg=MUTED)
            if job == "train":
                self.values["mode"].set("继续上次训练")
                self._mode_changed()
                self._persist()
            elif job == "split":
                self.task_label.configure(text="数据准备已停止")
                self.detail_label.configure(text="可重新执行导入；原始图片和已有划分均保留。", fg=MUTED)
        else:
            self.status_badge.configure(text="●  已完成", fg=ACCENT)
            self.task_label.configure(text="任务完成")
            if job != "inspect":
                self.detail_label.configure(text="结果已保存，可继续下一步。", fg=MUTED)
            if job == "split":
                try:
                    report = json.loads((resolve_path(self.values["split_output"].get()) / "split_report.json").read_text(encoding="utf-8"))
                    counts = report["counts"]
                    self.split_summary.configure(text=f"训练 {counts['train']:,} 张  ·  验证 {counts['val']:,} 张  ·  测试 {counts['test']:,} 张")
                except (OSError, ValueError, KeyError):
                    pass
        self.footer_text.configure(text=f"任务结束  ·  完整日志：{self.runner.log_path.name}")

    def _poll(self):
        for _ in range(180):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event.kind == "line":
                self._append_log(event.data)
                try:
                    data = json.loads(event.data)
                    if isinstance(data, dict):
                        self._structured(data)
                except (ValueError, TypeError, KeyError):
                    pass
            elif event.kind == "started":
                if event.job != "hardware":
                    self._append_log(f"[启动] {event.job} · 日志：{event.data['log_path']}")
                    self.footer_text.configure(text="任务运行中  ·  日志已自动记录")
            elif event.kind == "finished":
                self._finished(event.job, event.data)
        if self.closing and not self.runner.busy and not self.probe_runner.busy:
            self.root.destroy()
            return
        self.root.after(100, self._poll)

    def _close(self):
        self._persist()
        if self.runner.busy or self.probe_runner.busy:
            self.closing = True
            if self.runner.busy:
                self._stop()
            self.root.title("StreetGeo · 正在安全退出…")
            self.detail_label.configure(text="正在等待当前任务安全结束，然后关闭窗口。", fg=MUTED)
        else:
            self.root.destroy()

    def _open_prediction_map(self):
        interpreter = Path(sys.executable)
        if os.name == "nt" and interpreter.with_name("pythonw.exe").exists():
            interpreter = interpreter.with_name("pythonw.exe")
        log = ROOT / ".gui" / "prediction_startup.log"
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as stream:
                subprocess.Popen([str(interpreter), "-m", "streetgeo.ui.prediction"], cwd=ROOT,
                                 stdout=stream, stderr=stream,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self._append_log("已打开独立的本地预测地图窗口。")
        except OSError as error:
            self._error(str(error))

    def _open(self, path):
        path = Path(path)
        if not path.exists():
            if path.suffix:
                self._error("该文件尚未生成。")
                return
            path.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(path)
            else:
                self._append_log(str(path))
        except OSError as error:
            self._error(str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", help="alternative local preferences file")
    parser.add_argument("--no-probe", action="store_true", help="skip startup hardware detection")
    args = parser.parse_args()
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    TrainingApp(root, settings_path=args.settings, probe=not args.no_probe)
    root.mainloop()


if __name__ == "__main__":
    main()
