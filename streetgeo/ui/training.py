"""Chinese desktop workbench for Mapillary download, preparation and resumable training."""

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
from streetgeo import PROJECT_SHORT_NAME
from tkinter import filedialog, ttk

from streetgeo.ui.backend import (ROOT, MODEL_PRESETS, CUSTOM_MODEL_PRESET, ProcessRunner,
                         build_download, build_prepare, build_gpu_check, build_training, build_road_mask, load_settings, model_preset_for_config,
                         new_model_output, resolve_path, save_settings)

from streetgeo.ui.common import (BG, PANEL, NAV, TEXT, MUTED, BORDER, ACCENT, FONT,
                                 SURFACE, INPUT, DISABLED, SELECTION, WARNING, CHART_FILL,
                                 apply_dark_theme, label, button, ScrollFrame, SidebarLayout)


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
        self.values["token"] = tk.StringVar(value=os.environ.get("MAPILLARY_ACCESS_TOKEN", ""))
        self.download_then_prepare = False
        self.download_status = None
        self.events = queue.Queue()
        self.runner = ProcessRunner(self.events)
        self.probe_runner = ProcessRunner(self.events)
        self.pages, self.nav_buttons, self.editors, self.browse_buttons = {}, {}, {}, {}
        self.all_editors, self.all_browse_buttons = [], []
        self.action_buttons = []
        self.active_page, self.active_job = "train", None
        self.total_steps = 0
        self.last_checkpoint = ""
        self.closing = False
        self.mask_preview_window = None
        self._build()
        self._show_page("download")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll)
        if probe:
            self.root.after(300, self._probe)

    def _build(self):
        root = self.root
        root.title(f"{PROJECT_SHORT_NAME} · 训练工作台")
        apply_dark_theme(root)
        width = min(1460, int(root.winfo_screenwidth() * 0.9))
        height = min(1020, int(root.winfo_screenheight() * 0.9))
        root.geometry(f"{width}x{height}")
        root.minsize(1080, 780)
        self.sidebar_layout = SidebarLayout(root, width=self.values["sidebar_width"].get(), default_width=196,
                                            min_sidebar=176, max_sidebar=420, min_content=820,
                                            sidebar_bg=NAV, on_resize=self._sidebar_resized)
        self.sidebar_layout.pack(fill="both", expand=True)
        self.values["sidebar_width"].set(str(self.sidebar_layout.sidebar_width))
        sidebar = self.sidebar_layout.sidebar
        brand = tk.Canvas(sidebar, height=94, bg=NAV, highlightthickness=0)
        brand.pack(fill="x", padx=18, pady=(22, 15))
        brand.create_oval(1, 2, 33, 34, outline="#60d4bd", width=2)
        brand.create_oval(10, 2, 24, 34, outline="#60d4bd", width=1)
        brand.create_line(1, 18, 33, 18, fill="#60d4bd")
        brand.create_text(0, 58, text=PROJECT_SHORT_NAME, anchor="w", fill="white", font=("Segoe UI", 12, "bold"))
        brand.create_text(0, 80, text="TRAINING STUDIO", anchor="w", fill="#8aa4b8", font=("Segoe UI", 8))
        label(sidebar, "工作流程", color="#7f9bad", size=9).pack(fill="x", padx=22, pady=(10, 12))
        for key, text in (("download", "01   Mapillary 下载"), ("split", "02   导入与划分"), ("train", "03   模型训练")):
            item = tk.Button(sidebar, text=text, anchor="w", command=lambda key=key: self._show_page(key),
                             font=(FONT, 10), bg=NAV, fg="#bdceda", activebackground="#263e50",
                             activeforeground="white", relief="flat", bd=0, padx=18, pady=14, cursor="hand2")
            item.pack(fill="x", padx=10, pady=3)
            self.nav_buttons[key] = item
        tk.Button(sidebar, text="04   预测地图", anchor="w", command=self._open_prediction_map,
                  font=(FONT, 10), bg=NAV, fg="#bdceda", activebackground="#263e50",
                  activeforeground="white", relief="flat", bd=0, padx=18, pady=14,
                  cursor="hand2").pack(fill="x", padx=10, pady=3)
        footer = tk.Frame(sidebar, bg=NAV)
        footer.pack(side="bottom", fill="x", padx=20, pady=22)
        label(footer, "本地运行 · 自动保存", color="#7f9bad", size=9).pack(anchor="w", pady=(0, 12))
        for text, command in (("GPU 运行检查", self._gpu_check), ("打开项目目录", lambda: self._open(ROOT)),
                              ("Mapillary 训练指南", lambda: self._open(ROOT / "docs" / "training" / "mapillary.md"))):
            tk.Button(footer, text=text, command=command, anchor="w", bg=NAV, fg="#bdceda",
                      activebackground=NAV, activeforeground="white", font=(FONT, 9),
                      relief="flat", bd=0, cursor="hand2", pady=6).pack(fill="x")

        main = tk.Frame(self.sidebar_layout.content, bg=BG)
        main.pack(fill="both", expand=True, padx=24, pady=(22, 16))
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
        self._build_download()
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
        self.footer_text = label(main, "就绪  ·  配置仅保存在本机 · 仅接受 Mapillary", color=MUTED, size=9)
        self.footer_text.pack(fill="x", pady=(10, 0))
        self._append_log(f"欢迎使用 {PROJECT_SHORT_NAME}（Street-View Geolocation Transformer）训练工作台。仅支持 Mapillary。可选原版 1.35 亿或新版 3.98 亿参数；切换架构需新建训练。")

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
        if key == "token":
            editor.configure(show="●")
        editor.grid(row=1, column=0, sticky="ew", ipady=6 if not choices else 0)
        self.editors[key] = editor
        self.all_editors.append(editor)
        if browse:
            chooser = button(frame, "选择", lambda: self._browse(key, browse), padx=10, pady=6)
            chooser.grid(row=1, column=1, padx=(8, 0))
            self.browse_buttons[key] = chooser
            self.all_browse_buttons.append(chooser)
        return editor

    def _action(self, parent, text, command, primary=False):
        item = button(parent, text, command, primary=primary)
        item.pack(side="left", padx=(0, 10))
        self.action_buttons.append(item)
        return item

    def _build_download(self):
        from streetgeo.datasets.regions import CONTINENTS, country_catalog, region_catalog
        fields, actions = self._page("download", "Mapillary API 自动下载", "展开大洲选择国家/地区；按 Ctrl 或 Shift 可多选。多项选择取并集，自定义边界进一步限定范围。")
        self._field(fields, "Mapillary Client Access Token（仅本次运行，不保存）", "token", 0, span=2)
        self._field(fields, "下载目录 · 相同选区与参数可断点继续", "mapillary_root", 1, span=2, browse="directory")
        tree_frame = tk.Frame(fields, bg=PANEL)
        tree_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=(2, 12), pady=8)
        self.region_tree = ttk.Treeview(tree_frame, show="tree", selectmode="extended", height=9)
        scrollbar = ttk.Scrollbar(tree_frame, command=self.region_tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.region_tree.configure(yscrollcommand=scrollbar.set)
        self.region_tree.pack(fill="x", expand=True)
        for key, (title, _) in CONTINENTS.items():
            self.region_tree.insert("", "end", iid="continent:" + key, text=title)
        for country in country_catalog():
            self.region_tree.insert("continent:" + country["continent"], "end", iid="country:" + country["code"],
                                    text=country["name"] + " · " + country["code"])
        for region in sorted(region_catalog(), key=lambda item: item["code"]):
            self.region_tree.insert("country:" + region["country"], "end", iid="region:" + region["code"], text=region["code"])
        selected = []
        for kind, key in (("continent", "download_continents"), ("country", "download_countries"), ("region", "download_regions")):
            for code in self.values[key].get().split(","):
                item = kind + ":" + code.strip()
                if self.region_tree.exists(item):
                    selected.append(item)
        self.region_tree.selection_set(selected)
        if selected:
            self.region_tree.see(selected[0])
        self.region_tree.bind("<<TreeviewSelect>>", self._region_selection_changed)
        self.region_tree.bind("<Button-1>", lambda _: "break" if self.active_job else None)
        self.region_tree.bind("<KeyPress>", lambda _: "break" if self.active_job else None)
        self._field(fields, "已选大洲代码（可编辑，逗号分隔）", "download_continents", 3, span=2)
        self._field(fields, "已选国家/地区代码 · ISO 三字母", "download_countries", 4)
        self._field(fields, "已选行政区代码（内置美国州）", "download_regions", 4, 1)
        self._field(fields, "任意省市/地区边界 · WGS84 GeoJSON（可选）", "download_geojson", 5, span=2, browse="geojson")
        self._field(fields, "自定义地区范围 · 西经度 南纬度 东经度 北纬度（可选）", "download_bbox", 6, span=2)
        self._field(fields, "下载目标总张数（含已下载）", "max_images", 7)
        self._field(fields, "每次最多查询页数", "max_pages", 7, 1)
        self._field(fields, "图片容量上限 / GiB", "max_gb", 8)
        self._field(fields, "至少保留磁盘 / GiB", "min_free_gb", 8, 1)
        self._field(fields, "下载并发数", "workers", 9)
        self._field(fields, "每秒 API 请求数", "requests_per_second", 9, 1)
        self._field(fields, "优先下载分辨率", "resolution", 10, span=2, choices=("256", "1024", "2048"))
        self._action(actions, "下载并划分", lambda: self._download(prepare=True), primary=True)
        self._action(actions, "仅下载 / 继续", self._download)

    def _region_selection_changed(self, _event=None):
        if self.active_job:
            return
        selected = self.region_tree.selection()
        for kind, key in (("continent", "download_continents"), ("country", "download_countries"), ("region", "download_regions")):
            self.values[key].set(",".join(sorted(item.split(":", 1)[1] for item in selected if item.startswith(kind + ":"))))

    def _download(self, prepare=False):
        try:
            stop = self.runner.new_stop_file()
            arguments = build_download(self._settings(), stop)
            token = self.values["token"].get().strip()
            if not token:
                raise ValueError("请填写 Mapillary Client Access Token，或设置 MAPILLARY_ACCESS_TOKEN 环境变量")
            self.download_then_prepare, self.download_status = prepare, None
            self._start("download", "streetgeo.datasets.download_mapillary", arguments, stop_file=stop,
                        extra_env={"MAPILLARY_ACCESS_TOKEN": token})
        except ValueError as error:
            self._error(error)

    def _build_split(self):
        fields, actions = self._page("split", "Mapillary 导入与划分", "仅导入 Mapillary API 下载的图片清单；按采集序列与地理位置划分。")
        label(fields, "数据来源：Mapillary（固定）", color=ACCENT, size=10).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=2, pady=(8, 12))
        self._field(fields, "Mapillary 下载目录 · 含 all.jsonl 与 images/", "mapillary_root", 1, span=2, browse="directory")
        self._field(fields, "划分结果保存目录", "split_output", 2, span=2, browse="directory")
        self._field(fields, "地理分组大小 / 度", "block_degrees", 3)
        self._field(fields, "不同集合间的隔离距离 / 米", "guard_m", 3, 1)
        description = label(fields, "只导入已完成下载的 Mapillary 图片，逐条校验来源与原始图片 ID。\n按采集序列和地理块整体分组，训练、验证、测试之间隔离近邻。\n补充下载图片后，请使用新的划分输出目录。",
                            size=9, color=MUTED, wraplength=500, justify="left")
        description.grid(row=4, column=0, columnspan=2, sticky="w", padx=2, pady=14)
        self.split_summary = label(fields, "尚未生成数据划分", size=10, color=ACCENT, wraplength=500)
        self.split_summary.grid(row=5, column=0, columnspan=2, sticky="w", padx=2, pady=8)
        self._action(actions, "导入并生成划分", self._prepare, primary=True)
        self._action(actions, "填入训练配置", self._use_splits)

    def _build_train(self):
        fields, actions = self._page("train", "Mapillary 全球街景训练", "可选原版 1.35 亿或新版 3.98 亿参数；续训沿用 checkpoint 架构。切换规模会填入建议批次大小和独立输出目录。")
        preset = self._field(fields, "模型规模", "model_preset", 0, span=2,
                             choices=(*MODEL_PRESETS, CUSTOM_MODEL_PRESET))
        preset.bind("<<ComboboxSelected>>", lambda _: self._model_preset_changed())
        self.values["model_config"].trace_add("write", lambda *_: self._model_config_changed())
        editor = self._field(fields, "运行方式", "mode", 1, choices=("新建训练", "继续上次训练"))
        editor.bind("<<ComboboxSelected>>", lambda _: self._mode_changed())
        self._field(fields, "训练设备", "device", 1, 1, choices=("cuda", "auto", "cpu"))
        self._field(fields, "恢复进度 / *.last.pt", "resume", 2, span=2, browse="checkpoint")
        self._field(fields, "Mapillary 训练清单", "train_manifest", 3, span=2, browse="manifest")
        self._field(fields, "Mapillary 验证清单（可留空）", "val_manifest", 4, span=2, browse="manifest")
        self._field(fields, "最佳模型输出文件", "output", 5, span=2, browse="save_checkpoint")
        self._field(fields, "训练轮数 / epochs", "epochs", 6)
        self._field(fields, "批次大小 / batch size", "batch_size", 6, 1)
        self._field(fields, "训练精度", "amp", 7, choices=("bf16", "fp16", "off", "auto"))
        self._field(fields, "数据读取进程数", "num_workers", 7, 1)
        self._field(fields, "每多少批次保存一次", "save_every_steps", 8)
        self._field(fields, "最长保存间隔 / 秒（0 关闭）", "save_every_seconds", 8, 1)
        self._field(fields, "初始学习率", "learning_rate", 9)
        self._field(fields, "预热步数", "warmup_steps", 9, 1)
        self._field(fields, "模型配置 JSON（可自定义）", "model_config", 10, span=2, browse="json")
        self._field(fields, "新阶段初始权重（可选）", "init_from", 11, span=2, browse="checkpoint")
        environment = self._field(fields, "环境专项微调 · 使用已有模型", "environment_specialist", 12, span=2,
                                  choices=("关闭", "开启 · 全图/专项交替"))
        environment.bind("<<ComboboxSelected>>", lambda _: self._environment_changed())
        self._field(fields, "末端视觉模块解冻层数", "unfreeze_last", 13)
        self._field(fields, "末端视觉模块学习率", "visual_learning_rate", 13, 1)
        self._field(fields, "梯形上底 / 原图宽度（%）", "road_mask_top_width", 14)
        self._field(fields, "梯形下底 / 原图宽度（%）", "road_mask_bottom_width", 14, 1)
        self._field(fields, "梯形高度 / 原图高度（%）", "road_mask_height", 15)
        self.mask_preview_button = button(fields, "选择图片预览梯形遮挡", self._preview_environment_mask, padx=8, pady=8)
        self.mask_preview_button.grid(row=15, column=1, sticky="ew", padx=(2, 12), pady=(20, 8))
        self.action_buttons.append(self.mask_preview_button)
        label(fields, "专项遮挡底部居中的梯形，每批先全图、再专项各训练一次；坐标监督相同。\n默认上底 20%、下底 100%、高 45%，可预览调整；弯道、侧拍需单独检查。\n需选择初始模型与验证清单，模型规模须与初始模型一致；每轮分别验证全图和专项。",
              size=9, color=MUTED, wraplength=500, justify="left").grid(
                  row=16, column=0, columnspan=2, sticky="w", padx=2, pady=10)
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
        self.environment_metric = label(inner, "专项验证中位误差：—", size=9, color=MUTED)
        self.environment_metric.pack(fill="x", pady=(0, 8))
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
        keys = ("model_preset", "train_manifest", "val_manifest", "output", "epochs", "batch_size", "amp", "learning_rate",
                "warmup_steps", "model_config", "init_from", "environment_specialist", "unfreeze_last", "visual_learning_rate",
                "road_mask_top_width", "road_mask_bottom_width", "road_mask_height")
        for key in keys:
            editor = self.editors.get(key)
            if editor is not None:
                editor.configure(state="disabled" if resume else ("readonly" if isinstance(editor, ttk.Combobox) else "normal"))
            if key in self.browse_buttons:
                self.browse_buttons[key].configure(state="disabled" if resume else "normal")
        if "resume" in self.editors:
            self.editors["resume"].configure(state="normal" if resume else "disabled")
            self.browse_buttons["resume"].configure(state="normal" if resume else "disabled")
        if not resume and self.values["environment_specialist"].get() == "关闭":
            for key in ("unfreeze_last", "visual_learning_rate", "road_mask_top_width", "road_mask_bottom_width", "road_mask_height"):
                self.editors[key].configure(state="disabled")
        self.mask_preview_button.configure(state="disabled" if resume or self.values["environment_specialist"].get() == "关闭" else "normal")

    def _preview_environment_mask(self):
        from PIL import Image, ImageDraw, ImageOps, ImageTk
        try:
            build_road_mask(self._settings())
            path = filedialog.askopenfilename(parent=self.root, title="选择街景预览遮挡范围",
                     initialdir=str(resolve_path(self.values["mapillary_root"].get()) / "images"),
                     filetypes=[("街景图片", "*.jpg *.jpeg *.png *.webp"), ("所有文件", "*")])
            if not path:
                return None
            with Image.open(path) as source:
                picture = ImageOps.exif_transpose(source).convert("RGB")
        except (ValueError, OSError) as error:
            self._error(error)
            return None
        if self.mask_preview_window is not None and self.mask_preview_window.winfo_exists():
            self.mask_preview_window.destroy()
        window = self.mask_preview_window = tk.Toplevel(self.root)
        window.title("环境专项 · 梯形遮挡预览")
        window.geometry("1100x650")
        window.minsize(650, 400)
        window.configure(bg=BG)
        window.transient(self.root)
        info = label(window, color=ACCENT, size=10, wraplength=1000)
        info.pack(fill="x", padx=16, pady=12)
        panels = tk.Frame(window, bg=BG)
        panels.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        panels.columnconfigure((0, 1), weight=1)
        panels.rowconfigure(1, weight=1)
        canvases = []
        for column, title in enumerate(("原图与遮挡轮廓", "专项实际输入（缩放前）")):
            label(panels, title, size=10).grid(row=0, column=column, pady=6)
            canvas = tk.Canvas(panels, bg=INPUT, highlightthickness=0)
            canvas.grid(row=1, column=column, sticky="nsew", padx=5)
            canvases.append(canvas)

        def render(*_):
            if not window.winfo_exists():
                return
            try:
                mask = build_road_mask(self._settings())
            except ValueError as error:
                info.configure(text=str(error), fg=WARNING)
                return
            window.mask_config = mask
            info.configure(text=f"上底 {mask.top_width:.0%} W · 下底 {mask.bottom_width:.0%} W · 高 {mask.height:.0%} H"
                                f" · 理论遮挡面积 {mask.area_fraction:.1%}；修改工作台参数可实时预览。", fg=ACCENT)
            outlined = picture.copy()
            ImageDraw.Draw(outlined).polygon(mask.vertices(picture.size), outline=ACCENT, width=max(2, picture.width // 200))
            for canvas, image in zip(canvases, (outlined, mask.apply(picture))):
                small = image.copy()
                small.thumbnail((max(1, canvas.winfo_width() - 12), max(1, canvas.winfo_height() - 12)))
                canvas.photo = ImageTk.PhotoImage(small, master=window)
                canvas.delete("all")
                canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, image=canvas.photo)

        traces = [(self.values[key], self.values[key].trace_add("write", render))
                  for key in ("road_mask_top_width", "road_mask_bottom_width", "road_mask_height")]
        for canvas in canvases:
            canvas.bind("<Configure>", render)

        def closed(event):
            if event.widget == window:
                for variable, trace in traces:
                    variable.trace_remove("write", trace)
                if self.mask_preview_window == window:
                    self.mask_preview_window = None
        window.bind("<Destroy>", closed)
        render()
        return window

    def _environment_changed(self):
        enabled = self.values["environment_specialist"].get() != "关闭"
        self.values["learning_rate"].set("0.00002" if enabled else "0.0003")
        self.values["epochs"].set("2" if enabled else "4")
        self.values["warmup_steps"].set("100" if enabled else "1000")
        candidate = new_model_output(self.values["model_config"].get(), "environment.pt" if enabled else "best.pt")
        self.values["output"].set(str(candidate))
        self.values["resume"].set(str(candidate.with_name(candidate.stem + ".last" + candidate.suffix)))
        self._mode_changed()
        self._persist()

    def _model_config_changed(self):
        self.values["model_preset"].set(model_preset_for_config(self.values["model_config"].get()))

    def _model_preset_changed(self):
        name = self.values["model_preset"].get()
        if name == CUSTOM_MODEL_PRESET:
            self._browse("model_config", "json")
            self._model_config_changed()
            return
        preset = MODEL_PRESETS[name]
        self.values["model_config"].set(preset["config"])
        self.values["batch_size"].set(preset["batch_size"])
        self.values["amp"].set("bf16" if self.values["device"].get() != "cpu" else "off")
        self.values["init_from"].set("")
        candidate = new_model_output(preset["config"], "environment.pt" if self.values["environment_specialist"].get() != "关闭" else "best.pt")
        self.values["output"].set(str(candidate))
        self.values["resume"].set(str(candidate.with_name(candidate.stem + ".last" + candidate.suffix)))
        self._persist()
        self.detail_label.configure(text=f"已选择{name}；新建训练，建议 batch {preset['batch_size']}。", fg=ACCENT)

    def _show_page(self, key):
        self.active_page = key
        self.pages[key].tkraise()
        titles = {"train": ("模型训练", "设置训练参数，随时保存进度并继续。"),
                  "split": ("数据导入与划分", "仅 Mapillary · 全球坐标 · 地点隔离。"),
                  "download": ("Mapillary 街景下载", "输入 API Token，选择大洲、国家或自定义地区，自动下载并保存进度。")}
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
                       "json": [("JSON 配置", "*.json"), ("所有文件", "*")],
                       "geojson": [("区域 GeoJSON", "*.geojson *.json"), ("所有文件", "*")]}
            value = filedialog.askopenfilename(**options, filetypes=filters[kind])
        if value:
            self.values[key].set(value)

    def _settings(self):
        return {key: value.get() for key, value in self.values.items()}

    def _sidebar_resized(self, width):
        self.values["sidebar_width"].set(str(width))
        self._persist()

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
        for editor in self.all_editors:
            editor.configure(state="disabled" if locked else ("readonly" if isinstance(editor, ttk.Combobox) else "normal"))
        for item in (*self.all_browse_buttons, *self.action_buttons):
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
        names = {"download": "正在下载 Mapillary 街景", "split": "正在导入与划分数据", "train": "训练启动中", "inspect": "正在读取 checkpoint", "gpu": "正在检查 GPU"}
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
        from streetgeo.datasets.policy import check_mapillary_manifest_header
        try:
            for name in ("train", "val", "test"):
                check_mapillary_manifest_header(output / f"{name}.jsonl")
        except (ValueError, OSError) as error:
            self._error(str(error))
            return
        self.values["train_manifest"].set(str(output / "train.jsonl"))
        self.values["val_manifest"].set(str(output / "val.jsonl"))
        self.values["mode"].set("新建训练")
        environment = self.values["environment_specialist"].get() != "关闭"
        candidate = new_model_output(self.values["model_config"].get(), "environment.pt" if environment else "best.pt")
        self.values["output"].set(str(candidate))
        self.values["resume"].set(str(candidate.with_name(candidate.stem + ".last" + candidate.suffix)))
        if not environment:
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
            self.environment_metric.configure(text="专项验证中位误差：—")
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
        try:
            self._start("gpu", "streetgeo.cli.check_gpu", build_gpu_check(self._settings()))
        except ValueError as error:
            self._error(error)

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        line_count = int(self.log.index("end-1c").split(".")[0])
        if line_count > 2200:
            self.log.delete("1.0", f"{line_count - 2000}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _structured(self, data):
        if data.get("event") == "download_finished":
            self.download_status = data["status"]
            self.detail_label.configure(text=f"已下载 {data['downloaded']:,} 张 · {data['bytes'] / 1024 ** 3:.2f} GiB · {data['status']}", fg=ACCENT)
        elif "downloaded" in data:
            self.detail_label.configure(text=f"已下载 {data['downloaded']:,} 张 · 失败 {data.get('failed', 0):,} 张", fg=MUTED)
        if data.get("event") == "dataset_import_progress":
            self.progress.stop()
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
            self.detail_label.configure(text=f"Mapillary：已校验并导入 {data['samples']:,} 张图片…", fg=MUTED)
        elif data.get("event") == "dataset_import_complete":
            self.split_summary.configure(text=f"Mapillary：已导入 {data['samples']:,} 张图片。")
        elif data.get("event") == "dataset_splitting":
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
            self.detail_label.configure(text=f"已导入 {data['samples']:,} 张，正在按地点与地理块划分…", fg=MUTED)
        elif data.get("event") == "dataset_ready":
            self.detail_label.configure(text=f"已准备 {data['samples']:,} 张 · " +
                                             " / ".join(f"{key}: {value:,}" for key, value in data["source_counts"].items()), fg=ACCENT)
        event = data.get("event")
        if event == "validation_progress":
            completed, total = data["batch"], data["total_batches"]
            self.progress.stop()
            self.progress.configure(mode="determinate", value=100 * completed / max(1, total))
            self.status_badge.configure(text="●  验证中", fg=ACCENT)
            view = "专项遮挡" if data.get("validation_view") == "environment" else "全图"
            title = "初始模型基线" if data.get("baseline") else f"第 {data['epoch']} 轮"
            self.task_label.configure(text=f"{title} · {view}验证中")
            self.detail_label.configure(text=f"正在评估{view}定位精度。\n验证 {completed:,}/{total:,} 批 · {data['samples']:,}/{data['total_samples']:,} 张图片", fg=MUTED)
            return
        if event == "baseline_evaluation":
            self.metric_labels["distance"].configure(text=f"{data['full']['median_km']:,.1f}")
            self.environment_metric.configure(text=f"基线专项中位误差：{data['environment']['median_km']:,.1f} km")
            return
        if event == "training_phase":
            saving = data["phase"] == "saving"
            self.status_badge.configure(text="●  保存中" if saving else "●  训练中", fg=ACCENT)
            self.task_label.configure(text="正在保存模型" if saving else "模型训练中")
            self.detail_label.configure(text="正在写入 checkpoint，保存完成后将继续或结束任务。" if saving else "正在处理训练批次；每轮结束后还需验证和保存。", fg=MUTED)
            if saving:
                self.progress.stop()
                self.progress.configure(mode="indeterminate")
                self.progress.start(15)
            return
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
            for key in ("epochs", "batch_size", "learning_rate", "warmup_steps", "amp", "unfreeze_last", "visual_learning_rate"):
                if key in data["run_config"]:
                    self.values[key].set(str(data["run_config"][key]))
            self.values["environment_specialist"].set("开启 · 全图/专项交替" if data["run_config"].get("environment_specialist") else "关闭")
            for key in ("road_mask_top_width", "road_mask_bottom_width", "road_mask_height"):
                if key in data["run_config"]:
                    self.values[key].set(f"{100 * data['run_config'][key]:.6g}")
            if "model_config" in data:
                name = model_preset_for_config(data["model_config"])
                if name in MODEL_PRESETS:
                    self.values["model_config"].set(MODEL_PRESETS[name]["config"])
                self.values["model_preset"].set(name)
            self.metric_labels["step"].configure(text=str(data["step"]))
            self.metric_labels["epoch"].configure(text=str(data["epoch"] + 1))
            phase = "专项遮挡" if data.get("next_phase") == "environment" else "全图"
            self.detail_label.configure(text=f"可恢复：已完成 {data['step']} 批次，下次从第 {data['epoch'] + 1} 轮的批次 {data['next_batch'] + 1}（{phase}）继续。", fg=ACCENT)
            if data["run_config"].get("environment_specialist") and (data.get("environment_recipe") or {}).get("version", 1) == 1:
                self.detail_label.configure(text=self.detail_label.cget("text") + "旧断点沿用矩形遮挡；切换梯形请新建微调。")
            self._persist()
            return
        if event == "training_plan":
            self.total_steps = data["total_steps"]
            self.values["resume"].set(data["resume_checkpoint"])
            self.last_checkpoint = data["resume_checkpoint"]
            self.task_label.configure(text="模型训练中")
            self.detail_label.configure(text=f"共 {data['epochs']} 轮 · 每轮 {data['batches_per_epoch']} 批次", fg=MUTED)
            if data.get("environment_specialist"):
                shape = "梯形专项" if (data.get("environment_recipe") or {}).get("version") == 2 else "旧矩形专项"
                self.detail_label.configure(text=f"全图/{shape} 1:1 交替 · 每轮 {data['source_batches_per_epoch']} 组图片、{data['batches_per_epoch']} 次更新", fg=MUTED)
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
        if "environment_validation" in data:
            self.environment_metric.configure(text=f"专项验证中位误差：{data['environment_validation']['median_km']:,.1f} km")
        if "input_view" in data:
            phase = "专项遮挡" if data["input_view"] == "environment" else "全图"
            self.task_label.configure(text=f"模型训练中 · {phase}")
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
        if job == "download":
            if self.download_status in ("paused", "stopped"):
                self.status_badge.configure(text="●  下载已暂停", fg=WARNING)
                self.detail_label.configure(text="下载进度已保留；稍后使用相同目录和选区继续。", fg=MUTED)
            elif result["returncode"] == 0 and not result["stopped"] and self.download_then_prepare and not self.closing:
                self.download_then_prepare = False
                self.root.after(100, self._prepare)
        elif job == "split" and result["returncode"] == 0 and not result["stopped"] and not self.closing:
            self._use_splits()

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
            self.root.title(f"{PROJECT_SHORT_NAME} · 正在安全退出…")
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
