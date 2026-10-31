"""Local street-view prediction map and manually calibrated screenshot overlay."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from streetgeo.ui.common import (ACCENT, BG, BORDER, FONT, INPUT, MUTED, NAV, PANEL,
                                 SELECTION, SURFACE, TEXT, WARNING, ScrollFrame,
                                 apply_dark_theme, button, label)
from streetgeo.maps.projection import ControlPoint, ScreenshotCalibration, parse_coordinates
from streetgeo.maps.widgets import ScreenshotCanvas, WorldMap
from streetgeo.inference.backend import InferenceEngine, ROOT, read_picture

IMAGE_TYPES = [("图片", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"), ("所有文件", "*.*")]


class PredictionApp:
    def __init__(self, root, *, settings_path=None, default_checkpoint=None, default_device="auto"):
        self.root = root
        self.settings_path = Path(settings_path) if settings_path else ROOT / ".gui" / "prediction_settings.json"
        preferences = {}
        try:
            preferences = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if not isinstance(preferences, dict):
                preferences = {}
        except (OSError, ValueError):
            pass
        if default_checkpoint is None:
            default_checkpoint = "checkpoints/us_pilot/best.pt"
            try:
                training = json.loads((ROOT / ".gui" / "settings.json").read_text(encoding="utf-8"))
                default_checkpoint = training.get("output", default_checkpoint)
            except (OSError, ValueError, AttributeError):
                pass
        self.checkpoint = tk.StringVar(value=str(preferences.get("checkpoint", default_checkpoint)))
        device = preferences.get("device", default_device)
        self.device = tk.StringVar(value=device if device in ("auto", "cuda", "cpu") else "auto")
        self.online = tk.BooleanVar(value=False)
        self.manual = tk.StringVar()
        self.status = tk.StringVar(value="就绪 · 图片与推理均留在本机")
        self.pointer = tk.StringVar(value="滚轮缩放 · 拖动平移")
        self.images, self.candidates, self.points = [], [], []
        self.selected, self.result, self.preview_photo = 0, None, None
        self.screenshot_path, self.screenshot_hash = None, None
        self.fit, self.target = None, None
        self.adding, self.pending_geo = False, None
        self.busy, self.closing = False, False
        self.events = queue.Queue()
        self.engine = InferenceEngine()
        self.input_widgets = []
        self._poll_id, self._balance_id = None, None
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._poll_id = self.root.after(100, self._poll)

    def _build(self):
        root = self.root
        root.title("StreetGeo · 本地预测地图")
        apply_dark_theme(root)
        root.geometry(f"{min(1560, int(root.winfo_screenwidth() * .92))}x{min(1000, int(root.winfo_screenheight() * .90))}")
        root.minsize(1160, 760)
        header = tk.Frame(root, bg=NAV)
        header.pack(fill="x")
        label(header, "STREETGEO  /  本地预测地图", size=17, color="white", bold=True).pack(side="left", padx=24, pady=20)
        label(header, "街景 → 经纬度 → 地图落点", size=10, color="#9ab9c8").pack(side="right", padx=24)
        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=18, pady=(16, 6))
        left = tk.Frame(body, bg=PANEL, width=338, highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0, 14))
        left.pack_propagate(False)
        scroll = ScrollFrame(left)
        scroll.pack(fill="both", expand=True, padx=12, pady=10)
        fields = scroll.content
        fields.columnconfigure(0, weight=1)
        row = 0

        def heading(text, note=None):
            nonlocal row
            label(fields, text, size=11, bold=True).grid(row=row, column=0, sticky="w", pady=(12, 8))
            row += 1
            if note:
                label(fields, note, size=9, color=MUTED, wraplength=295, justify="left").grid(row=row, column=0, sticky="w", pady=(0, 8))
                row += 1

        heading("01  导入街景", "可选同一地点的 1–4 张透视图片。")
        actions = tk.Frame(fields, bg=PANEL)
        actions.grid(row=row, column=0, sticky="ew")
        self.input_widgets.append(button(actions, "选择图片", self.choose_images, primary=True, padx=12, pady=7))
        self.input_widgets[-1].pack(side="left")
        self.input_widgets.append(button(actions, "清空", self.clear_images, padx=12, pady=7))
        self.input_widgets[-1].pack(side="left", padx=8)
        row += 1
        self.preview = tk.Label(fields, text="尚未导入街景", bg=SURFACE, fg=MUTED, font=(FONT, 10), width=29, height=8)
        self.preview.grid(row=row, column=0, sticky="ew", pady=(10, 6))
        row += 1
        self.image_names = label(fields, "", size=8, color=MUTED, wraplength=295)
        self.image_names.grid(row=row, column=0, sticky="w")
        row += 1
        heading("02  选择模型", "预测范围固定为全球，地图缩放仅影响显示。")
        entry = ttk.Entry(fields, textvariable=self.checkpoint, font=(FONT, 9))
        entry.grid(row=row, column=0, sticky="ew")
        self.input_widgets.append(entry)
        row += 1
        model_controls = tk.Frame(fields, bg=PANEL)
        model_controls.grid(row=row, column=0, sticky="ew", pady=(7, 10))
        self.input_widgets.append(button(model_controls, "浏览 .pt", self.choose_checkpoint, padx=10, pady=5))
        self.input_widgets[-1].pack(side="left")
        label(model_controls, "设备", size=9, color=MUTED).pack(side="left", padx=(12, 6))
        self.device_box = ttk.Combobox(model_controls, textvariable=self.device, values=("auto", "cuda", "cpu"), state="readonly", width=7)
        self.device_box.pack(side="left")
        row += 1
        self.predict_button = button(fields, "开始预测 · 显示前 5 个候选", self.start_prediction, primary=True)
        self.predict_button.grid(row=row, column=0, sticky="ew")
        self.input_widgets.append(self.predict_button)
        row += 1
        self.inference_info = label(fields, "首次加载模型较慢，后续预测复用权重。", size=8, color=MUTED, wraplength=295)
        self.inference_info.grid(row=row, column=0, sticky="w", pady=(7, 0))
        row += 1
        heading("03  预测结果", "点击候选可切换地图与截图提示。")
        self.candidate_list = tk.Listbox(fields, height=5, font=("Consolas", 10), bg=INPUT, fg=TEXT,
                                        bd=0, highlightthickness=1, highlightbackground=BORDER,
                                        selectbackground=SELECTION, selectforeground=TEXT, highlightcolor=ACCENT,
                                        exportselection=False)
        self.candidate_list.grid(row=row, column=0, sticky="ew")
        self.candidate_list.bind("<<ListboxSelect>>", self.select_candidate)
        row += 1
        self.probability_label = label(fields, "", size=8, color=MUTED, wraplength=295)
        self.probability_label.grid(row=row, column=0, sticky="w", pady=(7, 0))
        row += 1
        results_actions = tk.Frame(fields, bg=PANEL)
        results_actions.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        button(results_actions, "复制坐标", self.copy_coordinates, padx=10, pady=7).pack(side="left")
        button(results_actions, "导出结果", self.export_result, padx=10, pady=7).pack(side="left", padx=8)
        row += 1
        heading("手动坐标", "没有权重时，也可测试地图与截图投射。")
        manual_entry = ttk.Entry(fields, textvariable=self.manual, font=("Consolas", 10))
        manual_entry.grid(row=row, column=0, sticky="ew")
        manual_entry.bind("<Return>", lambda event: self.use_manual())
        row += 1
        self.input_widgets.append(button(fields, "在地图上标记", self.use_manual, pady=7))
        self.input_widgets[-1].grid(row=row, column=0, sticky="ew", pady=(7, 12))

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        top = tk.Frame(right, bg=BG)
        top.pack(fill="x", pady=(0, 10))
        label(top, "预测位置", size=14, bold=True).pack(side="left")
        self.calibration_toggle = button(top, "截图校准", self.toggle_screenshot, padx=14, pady=7)
        self.calibration_toggle.pack(side="right")
        self.panes = tk.PanedWindow(right, orient="horizontal", bg=BG, sashwidth=10, bd=0)
        self.panes.pack(fill="both", expand=True)
        self.map_panel = tk.Frame(self.panes, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        self.panes.add(self.map_panel, minsize=330, stretch="always")
        tools = tk.Frame(self.map_panel, bg=PANEL)
        tools.pack(fill="x", padx=8, pady=7)
        for title, command in (("全球总览", self.world_view), ("定位候选", self.focus_candidate)):
            button(tools, title, command, padx=8, pady=5).pack(side="left", padx=2)
        self.zoom_label = label(tools, "z0", size=8, color=MUTED)
        self.zoom_label.pack(side="right", padx=4)
        layer = tk.Frame(self.map_panel, bg=PANEL)
        layer.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Checkbutton(layer, text="在线道路底图", variable=self.online,
                        command=lambda: self.map.set_online(self.online.get())).pack(side="left")
        button(layer, "−", lambda: self.map.change_zoom(-1), padx=9, pady=3).pack(side="right")
        button(layer, "+", lambda: self.map.change_zoom(1), padx=9, pady=3).pack(side="right", padx=4)
        self.map = WorldMap(self.map_panel, on_click=self.map_click,
                            on_move=lambda lat, lon: self.pointer.set(f"纬度 {lat:.6f}  ·  经度 {lon:.6f}"),
                            on_view=lambda zoom: self.zoom_label.configure(text=f"z{zoom}"))
        self.map.pack(fill="both", expand=True)
        self.screen_panel = tk.Frame(self.panes, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        screen_tools = tk.Frame(self.screen_panel, bg=PANEL)
        screen_tools.pack(fill="x", padx=10, pady=9)
        label(screen_tools, "游戏地图截图", size=10, bold=True).pack(side="left")
        button(screen_tools, "导入截图", self.choose_screenshot, padx=8, pady=5).pack(side="right")
        self.screen = ScreenshotCanvas(self.screen_panel, on_click=self.screenshot_click)
        self.screen.pack(fill="both", expand=True)
        self.screen_info = label(self.screen_panel, "截图仅作落点提示", size=9, color=MUTED, wraplength=340, justify="left")
        self.screen_info.pack(fill="x", padx=12, pady=10)

        self.calibration_panel = tk.Frame(right, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        controls = tk.Frame(self.calibration_panel, bg=PANEL)
        controls.pack(fill="x", padx=10, pady=(9, 5))
        self.add_button = button(controls, "添加对应点", self.begin_point, primary=True, padx=10, pady=5)
        self.add_button.pack(side="left")
        for title, command in (("删除所选", self.delete_point), ("清空", self.clear_points),
                               ("保存校准", self.save_calibration), ("载入校准", self.load_calibration),
                               ("导出提示图", self.export_overlay)):
            button(controls, title, command, padx=8, pady=5).pack(side="left", padx=(5, 0))
        self.calibration_hint = label(self.calibration_panel, "先导入游戏地图截图，再依次点击本地地图与截图中的同一地点。", size=9, color=MUTED, wraplength=900)
        self.calibration_hint.pack(fill="x", padx=12, pady=(3, 6))
        self.point_table = ttk.Treeview(self.calibration_panel, columns=("id", "geo", "pixel"), show="headings", height=3, selectmode="browse")
        for name, title, width in (("id", "对应点", 65), ("geo", "本地地图 · 纬度, 经度", 290), ("pixel", "原始截图 · 像素 x, y", 240)):
            self.point_table.heading(name, text=title)
            self.point_table.column(name, width=width, minwidth=50, stretch=name != "id")
        self.point_table.pack(fill="x", padx=10, pady=(0, 8))
        self.point_table.bind("<Delete>", lambda event: self.delete_point())
        footer = tk.Frame(root, bg=BG)
        footer.pack(fill="x", padx=20, pady=(2, 10))
        tk.Label(footer, textvariable=self.status, bg=BG, fg=ACCENT, font=(FONT, 9), anchor="w").pack(side="left")
        tk.Label(footer, textvariable=self.pointer, bg=BG, fg=MUTED, font=(FONT, 8), anchor="e").pack(side="right")

    def error(self, error):
        messagebox.showerror("StreetGeo", str(error), parent=self.root)

    def choose_images(self):
        paths = filedialog.askopenfilenames(parent=self.root, title="选择同一地点的 1–4 张街景", filetypes=IMAGE_TYPES)
        if not paths:
            return
        if len(paths) > 4:
            self.error("一次最多选择 4 张同一地点的街景")
            return
        try:
            thumbnails = []
            for path in paths:
                image = read_picture(path)
                image.thumbnail((288, 160) if len(paths) == 1 else (141, 79))
                thumbnails.append(image)
            canvas = Image.new("RGB", (294, 166), SURFACE)
            for index, image in enumerate(thumbnails):
                if len(paths) == 1:
                    x, y = (294 - image.width) // 2, (166 - image.height) // 2
                else:
                    x, y = (index % 2) * 147 + (147 - image.width) // 2, (index // 2) * 83 + (83 - image.height) // 2
                canvas.paste(image, (x, y))
            self.preview_photo = ImageTk.PhotoImage(canvas, master=self.root)
            self.preview.configure(image=self.preview_photo, text="", width=294, height=166)
            self.images = list(paths)
            self.image_names.configure(text="\n".join(f"{i + 1}. {Path(path).name}" for i, path in enumerate(paths)))
            self.clear_result()
            self.status.set(f"已导入 {len(paths)} 张街景 · 等待预测")
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            self.error(error)

    def clear_images(self):
        self.images = []
        self.preview_photo = None
        self.preview.configure(image="", text="尚未导入街景", width=29, height=8)
        self.image_names.configure(text="")
        self.clear_result()

    def choose_checkpoint(self):
        path = filedialog.askopenfilename(parent=self.root, title="选择训练得到的视觉模型 checkpoint",
                                          initialdir=str(ROOT / "checkpoints" if (ROOT / "checkpoints").exists() else ROOT),
                                          filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("所有文件", "*.*")])
        if path:
            self.checkpoint.set(path)

    def _busy(self, value):
        self.busy = value
        for widget in self.input_widgets:
            widget.configure(state="disabled" if value else "normal")
        self.device_box.configure(state="disabled" if value else "readonly")
        self.predict_button.configure(text="正在预测…" if value else "开始预测 · 显示前 5 个候选")

    def start_prediction(self):
        if self.busy:
            return
        if not self.images:
            self.error("请先导入街景图片")
            return
        checkpoint, device, images = self.checkpoint.get().strip(), self.device.get(), list(self.images)
        if not checkpoint:
            self.error("请选择 checkpoint")
            return
        self._busy(True)
        self.status.set("正在准备模型…")

        def worker():
            try:
                result = self.engine.predict(checkpoint, images, device=device,
                                             progress=lambda text: self.events.put(("progress", text)))
                self.events.put(("result", result))
            except Exception as error:
                self.events.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True, name="streetview-prediction").start()

    def _poll(self):
        self._poll_id = None
        for _ in range(30):
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                self.status.set(data)
            elif kind == "result":
                self._busy(False)
                self.show_result(data)
                self.inference_info.configure(text=f"{data['device']} · {data['elapsed_seconds']:.2f} 秒 · " + ("复用已加载模型" if data["reused_model"] else "已加载 checkpoint"))
                self.status.set("全球预测完成 · 可选择候选并对照地图落点")
            elif kind == "error":
                self._busy(False)
                self.status.set("预测失败 · 请检查 checkpoint、图片与设备")
                if not self.closing:
                    self.error(data)
        if self.closing and not self.busy:
            self._destroy()
            return
        self._poll_id = self.root.after(100, self._poll)

    def clear_result(self):
        self.result, self.candidates, self.selected = None, [], 0
        self.candidate_list.delete(0, "end")
        self.probability_label.configure(text="")
        self.map.candidates = []
        self.map.redraw()
        self.update_target()

    def show_result(self, result):
        self.result = result
        self.candidates = result["candidates"]
        self.selected = 0
        self.candidate_list.delete(0, "end")
        for index, candidate in enumerate(self.candidates):
            self.candidate_list.insert("end", f"{index + 1}  {candidate['latitude']:.6f}, {candidate['longitude']:.6f}")
        self.candidate_list.selection_set(0)
        self.map.candidates = self.candidates
        self.select_candidate()
        self.world_view()

    def select_candidate(self, event=None):
        selection = self.candidate_list.curselection()
        if not self.candidates or not selection:
            return
        self.selected = selection[0]
        candidate = self.candidates[self.selected]
        self.map.selected = self.selected
        self.map.redraw()
        probability = candidate.get("cell_probability")
        self.probability_label.configure(text="手动坐标 · 未经过模型预测" if probability is None else f"网格概率 {probability:.2%} · 非校准正确率")
        self.update_target()

    def use_manual(self):
        if self.busy:
            return
        try:
            latitude, longitude = parse_coordinates(self.manual.get())
            self.show_result({"source": "manual", "created_at": datetime.now(timezone.utc).isoformat(),
                              "candidates": [{"latitude": latitude, "longitude": longitude,
                                              "coordinates": f"{latitude:.6f}, {longitude:.6f}"}]})
            self.status.set("已标记手动坐标")
        except (ValueError, TypeError) as error:
            self.error(error)

    def world_view(self):
        self.map.show_world()

    def focus_candidate(self):
        if self.candidates:
            candidate = self.candidates[self.selected]
            self.map.focus(candidate["latitude"], candidate["longitude"], max(4, self.map.zoom))

    def copy_coordinates(self):
        if not self.candidates:
            self.status.set("请先预测或输入手动坐标")
            return
        text = self.candidates[self.selected]["coordinates"]
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status.set(f"已复制：{text}")

    def export_result(self):
        if self.result is None:
            self.status.set("尚无可导出的预测结果")
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="导出预测结果", defaultextension=".json",
                                            initialfile="prediction.json", filetypes=[("JSON", "*.json")])
        if path:
            try:
                result = dict(self.result, selected_candidate=self.selected + 1)
                if self.target is not None:
                    result["screenshot_hint"] = {"image": str(self.screenshot_path), "x": self.target[0], "y": self.target[1],
                                                 "coordinate_space": "original_screenshot_pixels", "rms_pixels": self.fit.rms_pixels}
                Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                self.status.set("预测结果已导出")
            except (OSError, ValueError) as error:
                self.error(error)

    def screenshot_visible(self):
        return str(self.screen_panel) in [str(pane) for pane in self.panes.panes()]

    def toggle_screenshot(self):
        if self.screenshot_visible():
            self.panes.forget(self.screen_panel)
            self.calibration_panel.pack_forget()
            self.cancel_point()
            self.calibration_toggle.configure(text="截图校准")
        else:
            self.panes.add(self.screen_panel, minsize=310, stretch="always")
            self.calibration_panel.pack(fill="x", pady=(10, 0))
            self.calibration_toggle.configure(text="收起截图")
            if self._balance_id is not None:
                self.root.after_cancel(self._balance_id)
            self._balance_id = self.root.after(100, self.balance_panes)

    def balance_panes(self):
        self._balance_id = None
        if self.screenshot_visible():
            self.panes.update_idletasks()
            self.panes.sash_place(0, self.panes.winfo_width() // 2, 0)

    def choose_screenshot(self):
        path = filedialog.askopenfilename(parent=self.root, title="导入游戏地图截图", filetypes=IMAGE_TYPES)
        if path:
            try:
                self.set_screenshot(path)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                self.error(error)

    def set_screenshot(self, path):
        path = Path(path).resolve()
        picture = read_picture(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.screen.picture = picture
        self.screenshot_path, self.screenshot_hash = path, digest
        self.points = []
        self.cancel_point()
        if not self.screenshot_visible():
            self.toggle_screenshot()
        self.refresh_calibration()
        self.status.set("已导入新截图 · 原有校准已清除，请标注对应点")

    def cancel_point(self):
        self.adding, self.pending_geo = False, None
        self.map.pending_point = None
        self.map.configure(cursor="fleur")
        self.add_button.configure(text="添加对应点")
        self.map.redraw()

    def begin_point(self):
        if self.adding:
            self.cancel_point()
            self.refresh_calibration()
            return
        if self.screen.picture is None:
            self.choose_screenshot()
            if self.screen.picture is None:
                return
        if len(self.points) >= 12:
            self.error("最多添加 12 对对应点，请先删除多余点")
            return
        self.adding = True
        self.add_button.configure(text="取消添加")
        self.map.configure(cursor="crosshair")
        self.calibration_hint.configure(text=f"对应点 {chr(65 + len(self.points))}：先点击左侧本地地图中的一个明确地点。", fg=ACCENT)

    def map_click(self, latitude, longitude):
        if self.adding:
            self.pending_geo = (latitude, longitude)
            self.map.pending_point = self.pending_geo
            self.map.redraw()
            self.calibration_hint.configure(text=f"已选本地坐标 {latitude:.5f}, {longitude:.5f}；请点击右侧截图中的同一地点。", fg=ACCENT)
        else:
            self.pointer.set(f"选中坐标：{latitude:.6f}, {longitude:.6f}")

    def screenshot_click(self, x, y):
        if self.adding and self.pending_geo is not None:
            self.points.append(ControlPoint(*self.pending_geo, x, y))
            self.cancel_point()
            self.refresh_calibration()
        elif self.adding:
            self.calibration_hint.configure(text="请先在本地地图选择对应地点，再点击截图。", fg=WARNING)

    def delete_point(self):
        selection = self.point_table.selection()
        if selection:
            self.points.pop(int(selection[0]))
            self.cancel_point()
            self.refresh_calibration()

    def clear_points(self):
        self.points = []
        self.cancel_point()
        self.refresh_calibration()

    def refresh_calibration(self):
        self.point_table.delete(*self.point_table.get_children())
        for index, point in enumerate(self.points):
            self.point_table.insert("", "end", iid=str(index), values=(chr(65 + index), f"{point.latitude:.6f}, {point.longitude:.6f}", f"{point.x:.1f}, {point.y:.1f}"))
        self.map.controls = self.points
        self.screen.controls = self.points
        self.fit = None
        if len(self.points) < 3:
            self.calibration_hint.configure(text=f"已有 {len(self.points)} 对对应点；至少需要 3 对，建议 4 对以上分散在地图各处。", fg=MUTED)
        else:
            try:
                self.fit = ScreenshotCalibration(self.points)
                note = "仅 3 点不能独立检验精度，建议再加对应点。" if len(self.points) == 3 else "请检查十字位置，误差较大时调整对应点。"
                self.calibration_hint.configure(text=f"拟合误差 RMS {self.fit.rms_pixels:.1f} 原图像素 · {note} 地图缩放或移动后须重新截图校准。",
                                                fg=ACCENT if self.fit.rms_pixels < 8 else WARNING)
            except ValueError as error:
                self.calibration_hint.configure(text=str(error), fg=WARNING)
        self.map.redraw()
        self.update_target()

    def update_target(self):
        self.target = None
        note = "截图仅作落点提示；使用相同投影的平面地图。"
        if self.fit is not None and self.candidates and self.screen.picture is not None:
            candidate = self.candidates[self.selected]
            try:
                x, y = self.fit.locate(candidate["latitude"], candidate["longitude"])
                if 0 <= x < self.screen.picture.width and 0 <= y < self.screen.picture.height:
                    self.target = (x, y)
                    note = f"建议落点：原图 x={x:.1f}, y={y:.1f}。请对照十字标记手动点击。"
                else:
                    note = "预测位置不在此截图范围内。请调整游戏地图后，重新截图与校准。"
            except ValueError as error:
                note = str(error)
        self.screen.target = self.target
        self.screen_info.configure(text=note, fg=ACCENT if self.target else MUTED)
        self.screen.draw()

    def save_calibration(self):
        if self.fit is None:
            self.status.set("请先完成至少 3 对有效对应点")
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="保存截图校准", initialfile="map_calibration.json",
                                            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            try:
                data = {"version": 1, "projection": "web_mercator_affine", "screenshot": str(self.screenshot_path),
                        "sha256": self.screenshot_hash, "size": list(self.screen.picture.size),
                        "points": [asdict(point) for point in self.points]}
                Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                self.status.set("截图校准已保存 · 仅适用于这张截图")
            except OSError as error:
                self.error(error)

    def load_calibration(self):
        path = filedialog.askopenfilename(parent=self.root, title="载入截图校准", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data.get("version") != 1 or data.get("projection") != "web_mercator_affine":
                raise ValueError("不是支持的截图校准文件")
            if not 3 <= len(data["points"]) <= 12:
                raise ValueError("校准文件的对应点数量无效")
            points = [ControlPoint(**point) for point in data["points"]]
            ScreenshotCalibration(points)
            if self.screenshot_hash != data["sha256"]:
                self.error("请先导入该校准文件对应的原始截图；当前截图不同，不能复用校准。")
                return
            width, height = self.screen.picture.size
            if list((width, height)) != data["size"] or any(not (0 <= p.x < width and 0 <= p.y < height) for p in points):
                raise ValueError("校准文件的截图尺寸或像素位置无效")
            self.points = points
            self.cancel_point()
            self.refresh_calibration()
            self.status.set("已恢复当前截图的校准")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            self.error(error)

    def export_overlay(self):
        if self.target is None:
            self.status.set("请先完成校准，并选择位于截图范围内的预测坐标")
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="导出落点提示图", initialfile="map_hint.png",
                                            defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            try:
                picture = self.screen.picture.copy()
                draw = ImageDraw.Draw(picture)
                x, y = self.target
                radius = max(12, min(picture.size) // 35)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="#e33955", width=3)
                draw.line((x - radius * 2, y, x + radius * 2, y), fill="#e33955", width=3)
                draw.line((x, y - radius * 2, x, y + radius * 2), fill="#e33955", width=3)
                picture.save(path)
                self.status.set("已导出原始尺寸的落点提示图")
            except OSError as error:
                self.error(error)

    def close(self):
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.settings_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"checkpoint": self.checkpoint.get(), "device": self.device.get()}, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, self.settings_path)
        except OSError:
            pass
        if self.busy:
            self.closing = True
            self.status.set("正在等待当前推理完成后退出…")
        else:
            self._destroy()

    def _destroy(self):
        for callback in (self._poll_id, self._balance_id):
            if callback is not None:
                self.root.after_cancel(callback)
        self._poll_id = self._balance_id = None
        self.map.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", help="alternative local preferences file")
    args = parser.parse_args()
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    PredictionApp(root, settings_path=args.settings)
    root.mainloop()


if __name__ == "__main__":
    main()
