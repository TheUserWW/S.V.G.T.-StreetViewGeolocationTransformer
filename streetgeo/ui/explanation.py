"""Shared bilingual prediction explanation window; model work uses the app worker."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from streetgeo import PROJECT_SHORT_NAME
from streetgeo.inference.explanation_text import translate
from streetgeo.ui.common import BG, PANEL, INPUT, TEXT, MUTED, WARNING, ACCENT, button, label


class PictureCanvas(tk.Canvas):
    """Aspect-preserving picture with selection in oriented source coordinates."""

    def __init__(self, parent, *, on_select=None, language="zh"):
        super().__init__(parent, bg=INPUT, highlightthickness=0, width=360, height=260)
        self.picture = self.photo = None
        self.bounds = None
        self.rectangle = None
        self.on_select = on_select
        self.language = language
        self.enabled = True
        self.start = None
        self._resize_id = None
        self.bind("<Configure>", self._resize)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)

    def set_picture(self, picture, rectangle=None):
        self.picture = picture
        self.rectangle = rectangle
        self.redraw()

    def _resize(self, event=None):
        if self._resize_id is None:
            self._resize_id = self.after_idle(self.redraw)

    def redraw(self):
        if self._resize_id is not None:
            self.after_cancel(self._resize_id)
            self._resize_id = None
        self.delete("all")
        if self.picture is None:
            self.create_text(max(1, self.winfo_width()) / 2, max(1, self.winfo_height()) / 2,
                             text=translate("等待分析…", self.language), fill=MUTED)
            return
        width, height = max(1, self.winfo_width()), max(1, self.winfo_height())
        shown = ImageOps.contain(self.picture, (width, height), Image.Resampling.LANCZOS)
        if shown.width == 0 or shown.height == 0:
            self.bounds = None
            return
        left, top = (width - shown.width) / 2, (height - shown.height) / 2
        self.bounds = left, top, shown.width, shown.height
        self.photo = ImageTk.PhotoImage(shown, master=self)
        self.create_image(left, top, anchor="nw", image=self.photo)
        self._draw_rectangle()

    def _draw_rectangle(self):
        self.delete("selection")
        if self.rectangle is not None and self.bounds is not None:
            left, top, width, height = self.bounds
            x0, y0, x1, y1 = self.rectangle
            self.create_rectangle(left + x0 * width, top + y0 * height,
                                  left + x1 * width, top + y1 * height,
                                  outline=WARNING, width=3, tags="selection")

    def _point(self, event, clamp=False):
        if self.bounds is None:
            return None
        left, top, width, height = self.bounds
        x, y = (event.x - left) / width, (event.y - top) / height
        if not clamp and not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        return min(1, max(0, x)), min(1, max(0, y))

    def _press(self, event):
        if self.enabled and self.on_select is not None:
            self.start = self._point(event)

    def _drag(self, event):
        if self.start is None:
            return
        point = self._point(event, clamp=True)
        self.rectangle = (min(self.start[0], point[0]), min(self.start[1], point[1]),
                          max(self.start[0], point[0]), max(self.start[1], point[1]))
        self._draw_rectangle()

    def _release(self, event):
        if self.start is None:
            return
        self._drag(event)
        self.start = None
        x0, y0, x1, y1 = self.rectangle
        if x1 > x0 and y1 > y0:
            self.on_select(self.rectangle)
        else:
            self.rectangle = None
            self._draw_rectangle()

    def destroy(self):
        if self._resize_id is not None:
            self.after_cancel(self._resize_id)
            self._resize_id = None
        self.photo = self.picture = None
        super().destroy()


class ExplanationWindow(tk.Toplevel):
    def __init__(self, parent, *, request, cancel, language="zh"):
        super().__init__(parent)
        self.language = language
        self._texts = []
        self._status_source, self._status_values = "正在准备模型…", {}
        self.title(f"{PROJECT_SHORT_NAME} · {self.t('预测解释')}")
        self.configure(bg=BG)
        self.geometry("1200x860")
        self.minsize(920, 720)
        self.request, self.cancel = request, cancel
        self.analysis = self.occlusion = None
        self.rectangle = None
        self.closed = self.working = False
        self.view = tk.StringVar(value=self.t("图片 {index}", index=1))
        self.query = tk.StringVar(value=self.t("查询平均"))
        self.opacity = tk.DoubleVar(value=0.5)
        self.status = tk.StringVar(value=self.t("正在准备模型…"))
        self._heat_key = self._heat = self._plain = None
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        self._text(label(self, size=17, bold=True), "预测解释 · 池化注意力与遮挡实验").grid(
            row=0, column=0, sticky="ew", padx=18, pady=(14, 8))
        self.note = self._text(label(self, size=9, color=MUTED, wraplength=1100, justify="left"),
                              "蓝色较弱 → 红色较强；每幅图独立归一化。注意力是特征聚合线索，不是预测原因。")
        self.note.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 8))
        self.bind("<Configure>", self._wrap_note)
        controls = tk.Frame(self, bg=PANEL)
        controls.grid(row=2, column=0, sticky="ew", padx=18)
        controls.columnconfigure(5, weight=1)
        self._text(label(controls, size=10), "视角").grid(row=0, column=0, padx=6, pady=8)
        self.view_box = ttk.Combobox(controls, textvariable=self.view, width=9, state="disabled")
        self.view_box.grid(row=0, column=1, padx=5)
        self.view_box.bind("<<ComboboxSelected>>", self.change_view)
        self.query_box = ttk.Combobox(controls, textvariable=self.query, width=12, state="disabled")
        self.query_box.grid(row=0, column=2, padx=10)
        self.query_box.bind("<<ComboboxSelected>>", lambda event: self.redraw())
        self._text(label(controls, size=9), "叠加强度").grid(row=0, column=3, padx=5)
        ttk.Scale(controls, from_=0, to=1, variable=self.opacity,
                  command=lambda value: self.redraw()).grid(row=0, column=4, columnspan=2,
                                                            sticky="ew", padx=12)
        actions = tk.Frame(self, bg=BG)
        actions.grid(row=3, column=0, sticky="ew", padx=18, pady=10)
        self.refresh_button = button(actions, "重新分析当前输入", lambda: self.request(self, "attention", None),
                                     padx=10, pady=6)
        self.refresh_button.pack(side="left")
        self.occlude_button = button(actions, "运行框选遮挡", self.run_occlusion, primary=True,
                                     state="disabled", padx=10, pady=6)
        self.occlude_button.pack(side="left", padx=8)
        self.clear_button = button(actions, "清除框选", self.clear_selection, padx=10, pady=6)
        self.clear_button.pack(side="left")
        self.export_button = button(actions, "导出 PNG", self.export, state="disabled", padx=10, pady=6)
        self.export_button.pack(side="left", padx=8)
        self.cancel_button = button(actions, "取消分析", self.cancel, state="disabled", padx=10, pady=6)
        self.cancel_button.pack(side="right")
        for widget, source in ((self.refresh_button, "重新分析当前输入"), (self.occlude_button, "运行框选遮挡"),
                               (self.clear_button, "清除框选"), (self.export_button, "导出 PNG"),
                               (self.cancel_button, "取消分析")):
            self._text(widget, source)

        pictures = tk.Frame(self, bg=BG)
        pictures.grid(row=4, column=0, sticky="nsew", padx=18)
        pictures.columnconfigure((0, 1), weight=1, uniform="pictures")
        pictures.rowconfigure(1, weight=1)
        self._text(label(pictures, size=10, bold=True), "原图 · 拖动框选遮挡区域").grid(row=0, column=0, sticky="w", pady=5)
        self.original = PictureCanvas(pictures, on_select=self.select_region, language=language)
        self.original.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.tabs = ttk.Notebook(pictures)
        self.tabs.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(6, 0))
        heat_tab, mask_tab = tk.Frame(self.tabs, bg=PANEL), tk.Frame(self.tabs, bg=PANEL)
        self.tabs.add(heat_tab, text=self.t("注意力热力图"))
        self.tabs.add(mask_tab, text=self.t("遮挡区域示意"))
        self.heat_canvas = PictureCanvas(heat_tab, language=language)
        self.heat_canvas.pack(fill="both", expand=True)
        self.mask_canvas = PictureCanvas(mask_tab, language=language)
        self.mask_canvas.pack(fill="both", expand=True)

        report = tk.Frame(self, bg=BG)
        report.grid(row=5, column=0, sticky="ew", padx=18, pady=(10, 4))
        report.columnconfigure(0, weight=1)
        self.details = tk.Text(report, height=10, wrap="word", bg=INPUT, fg=TEXT,
                               relief="flat", state="disabled", font=("Microsoft YaHei UI", 9))
        self.details.grid(row=0, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(report, orient="vertical", command=self.details.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.details.configure(yscrollcommand=scrollbar.set)
        self.status_label = label(self, textvariable=self.status, size=9, color=ACCENT, wraplength=1100,
                                  justify="left")
        self.status_label.grid(row=6, column=0, sticky="ew", padx=18, pady=(4, 12))

    def t(self, source, **values):
        return translate(source, self.language, **values)

    def _text(self, widget, source):
        self._texts.append((widget, source))
        widget.configure(text=self.t(source))
        return widget

    def set_status(self, source, **values):
        self._status_source, self._status_values = source, values
        self.status.set(self.t(source, **values))

    def set_language(self, language):
        if self.closed:
            return
        view, query = self.view_index, self.query_index
        self.language = language
        self.title(f"{PROJECT_SHORT_NAME} · {self.t('预测解释')}")
        for widget, source in self._texts:
            widget.configure(text=self.t(source))
        for index, source in enumerate(("注意力热力图", "遮挡区域示意")):
            self.tabs.tab(index, text=self.t(source))
        for canvas in (self.original, self.heat_canvas, self.mask_canvas):
            canvas.language = language
            canvas.redraw()
        if self.analysis is not None:
            self._set_choices(view, query)
            self.refresh_details()
        self.set_status(self._status_source, **self._status_values)

    def _set_choices(self, view=0, query=None):
        self.view_box.configure(values=[self.t("图片 {index}", index=i + 1)
                                        for i in range(len(self.analysis["pictures"]))])
        self.query_box.configure(values=[self.t("查询平均")] + [self.t("查询 {index}", index=i + 1)
                                         for i in range(self.analysis["maps"].shape[1])])
        self.view_box.current(view)
        self.query_box.current(0 if query is None else query + 1)

    def _wrap_note(self, event):
        if event.widget is self:
            self.note.configure(wraplength=max(300, event.width - 36))
            self.status_label.configure(wraplength=max(300, event.width - 36))

    @property
    def view_index(self):
        return max(0, self.view_box.current())

    @property
    def query_index(self):
        index = self.query_box.current()
        return None if index <= 0 else index - 1

    def set_busy(self, value):
        if self.closed:
            return
        self.working = value
        for widget in (self.refresh_button, self.clear_button):
            widget.configure(state="disabled" if value else "normal")
        for widget in (self.view_box, self.query_box):
            widget.configure(state="readonly" if self.analysis is not None and not value else "disabled")
        self.export_button.configure(state="normal" if self.analysis is not None and not value else "disabled")
        self.cancel_button.configure(state="normal" if value else "disabled")
        self.original.enabled = not value
        self.occlude_button.configure(state="normal" if self.rectangle is not None and not value else "disabled")

    def set_analysis(self, analysis):
        self.analysis, self.occlusion, self.rectangle = analysis, None, None
        self._set_choices()
        self._heat_key = None
        self.change_view()
        self.refresh_details()
        self.set_status("热力图已生成 · 当前图片：{path}", path=analysis["paths"][0])

    def refresh_details(self):
        if self.occlusion is not None:
            from streetgeo.inference.explain import comparison_text
            self.set_details(comparison_text(self.analysis, self.occlusion, self.language))
            return
        analysis = self.analysis
        item = analysis["candidates"][0]
        self.set_details("\n".join((self.t("模型：{path}", path=analysis["checkpoint"]),
            self.t("Top-1：{lat:.6f}, {lon:.6f} · 网格概率 {prob:.4%}", lat=item["latitude"], lon=item["longitude"], prob=item["cell_probability"]),
            self.t("输入：{size} × {size} · 注意力网格 {side} × {side}", size=analysis["image_size"], side=analysis["maps"].shape[-1]),
            self.t("四个查询没有预先规定的语义；不同查询的颜色不能直接比较绝对贡献。"),
            self.t("请在左侧原图拖动框选，再运行遮挡；其余视角会一起保留用于预测。"))))

    def set_details(self, text):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def change_view(self, event=None):
        self.clear_selection()
        self.redraw()

    def redraw(self):
        if self.analysis is None or self.closed:
            return
        from streetgeo.inference.explain import heatmap_image
        view, query = self.view_index, self.query_index
        key = (id(self.analysis), view, query)
        if key != self._heat_key:
            self._heat = heatmap_image(self.analysis, view, query, 1)
            self._heat.thumbnail((1400, 1000), Image.Resampling.LANCZOS)
            self._plain = self.analysis["pictures"][view].resize(self._heat.size, Image.Resampling.LANCZOS)
            self._heat_key = key
        self.original.set_picture(self._plain, self.rectangle)
        self.heat_canvas.set_picture(Image.blend(self._plain, self._heat, self.opacity.get()))

    def select_region(self, rectangle):
        if self.analysis is not None and not self.working:
            self.rectangle = rectangle
            self.original.rectangle = rectangle
            self.occlude_button.configure(state="normal")
            self.set_status("已框选新区域 · 点击“运行框选遮挡”；下方统计保留上次完成的实验")

    def clear_selection(self):
        self.rectangle = None
        self.original.rectangle = None
        self.original._draw_rectangle()
        self.occlude_button.configure(state="disabled")
        if self.analysis is not None:
            view = self.view_index
            self.mask_canvas.set_picture(self.occlusion["picture"] if self.occlusion is not None
                                         and self.occlusion["view"] == view else self.analysis["pictures"][view])

    def run_occlusion(self):
        if self.rectangle is not None and not self.working:
            self.request(self, "occlusion", (self.view_index, self.rectangle))

    def set_occlusion(self, result):
        self.occlusion = result
        self.rectangle = result["rectangle"]
        self.original.rectangle = self.rectangle
        self.original._draw_rectangle()
        self.mask_canvas.set_picture(result["picture"])
        self.refresh_details()
        self.tabs.select(1)
        self.set_status("遮挡对比完成 · 分数固定比较原候选；原文件和模型参数未改动")

    def export(self):
        if self.analysis is None or self.working:
            return
        path = filedialog.asksaveasfilename(parent=self, title=self.t("导出预测解释"), defaultextension=".png",
                                           initialfile="svgt_explanation.png", filetypes=[("PNG", "*.png")])
        if path:
            self.request(self, "export", (self.view_index, self.query_index, self.opacity.get(), path))

    def show_error(self, text):
        self.set_status("分析未完成：{error}", error=text)
        messagebox.showerror(PROJECT_SHORT_NAME, text, parent=self)

    def close(self):
        if self.closed:
            return
        self.destroy()

    def destroy(self):
        if self.closed:
            return
        self.closed = True
        if self.working:
            self.cancel()
        self.analysis = self.occlusion = self._plain = self._heat = None
        # Release Tcl variables/photos on the Tk thread, not a later worker GC.
        self.view = self.query = self.opacity = self.status = None
        self.request = self.cancel = None
        super().destroy()
