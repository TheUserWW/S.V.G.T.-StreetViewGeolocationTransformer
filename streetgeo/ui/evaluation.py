"""Chart tabs and detailed statistics for the developer prediction app."""

import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from streetgeo import PROJECT_SHORT_NAME
from streetgeo.inference.evaluate import format_evaluation_report
from streetgeo.inference.evaluation_charts import (CHART_VIEWS, build_evaluation_figure,
                                                 export_evaluation_chart)
from streetgeo.ui.common import BG, FONT, INPUT, MUTED, TEXT, button, label


class ChartViewport(tk.Frame):
    """Keep a readable chart aspect ratio independently of the window shape."""

    def __init__(self, parent, figure):
        super().__init__(parent, bg=BG)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.figure = figure
        self.aspect = figure.get_figwidth() / figure.get_figheight()
        self._resize_id = None
        self._content_size = None
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.viewport = tk.Canvas(self, bg=BG, bd=0, highlightthickness=0,
                                  xscrollincrement=28, yscrollincrement=28)
        self.viewport.grid(row=0, column=0, sticky="nsew")
        self.vertical = ttk.Scrollbar(self, orient="vertical", command=self.viewport.yview)
        self.vertical.grid(row=0, column=1, sticky="ns")
        self.horizontal = ttk.Scrollbar(self, orient="horizontal", command=self.viewport.xview)
        self.horizontal.grid(row=1, column=0, sticky="ew")
        self.viewport.configure(xscrollcommand=self.horizontal.set, yscrollcommand=self.vertical.set)
        self.content = tk.Frame(self.viewport, bg=BG)
        self.content.pack_propagate(False)
        self._content_window = self.viewport.create_window(0, 0, window=self.content, anchor="nw")
        self.chart = FigureCanvasTkAgg(figure, master=self.content)
        self.chart.get_tk_widget().pack(fill="both", expand=True)
        self.chart.get_tk_widget().bind("<MouseWheel>", self._wheel, add="+")
        # Matplotlib updates its DPI on Map. Schedule after that update so a
        # monitor/scaling change uses the same physical font and chart scale.
        self.chart.get_tk_widget().bind("<Map>", self._schedule_resize, add="+")
        self.viewport.bind("<Configure>", self._schedule_resize)
        self.bind("<Map>", self._schedule_resize)
        self._schedule_resize()

    def _schedule_resize(self, event=None):
        if self._resize_id is None:
            self._resize_id = self.after_idle(self._resize_chart)

    def _resize_chart(self):
        self._resize_id = None
        width, height = self.viewport.winfo_width(), self.viewport.winfo_height()
        if width <= 1 or height <= 1:
            return
        # 11 inches at the live figure DPI keeps fixed-size labels readable.
        # Never squeeze a 2x2 report into an arbitrarily short canvas.
        chart_width = max(width, math.ceil(11 * self.figure.dpi))
        chart_height = round(chart_width / self.aspect)
        size = (chart_width, chart_height)
        if size != self._content_size:
            self._content_size = size
            self.content.configure(width=chart_width, height=chart_height)
            self.viewport.itemconfigure(self._content_window, width=chart_width, height=chart_height)
        self.viewport.configure(scrollregion=(0, 0, chart_width, max(height, chart_height)))

    def _wheel(self, event):
        if self.chart.toolbar and self.chart.toolbar.mode:
            return None
        if not event.delta:
            return "break"
        steps = -int(event.delta / 120) if abs(event.delta) >= 120 else (-1 if event.delta > 0 else 1)
        if event.state & 1:
            self.viewport.xview_scroll(steps, "units")
        else:
            self.viewport.yview_scroll(steps, "units")
        return "break"

    def destroy(self):
        if self._resize_id is not None:
            self.after_cancel(self._resize_id)
            self._resize_id = None
        for name in ("_idle_draw_id", "_event_loop_id"):
            callback = getattr(self.chart, name, None)
            if callback:
                self.chart.get_tk_widget().after_cancel(callback)
                setattr(self.chart, name, None)
        super().destroy()


class EvaluationReportWindow(tk.Toplevel):
    def __init__(self, parent, report):
        super().__init__(parent)
        self.report = report
        self.figures, self.canvases, self.frames = {}, {}, {}
        self.viewports, self.toolbars = {}, {}
        self._stacked_header = None
        self.title(f"{PROJECT_SHORT_NAME} · test 可视化统计")
        width, height = min(1400, int(self.winfo_screenwidth() * .9)), min(960, int(self.winfo_screenheight() * .88))
        self.geometry(f"{width}x{height}")
        self.minsize(900, 650)
        self.configure(bg=BG)
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)
        self.actions = tk.Frame(self, bg=BG)
        self.actions.grid(row=0, column=0, sticky="ew", padx=14, pady=12)
        self.actions.columnconfigure(0, weight=1)
        self.heading = label(self.actions, "评估统计 · 图表与详细数据", bold=True)
        self.heading.grid(row=0, column=0, sticky="w")
        self.action_buttons = tk.Frame(self.actions, bg=BG)
        self.action_buttons.grid(row=0, column=1, sticky="e")
        self.export_button = button(self.action_buttons, "导出当前图表", self.export_chart, padx=10, pady=6)
        self.export_button.pack(side="left", padx=(0, 8))
        button(self.action_buttons, "打开报告目录", lambda: os.startfile(report["output_directory"]),
               padx=10, pady=6).pack(side="left")
        self.note = label(self, "图表仅统计成功样本；部分结果会明确标注。图表保持比例，窗口较小时可滚动查看；Shift＋滚轮横向滚动。",
                          size=9, color=MUTED, justify="left", wraplength=900)
        self.note.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 9))
        self.actions.bind("<Configure>", self._reflow_header)
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))
        for view, title in CHART_VIEWS:
            frame = tk.Frame(self.notebook, bg=BG)
            self.frames[view] = frame
            self.notebook.add(frame, text=title)
        details = tk.Frame(self.notebook, bg=BG)
        self.frames["details"] = details
        self.notebook.add(details, text="详细统计")
        text = ScrolledText(details, bg=INPUT, fg=TEXT, insertbackground=TEXT, wrap="word",
                            font=(FONT, 10), padx=14, pady=12, bd=0)
        text.pack(fill="both", expand=True)
        text.insert("1.0", format_evaluation_report(report))
        text.configure(state="disabled")
        self.notebook.bind("<<NotebookTabChanged>>", self._select_tab)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._select_tab()

    def _reflow_header(self, event):
        needed = self.heading.winfo_reqwidth() + self.action_buttons.winfo_reqwidth() + 24
        stacked = event.width < needed
        if stacked != self._stacked_header:
            self._stacked_header = stacked
            self.action_buttons.grid_configure(row=1 if stacked else 0, column=0 if stacked else 1,
                                                columnspan=2 if stacked else 1, sticky="w" if stacked else "e",
                                                pady=(8, 0) if stacked else 0)
        self.note.configure(wraplength=max(200, event.width - 4))

    def current_view(self):
        selected = self.notebook.select()
        return next((view for view, frame in self.frames.items() if str(frame) == selected), "details")

    def _select_tab(self, event=None):
        view = self.current_view()
        self.export_button.configure(state="disabled" if view == "details" else "normal")
        if view == "details" or view in self.canvases:
            return
        frame = self.frames[view]
        try:
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

            class ReportToolbar(NavigationToolbar2Tk):
                # Keep report layout fixed; zoom/pan modify data limits only.
                toolitems = tuple(item for item in NavigationToolbar2Tk.toolitems if item[0] != "Subplots")

                def save_figure(toolbar, *args):
                    self.export_chart(view)

            figure = build_evaluation_figure(self.report, view)
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            viewport = ChartViewport(frame, figure)
            viewport.grid(row=0, column=0, sticky="nsew")
            canvas = viewport.chart
            self.figures[view], self.canvases[view] = figure, canvas
            self.viewports[view] = viewport
            toolbar = ReportToolbar(canvas, frame, pack_toolbar=False)
            toolbar.update()
            toolbar.grid(row=1, column=0, sticky="ew")
            self.toolbars[view] = toolbar
            canvas.draw_idle()
        except Exception as error:
            label(frame, f"图表暂时无法显示：{error}\n数值统计仍可在“详细统计”页查看。",
                  wraplength=700, justify="left").grid(row=2, column=0, padx=20, pady=30)
            self.export_button.configure(state="disabled")

    def export_chart(self, view=None):
        view = view or self.current_view()
        figure = self.figures.get(view)
        if figure is None:
            return
        path = filedialog.asksaveasfilename(parent=self, title="导出统计图", defaultextension=".png",
                                          initialdir=self.report["output_directory"],
                                          initialfile=f"statistics_{view}.png",
                                          filetypes=[("PNG 图片", "*.png"), ("SVG 矢量图", "*.svg")])
        if path:
            try:
                limits = [(axis.get_xlim(), axis.get_ylim()) for axis in figure.axes]
                export_evaluation_chart(self.report, view, path, limits=limits)
            except (OSError, ValueError) as error:
                messagebox.showerror(PROJECT_SHORT_NAME, str(error), parent=self)

    def destroy(self):
        for viewport in self.viewports.values():
            viewport.destroy()
        self.viewports.clear()
        for canvas in self.canvases.values():
            # Cancel deferred draws before their Tk widgets disappear.
            if canvas._idle_draw_id:
                canvas.get_tk_widget().after_cancel(canvas._idle_draw_id)
                canvas._idle_draw_id = None
        for figure in self.figures.values():
            figure.clear()
        self.figures.clear()
        self.canvases.clear()
        self.toolbars.clear()
        super().destroy()
