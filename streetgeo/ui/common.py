"""Shared native widgets and colors for training and prediction windows."""
import tkinter as tk
from tkinter import ttk

BG = "#f3f5f8"
PANEL = "#ffffff"
NAV = "#142332"
TEXT = "#172b3a"
MUTED = "#6b7c8c"
BORDER = "#dce3ea"
ACCENT = "#087f75"
FONT = "Microsoft YaHei UI"


def label(parent, text="", *, size=10, color=TEXT, bold=False, **options):
    return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color,
                    font=(FONT, size, "bold" if bold else "normal"), anchor="w", **options)


def button(parent, text, command, *, primary=False, **options):
    spacing = {"padx": 16, "pady": 9}
    spacing.update(options)
    return tk.Button(parent, text=text, command=command, bg=ACCENT if primary else "#edf2f5",
                     fg="white" if primary else TEXT, activebackground="#06695f" if primary else "#dce8ed",
                     activeforeground="white" if primary else TEXT, relief="flat", bd=0,
                     font=(FONT, 10, "bold" if primary else "normal"),
                     cursor="hand2", disabledforeground="#91a0ac", **spacing)


class ScrollFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=PANEL)
        self.canvas = tk.Canvas(self, bg=PANEL, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.content = tk.Frame(self.canvas, bg=PANEL)
        window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(window, width=event.width))
        self.bind_all("<MouseWheel>", self._wheel, add="+")

    def _wheel(self, event):
        current = self.winfo_containing(event.x_root, event.y_root)
        while current is not None:
            if current == self:
                self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
                return "break"
            current = getattr(current, "master", None)


