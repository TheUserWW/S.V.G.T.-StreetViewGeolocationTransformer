"""Shared dark theme and native widgets for training and prediction windows."""
import sys
import tkinter as tk
from tkinter import ttk

BG = "#10161e"
PANEL = "#18222e"
SURFACE = "#202d3b"
INPUT = "#121c27"
NAV = "#0c121b"
TEXT = "#e3eaf2"
MUTED = "#a1b0c1"
DISABLED = "#7c8b9d"
BORDER = "#334255"
ACCENT = "#65d6c2"
PRIMARY = "#146f66"
HOVER = "#2c3d50"
SELECTION = "#245b61"
WARNING = "#efbb73"
CHART_FILL = "#183b3a"
MAP_OCEAN = "#101d2a"
MAP_LAND = "#27394a"
MAP_BORDER = "#8aa0b5"
MAP_DASH = "#71869c"
MAP_LABEL = "#b7c7d6"
FONT = "Microsoft YaHei UI"


def apply_dark_theme(root):
    """Style classic/ttk widgets, including combobox popups and disabled states."""
    root.configure(bg=BG)
    for pattern, value in {
        "*background": PANEL, "*foreground": TEXT,
        "*insertBackground": TEXT, "*selectBackground": SELECTION,
        "*selectForeground": TEXT, "*disabledForeground": DISABLED,
        "*TCombobox*Listbox.background": INPUT,
        "*TCombobox*Listbox.foreground": TEXT,
        "*TCombobox*Listbox.selectBackground": SELECTION,
        "*TCombobox*Listbox.selectForeground": TEXT,
    }.items():
        root.option_add(pattern, value)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=PANEL, foreground=TEXT, bordercolor=BORDER,
                    darkcolor=BORDER, lightcolor=BORDER, troughcolor=INPUT,
                    focuscolor=ACCENT, font=(FONT, 10))
    style.configure("TEntry", fieldbackground=INPUT, foreground=TEXT,
                    insertcolor=TEXT, padding=6, selectbackground=SELECTION,
                    selectforeground=TEXT)
    style.map("TEntry", fieldbackground=[("disabled", SURFACE), ("readonly", INPUT)],
              foreground=[("disabled", DISABLED)],
              bordercolor=[("focus", ACCENT)])
    style.configure("TCombobox", padding=6, fieldbackground=INPUT, background=SURFACE,
                    foreground=TEXT, arrowcolor=TEXT, insertcolor=TEXT,
                    selectbackground=SELECTION, selectforeground=TEXT)
    style.map("TCombobox", fieldbackground=[("disabled", SURFACE), ("readonly", INPUT)],
              foreground=[("disabled", DISABLED), ("readonly", TEXT)],
              background=[("active", HOVER)], arrowcolor=[("disabled", DISABLED)],
              bordercolor=[("focus", ACCENT)])
    style.configure("TCheckbutton", background=PANEL, foreground=TEXT,
                    indicatorbackground=INPUT, indicatorforeground=TEXT, font=(FONT, 9))
    style.map("TCheckbutton", background=[("active", PANEL)],
              foreground=[("disabled", DISABLED)],
              indicatorbackground=[("disabled", SURFACE), ("selected", PRIMARY), ("active", HOVER)])
    style.configure("Treeview", background=INPUT, fieldbackground=INPUT, foreground=TEXT,
                    rowheight=26, font=(FONT, 9))
    style.map("Treeview", background=[("selected", SELECTION)], foreground=[("selected", TEXT)])
    style.configure("Treeview.Heading", background=SURFACE, foreground=TEXT,
                    relief="flat", font=(FONT, 9))
    style.map("Treeview.Heading", background=[("active", HOVER)])
    style.configure("Horizontal.TProgressbar", background=PRIMARY, troughcolor=INPUT,
                    lightcolor=PRIMARY, darkcolor=PRIMARY, borderwidth=0)
    for orient in ("Vertical", "Horizontal"):
        name = f"{orient}.TScrollbar"
        style.configure(name, background=SURFACE, troughcolor=INPUT, arrowcolor=MUTED,
                        borderwidth=0, arrowsize=13)
        style.map(name, background=[("pressed", SELECTION), ("active", HOVER)],
                  arrowcolor=[("disabled", DISABLED)])
    root.after_idle(lambda: _dark_title_bar(root))


def _dark_title_bar(root):
    """Request the Windows dark title bar when supported; other systems use Tk."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        parent = ctypes.windll.user32.GetParent
        parent.argtypes, parent.restype = [wintypes.HWND], wintypes.HWND
        handle = parent(root.winfo_id())
        setter = ctypes.windll.dwmapi.DwmSetWindowAttribute
        setter.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        setter.restype = ctypes.c_long
        enabled = ctypes.c_int(1)
        for attribute in (20, 19):
            if setter(handle, attribute, ctypes.byref(enabled), ctypes.sizeof(enabled)) == 0:
                break
    except (AttributeError, OSError, tk.TclError):
        pass


def label(parent, text="", *, size=10, color=TEXT, bold=False, **options):
    return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color,
                    font=(FONT, size, "bold" if bold else "normal"), anchor="w", **options)


def button(parent, text, command, *, primary=False, **options):
    spacing = {"padx": 16, "pady": 9}
    spacing.update(options)
    return tk.Button(parent, text=text, command=command, bg=PRIMARY if primary else SURFACE,
                     fg="white" if primary else TEXT, activebackground="#197d72" if primary else HOVER,
                     activeforeground="white" if primary else TEXT, relief="flat", bd=0,
                     font=(FONT, 10, "bold" if primary else "normal"),
                     cursor="hand2", disabledforeground="#c0cbd4" if primary else DISABLED,
                     highlightthickness=0, **spacing)


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


