"""Clipboard images and dependency-free Windows Explorer file drops."""
import os
from pathlib import Path
import queue

from PIL import Image, ImageGrab


def clipboard_content():
    """Return an image, file list, or None; never read the clipboard on a timer."""
    content = ImageGrab.grabclipboard()
    if isinstance(content, Image.Image):
        return content.convert("RGB")
    if isinstance(content, list):
        return [str(Path(p).resolve()) for p in content if Path(p).is_file()]
    return None


class FileDropTarget:
    """Register the owning Tk window for WM_DROPFILES; callbacks run via Tk."""
    def __init__(self, widget, callback):
        self.widget, self.callback = widget, callback
        self.enabled = False
        self._pending = queue.SimpleQueue()
        self._poll_id = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.shell = ctypes.WinDLL("shell32", use_last_error=True)
        self.comctl = ctypes.WinDLL("comctl32", use_last_error=True)
        ptr = ctypes.c_size_t
        proc_type = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                                       wintypes.WPARAM, wintypes.LPARAM, ptr, ptr)
        self.comctl.SetWindowSubclass.argtypes = [wintypes.HWND, proc_type, ptr, ptr]
        self.comctl.SetWindowSubclass.restype = wintypes.BOOL
        self.comctl.RemoveWindowSubclass.argtypes = [wintypes.HWND, proc_type, ptr]
        self.comctl.DefSubclassProc.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        self.comctl.DefSubclassProc.restype = ctypes.c_ssize_t
        self.shell.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]
        self.shell.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
        self.shell.DragQueryFileW.restype = wintypes.UINT
        self.shell.DragFinish.argtypes = [wintypes.HANDLE]
        self.user.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        self.user.GetAncestor.restype = wintypes.HWND
        self.user.IsWindow.argtypes = [wintypes.HWND]
        self.widget.update_idletasks()
        self.hwnd = self.user.GetAncestor(widget.winfo_id(), 2) or widget.winfo_id()
        self.identifier = id(self)

        def procedure(hwnd, message, wparam, lparam, subclass_id, reference):
            if message == 0x0233:
                paths = []
                try:
                    count = self.shell.DragQueryFileW(wparam, 0xFFFFFFFF, None, 0)
                    for index in range(count):
                        length = self.shell.DragQueryFileW(wparam, index, None, 0)
                        buffer = ctypes.create_unicode_buffer(length + 1)
                        self.shell.DragQueryFileW(wparam, index, buffer, length + 1)
                        paths.append(buffer.value)
                finally:
                    self.shell.DragFinish(wparam)
                if paths and self.enabled:
                    self._pending.put(paths)
                return 0
            return self.comctl.DefSubclassProc(hwnd, message, wparam, lparam)

        self.procedure = proc_type(procedure)
        self.enabled = bool(self.comctl.SetWindowSubclass(self.hwnd, self.procedure, self.identifier, 0))
        if self.enabled:
            self.shell.DragAcceptFiles(self.hwnd, True)
            self._poll_id = self.widget.after(80, self._poll)

    def _poll(self):
        self._poll_id = None
        while not self._pending.empty():
            self.callback(self._pending.get_nowait())
        if self.enabled:
            self._poll_id = self.widget.after(80, self._poll)

    def close(self):
        if self._poll_id is not None:
            self.widget.after_cancel(self._poll_id)
            self._poll_id = None
        if self.enabled:
            self.shell.DragAcceptFiles(self.hwnd, False)
            self.comctl.RemoveWindowSubclass(self.hwnd, self.procedure, self.identifier)
            self.enabled = False
