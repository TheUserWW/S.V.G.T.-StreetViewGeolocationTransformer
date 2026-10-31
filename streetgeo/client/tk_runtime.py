"""Find Tcl/Tk scripts for the base Python, including exact-version fallbacks."""
import ctypes
import os
from pathlib import Path
import sys


def runtime_version():
    import _tkinter
    if os.name == "nt":
        for dll in Path(_tkinter.__file__).parent.glob("tcl*.dll"):
            try:
                library = ctypes.CDLL(str(dll))
                values = [ctypes.c_int() for _ in range(4)]
                library.Tcl_GetVersion(*(ctypes.byref(value) for value in values))
                return ".".join(str(value.value) for value in values[:3])
            except (OSError, AttributeError):
                continue
    return None


def configure_tk():
    import _tkinter
    version = runtime_version()
    bases = [Path(sys.base_prefix), Path(getattr(sys, "_base_executable", sys.executable)).parent]
    pairs = [(base / "tcl" / ("tcl" + _tkinter.TCL_VERSION),
              base / "tcl" / ("tk" + _tkinter.TK_VERSION)) for base in bases]
    if version:
        bundled = Path(__file__).resolve().parent / "vendor" / version
        pairs.append((bundled / ("tcl" + _tkinter.TCL_VERSION), bundled / ("tk" + _tkinter.TK_VERSION)))
    for tcl, tk in pairs:
        if (tcl / "init.tcl").is_file() and (tk / "tk.tcl").is_file():
            os.environ["TCL_LIBRARY"] = str(tcl)
            os.environ["TK_LIBRARY"] = str(tk)
            return str(tcl), str(tk)
    return None


def check_tk():
    import tkinter as tk
    configure_tk()
    try:
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.destroy()
    except tk.TclError as error:
        raise RuntimeError("Tcl/Tk 初始化失败。请使用 Python 安装器的 Modify/Repair，勾选 Tcl/Tk and IDLE；"
                           "无需重装模型依赖。原始错误：" + str(error)) from error


if __name__ == "__main__":
    check_tk()
    print("Tcl/Tk window initialization passed")
