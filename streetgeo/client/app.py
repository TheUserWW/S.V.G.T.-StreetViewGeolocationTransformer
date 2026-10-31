"""Portable prediction-only client entry point, copied as app.py by the builder."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time
import tkinter as tk

from PIL import Image

from streetgeo.paths import ROOT
from streetgeo.ui.prediction import PredictionApp

if __package__:
    from .tk_runtime import configure_tk
else:
    from tkinter_runtime import configure_tk


def create_app(root):
    app = PredictionApp(root, settings_path=ROOT / ".runtime" / "settings.json",
                        default_checkpoint="best.pt", default_device="cpu")
    root.title("StreetGeo · 预测客户端")
    selected = Path(app.checkpoint.get()).expanduser()
    selected = selected if selected.is_absolute() else ROOT / selected
    if not selected.is_file():
        app.checkpoint.set("best.pt")
    # The distributed environment uses the CPU wheel and requires no GPU driver.
    app.device.set("cpu")
    app.device_box.configure(values=("cpu",))
    return app


def self_test():
    root = tk.Tk()
    root.withdraw()
    app = create_app(root)
    errors = []
    app.error = lambda error: errors.append(str(error))
    try:
        root.update_idletasks()
        app.manual.set("39.9042, 116.4074")
        app.use_manual()
        if not app.result or app.result["source"] != "manual":
            raise RuntimeError("Manual map input failed")
        with tempfile.TemporaryDirectory(prefix="self-test-", dir=ROOT / ".runtime") as folder:
            image = Path(folder) / "sample.png"
            Image.new("RGB", (64, 48), (110, 150, 180)).save(image)
            app.images = [str(image)]
            app.start_prediction()
            deadline = time.monotonic() + 90
            while app.busy and time.monotonic() < deadline:
                root.update()
                time.sleep(.02)
            if app.busy or not app.result or app.result["source"] != "model":
                raise RuntimeError("Model prediction did not finish: " + "; ".join(errors))
            candidate = app.result["candidates"][0]
            if not all(math.isfinite(candidate[k]) for k in ("latitude", "longitude")):
                raise RuntimeError("Nonfinite coordinates")
            print(json.dumps({"self_test": "passed", "device": app.result["device"],
                              "candidates": len(app.candidates), "checkpoint": app.checkpoint.get(),
                              "runtime_root": str(ROOT), "model_width": app.engine.model.config.n_embed}))
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    configure_tk()
    os.chdir(ROOT)
    (ROOT / ".runtime").mkdir(exist_ok=True)
    if args.self_test:
        self_test()
        return
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (OSError, AttributeError):
            pass
    root = tk.Tk()
    create_app(root)
    root.mainloop()


if __name__ == "__main__":
    main()
