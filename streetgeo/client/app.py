"""Portable bilingual prediction/explanation client entry point."""
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

if __package__:
    from .tk_runtime import configure_tk
    from .workbench import DailyApp as ClientApp
else:
    from tkinter_runtime import configure_tk
    from daily_ui import DailyApp as ClientApp


def create_app(root, checkpoint=None):
    return ClientApp(root, checkpoint=checkpoint)


def self_test(checkpoint=None):
    with tempfile.TemporaryDirectory(prefix="self-test-", dir=ROOT / ".runtime") as folder:
        _self_test_in_folder(Path(folder), checkpoint)


def _self_test_in_folder(folder, checkpoint=None):
    root = tk.Tk()
    root.withdraw()
    app = ClientApp(root, settings_path=folder / "settings.json", checkpoint=checkpoint)
    errors = []
    app.error = lambda error: errors.append(str(error))
    try:
        root.update_idletasks()
        if checkpoint is None:
            for language in ("English", "中文"):
                app.language_choice.set(language)
                app.change_language()
            print(json.dumps({"self_test": "passed", "prediction": "not_run", "model_required": True,
                              "languages": ["zh", "en"], "runtime_root": str(ROOT)}))
            return
        image = Path(folder) / "sample.png"
        Image.new("RGB", (64, 48), (110, 150, 180)).save(image)
        if not app.set_images([image]):
            raise RuntimeError("Street-view image import failed")
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
        original_coordinates = candidate["coordinates"]
        for language in ("English", "中文"):
            app.language_choice.set(language)
            app.change_language()
            root.update_idletasks()
            if app.result["candidates"][0]["coordinates"] != original_coordinates:
                raise RuntimeError("Language switching changed the prediction")
        if app.map.candidates != app.candidates:
            raise RuntimeError("Predictions were not placed on the map")
        app.open_explanation()
        window = app.explanation_window
        window.withdraw()
        window.show_error = lambda error: errors.append(str(error))
        deadline = time.monotonic() + 90
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        if app.busy or window.analysis is None:
            raise RuntimeError("Attention analysis did not finish: " + "; ".join(errors))
        window.select_region((.25, .25, .75, .75))
        window.run_occlusion()
        deadline = time.monotonic() + 90
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        if app.busy or window.occlusion is None:
            raise RuntimeError("Occlusion analysis did not finish: " + "; ".join(errors))
        app.language_choice.set("English")
        app.change_language()
        if "Fixed original candidate" not in window.details.get("1.0", "end"):
            raise RuntimeError("Explanation translation did not update")
        exported = str(Path(folder) / "explanation.png")
        app.start_explanation_job(window, "export", (0, None, .5, exported))
        deadline = time.monotonic() + 90
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        if app.busy or not Path(exported).is_file():
            raise RuntimeError("Explanation PNG export did not finish")
        with Image.open(exported) as picture:
            if not picture.info.get("Description", "").startswith("Pooling attention"):
                raise RuntimeError("Explanation export language did not update")
        print(json.dumps({"self_test": "passed", "device": app.result["device"],
                          "candidates": len(app.candidates), "checkpoint": str(app.checkpoint_path),
                          "languages": ["zh", "en"],
                          "runtime_root": str(ROOT), "model_width": app.engine.model.config.n_embed,
                          "explanation": "passed"}))
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", help="external S.V.G.T. model file")
    args = parser.parse_args()
    configure_tk()
    os.chdir(ROOT)
    (ROOT / ".runtime").mkdir(exist_ok=True)
    if args.self_test:
        self_test(args.checkpoint)
        return
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (OSError, AttributeError):
            pass
    root = tk.Tk()
    create_app(root, args.checkpoint)
    root.mainloop()


if __name__ == "__main__":
    main()
