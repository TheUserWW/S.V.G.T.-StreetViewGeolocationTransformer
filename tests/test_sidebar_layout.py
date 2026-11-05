import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest

from streetgeo.client.prediction import ClientApp
from streetgeo.client.workbench import DailyApp
from streetgeo.ui.prediction import PredictionApp
from streetgeo.ui.training import TrainingApp


class SidebarLayoutTests(unittest.TestCase):
    def close_app(self, root, app):
        root.update_idletasks()
        if isinstance(app, TrainingApp):
            for timer in root.tk.call("after", "info"):
                script = root.tk.call("after", "info", timer)[0]
                if str(script).endswith(("_poll", "_probe")):
                    root.after_cancel(timer)
            app._close()
        else:
            app.close()

    def test_drag_resize_and_restore_in_every_workbench(self):
        for app_type in (TrainingApp, DailyApp, PredictionApp, ClientApp):
            with self.subTest(app=app_type.__name__), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "settings.json"
                options = {"probe": False} if app_type is TrainingApp else {}
                root = tk.Tk()
                app = app_type(root, settings_path=path, **options)
                try:
                    root.geometry("1460x900+40+40")
                    root.update()
                    layout = app.sidebar_layout
                    original = layout.sash_coord(0)[0]
                    target = original + 96
                    layout.event_generate("<ButtonPress-1>", x=original + 2, y=80)
                    root.update_idletasks()
                    layout.event_generate("<B1-Motion>", x=target + 2, y=80)
                    root.update_idletasks()
                    layout.event_generate("<ButtonRelease-1>", x=target + 2, y=80)
                    root.update()
                    self.assertAlmostEqual(layout.sash_coord(0)[0], target, delta=3)
                    saved_width = int(json.loads(path.read_text(encoding="utf-8"))["sidebar_width"])
                    self.assertAlmostEqual(saved_width, target, delta=3)
                    root.geometry("1000x780")
                    root.update()
                    self.assertGreaterEqual(layout.sidebar.winfo_width(), layout.min_sidebar)
                    self.assertGreaterEqual(layout.content.winfo_width(), layout.min_content)
                    root.geometry("1460x900")
                    root.update()
                    self.assertAlmostEqual(layout.sash_coord(0)[0], saved_width, delta=1)
                finally:
                    self.close_app(root, app)
                root = tk.Tk()
                root.withdraw()
                reopened = app_type(root, settings_path=path, **options)
                try:
                    self.assertEqual(reopened.sidebar_layout.sidebar_width, saved_width)
                finally:
                    self.close_app(root, reopened)

    def test_old_or_invalid_sidebar_preferences_still_open(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps({"sidebar_width": "broken", "language": "en", "checkpoint": "local model.pt"}))
            root = tk.Tk()
            root.withdraw()
            app = DailyApp(root, settings_path=path)
            try:
                self.assertEqual(app.sidebar_layout.sidebar_width, 350)
                app._save_settings()
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["language"], "en")
                self.assertEqual(saved["checkpoint"], "local model.pt")
                self.assertEqual(saved["sidebar_width"], 350)
                self.assertIn("auto_predict", saved)
            finally:
                app.close()


if __name__ == "__main__":
    unittest.main()
