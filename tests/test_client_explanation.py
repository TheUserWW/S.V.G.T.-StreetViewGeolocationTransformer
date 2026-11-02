from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from streetgeo.client.prediction import ClientApp
from streetgeo.core.checkpoints import save_checkpoint
from streetgeo.core.model import StreetGeoModel
from test_model import tiny_config


class ClientExplanationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name)
        self.checkpoint = self.path / "external.pt"
        save_checkpoint(StreetGeoModel(tiny_config()), self.checkpoint)
        self.images = [self.path / "wide.png", self.path / "tall.png"]
        Image.new("RGB", (160, 80), "orange").save(self.images[0])
        Image.new("RGB", (80, 160), "green").save(self.images[1])
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = ClientApp(self.root, settings_path=self.path / "settings.json",
                             checkpoint=self.checkpoint, language="en")
        self.messages = []
        self.app.error = self.messages.append
        self.app.set_images(self.images)

    def tearDown(self):
        if not self.app.closed:
            self.app.close()
        self.folder.cleanup()

    def wait(self):
        deadline = time.monotonic() + 15
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertFalse(self.app.busy)
        if not self.app.closed:
            self.root.update()

    def open(self):
        self.app.open_explanation()
        window = self.app.explanation_window
        window.withdraw()
        window.show_error = self.messages.append
        self.wait()
        self.assertIsNotNone(window.analysis, self.messages)
        return window

    def language(self, value):
        self.app.language_choice.set(value)
        self.app.change_language()

    def test_attention_occlusion_and_live_language_switch_preserve_results_and_selection(self):
        started, finish = threading.Event(), threading.Event()
        real = self.app.engine.predict

        def blocked(*args, **kwargs):
            started.set()
            finish.wait(5)
            return real(*args, **kwargs)

        with patch.object(self.app.engine, "predict", side_effect=blocked):
            self.app.open_explanation()
            window = self.app.explanation_window
            window.withdraw()
            window.show_error = self.messages.append
            self.assertTrue(started.wait(5))
            self.language("中文")
            self.assertEqual(window.refresh_button.cget("text"), "重新分析当前输入")
            self.assertEqual(str(self.app.predict_button.cget("state")), "disabled")
            self.language("English")
            finish.set()
            self.wait()
        self.assertEqual(window.refresh_button.cget("text"), "Analyze current inputs")
        self.assertIn("Heatmap ready", window.status.get())
        analysis = window.analysis
        model = self.app.engine.model
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        window.view_box.current(1)
        window.change_view()
        window.query_box.current(2)
        window.redraw()
        window.select_region((.2, .3, .6, .8))
        window.run_occlusion()
        self.wait()
        rectangle = window.rectangle
        occlusion = window.occlusion
        self.assertIn("Fixed original candidate", window.details.get("1.0", "end"))
        self.language("中文")
        self.assertIn("固定比较原候选", window.details.get("1.0", "end"))
        self.language("English")
        self.assertIs(window.analysis, analysis)
        self.assertIs(window.occlusion, occlusion)
        self.assertEqual(window.rectangle, rectangle)
        self.assertEqual(window.view_index, 1)
        self.assertEqual(window.query_index, 1)
        self.assertIs(self.app.engine.model, model)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        self.assertFalse(model.encoder.pool_attention._forward_hooks)
        self.assertFalse(self.messages)

    def test_english_export_dialog_metadata_and_small_window_layout(self):
        window = self.open()
        window.select_region((.1, .1, .5, .5))
        window.run_occlusion()
        self.wait()
        window.deiconify()
        for geometry in ("920x720", "1400x900"):
            window.geometry(geometry)
            self.root.update()
            self.assertLessEqual(window.cancel_button.winfo_rootx() + window.cancel_button.winfo_width(),
                                 window.winfo_rootx() + window.winfo_width())
            self.assertGreater(window.original.winfo_width(), 200)
        output = self.path / "explanation.png"
        with patch("streetgeo.ui.explanation.filedialog.asksaveasfilename", return_value=str(output)) as dialog:
            window.export()
        self.wait()
        self.assertEqual(dialog.call_args.kwargs["title"], "Export prediction explanation")
        self.assertIn("Exported:", window.status.get())
        with Image.open(output) as picture:
            self.assertEqual(picture.width, 1500)
            self.assertTrue(picture.info["Description"].startswith("Pooling attention"))
        self.assertFalse(self.messages)

    def test_stale_input_rejected_with_localized_error_and_log(self):
        window = self.open()
        self.app.engine.key = None
        window.select_region((.2, .2, .8, .8))
        window.run_occlusion()
        self.wait()
        self.assertIn("Analyze current inputs again", self.messages[0])
        self.assertTrue((self.path / "prediction.log").is_file())
        self.assertIsNone(window.occlusion)
        self.assertFalse(self.app.explaining)

    def test_parent_close_cancels_analysis_and_waits_for_current_forward(self):
        started, finish = threading.Event(), threading.Event()
        real = self.app.engine.predict

        def blocked(*args, **kwargs):
            started.set()
            finish.wait(5)
            return real(*args, **kwargs)

        with patch.object(self.app.engine, "predict", side_effect=blocked):
            self.app.open_explanation()
            window = self.app.explanation_window
            window.withdraw()
            self.assertTrue(started.wait(5))
            self.app.close()
            self.assertFalse(self.app.closed)
            self.assertTrue(self.app.explanation_stop.is_set())
            finish.set()
            self.wait()
        self.assertTrue(self.app.closed)
        self.assertTrue(window.closed)
        self.assertFalse(self.messages)


if __name__ == "__main__":
    unittest.main()
