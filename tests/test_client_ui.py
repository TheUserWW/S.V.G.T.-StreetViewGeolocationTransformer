import json
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest
import sys
from unittest.mock import patch

from PIL import Image

from streetgeo.client.i18n import LANGUAGES, read_language
from streetgeo.client.prediction import ClientApp


class PausedEngine:
    def __init__(self):
        self.release = threading.Event()
        self.called = None
        self.options = None

    def predict(self, checkpoint, images, **options):
        self.called = (checkpoint, images, options["device"])
        self.options = options
        options["progress"]("正在加载模型…")
        if not self.release.wait(5):
            raise RuntimeError("test worker timed out")
        return {"source": "model", "device": "cpu", "elapsed_seconds": .1,
                "candidates": [{"latitude": 35.0+i, "longitude": 110.0+i,
                                "coordinates": f"{35+i}, {110+i}"} for i in range(5)]}


class CompactClientTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.image = self.folder / "street.png"
        Image.new("RGB", (100, 60), (60, 130, 180)).save(self.image)
        self.checkpoint = self.folder / "best.pt"
        self.checkpoint.touch()
        self.root = tk.Tk()
        self.root.withdraw()
        self.engine = PausedEngine()
        self.app = ClientApp(self.root, engine=self.engine, checkpoint=self.checkpoint,
                             settings_path=self.folder / "settings.json", language="zh")
        self.errors = []
        self.app.error = self.errors.append
        self.root.update_idletasks()

    def tearDown(self):
        self.engine.release.set()
        deadline = time.monotonic()+6
        while self.app.busy and time.monotonic()<deadline:
            self.root.update()
            time.sleep(.02)
        if not self.app.closed:
            self.app.close()
        self.temporary.cleanup()

    def wait_prediction(self):
        deadline = time.monotonic()+5
        while self.app.busy and time.monotonic()<deadline:
            self.root.update()
            time.sleep(.02)
        self.assertFalse(self.app.busy)

    def test_language_switch_during_prediction_and_map_result(self):
        self.assertTrue(self.app.set_images([self.image]))
        self.app.start_prediction()
        self.assertEqual(str(self.app.choose_button.cget("state")), "disabled")
        self.app.language_choice.set("English")
        self.app.change_language()
        self.assertEqual(self.app.predict_button.cget("text"), "Predict location")
        self.engine.release.set()
        self.wait_prediction()
        self.assertEqual(self.engine.called[2], "cpu")
        self.assertIn("Prediction complete", self.app.status.get())
        self.assertEqual(len(self.app.map.candidates), 5)
        self.app.candidate_list.selection_clear(0, "end")
        self.app.candidate_list.selection_set(2)
        self.app.select_candidate()
        original_center = self.app.map.center
        self.app.language_choice.set("中文")
        self.app.change_language()
        self.assertEqual(self.app.map.center, original_center)
        self.assertEqual(self.app.selected, 2)
        self.assertEqual(self.app.predict_button.cget("text"), "预测位置")
        self.assertIn("预测完成", self.app.status.get())
        self.assertEqual(read_language(self.folder / "settings.json"), "zh")
        self.app.language_choice.set("English")
        self.app.change_language()
        self.assertEqual(read_language(self.folder / "settings.json"), "en")
        self.app.map.configure(width=750, height=500)
        self.app.map.update_idletasks()
        self.assertIn("Boundaries", self.app.map.translate("边界：UN Geospatial / OCHA · 离线概览"))
        self.assertTrue(self.app.set_images([self.image]))
        self.assertIsNone(self.app.result)
        self.assertEqual(self.app.map.candidates, [])

    def test_only_prediction_controls_and_invalid_inputs(self):
        for removed in ("manual", "test_sampler", "screen", "device_box", "point_table"):
            self.assertFalse(hasattr(self.app, removed), removed)
        self.assertEqual(str(self.app.predict_button.cget("state")), "disabled")
        self.assertFalse(self.app.set_images([self.image]*5))
        self.assertEqual(self.errors.pop(), "too_many")
        self.assertFalse(self.app.set_images([self.folder / "missing.jpg"]))
        self.assertEqual(self.errors.pop(), "invalid_image")
        self.assertTrue(self.app.set_images([self.image]))
        self.checkpoint.unlink()
        self.app.start_prediction()
        self.assertEqual(self.errors.pop(), "missing_model")
        self.assertFalse(self.app.busy)

    def test_coverage_toggle_persists_translates_and_reaches_worker(self):
        self.assertFalse(self.app.google_coverage_only.get())
        self.app.set_images([self.image])
        self.app.coverage_check.invoke()
        self.assertTrue(self.app.google_coverage_only.get())
        self.assertTrue(json.loads(self.app.settings_path.read_text(encoding="utf-8"))["google_coverage_only"])
        self.app.start_prediction()
        self.assertEqual(str(self.app.coverage_check.cget("state")), "disabled")
        self.app.language_choice.set("English")
        self.app.change_language()
        self.assertIn("Google-covered", self.app.coverage_check.cget("text"))
        self.engine.release.set()
        self.wait_prediction()
        self.assertTrue(self.engine.options["google_coverage_only"])
        self.assertIsNotNone(self.app.result)
        self.app.coverage_check.invoke()
        self.assertIsNone(self.app.result)
        self.assertEqual(self.app.map.candidates, [])
        self.app.coverage_check.invoke()
        self.app.close()
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = ClientApp(self.root, engine=self.engine, settings_path=self.folder / "settings.json")
        self.root.update_idletasks()
        self.assertTrue(self.app.google_coverage_only.get())
        self.assertEqual(self.app.checkpoint_path, self.checkpoint)

    def test_worker_import_failure_and_empty_domain_release_busy_with_localized_errors(self):
        from streetgeo.core import EmptyPredictionDomainError
        self.app.set_images([self.image])
        for error, key in ((ImportError("torch is unavailable"), "failed"),
                           (EmptyPredictionDomainError("empty"), "coverage_empty")):
            with patch.object(self.engine, "predict", side_effect=error), patch.dict(sys.modules, {"torch": None}):
                self.app.start_prediction()
                self.wait_prediction()
            self.assertEqual(self.errors.pop(), key)

    def test_model_picker_persists_path_and_clears_old_results(self):
        other = self.folder / "another model.pth"
        other.touch()
        self.app.set_images([self.image])
        self.app.start_prediction()
        self.assertEqual(str(self.app.model_button.cget("state")), "disabled")
        self.engine.release.set()
        self.wait_prediction()
        self.assertIsNotNone(self.app.result)
        with patch("streetgeo.client.prediction.filedialog.askopenfilename", return_value=str(other)):
            self.app.choose_checkpoint()
        self.assertEqual(self.app.checkpoint_path, other)
        self.assertIsNone(self.app.result)
        self.assertEqual(self.app.map.candidates, [])
        self.app.language_choice.set("English")
        self.app.change_language()
        self.assertEqual(self.app.model_button.cget("text"), "Choose model file")
        self.assertEqual(json.loads(self.app.settings_path.read_text(encoding="utf-8"))["checkpoint"], str(other))
        self.app.start_prediction()
        self.wait_prediction()
        self.assertEqual(self.engine.called[0], other)
        self.app.close()
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = ClientApp(self.root, settings_path=self.folder / "settings.json", engine=self.engine)
        self.root.update_idletasks()
        self.assertEqual(self.app.checkpoint_path, other)
        self.assertEqual(self.app.language, "en")

    def test_no_model_selected_still_opens_and_can_choose_later(self):
        self.app.checkpoint.set("")
        self.assertIsNone(self.app.checkpoint_path)
        self.app.set_images([self.image])
        self.assertEqual(str(self.app.predict_button.cget("state")), "disabled")
        self.app.start_prediction()
        self.assertEqual(self.errors.pop(), "missing_model")
        self.app.checkpoint.set(str(self.checkpoint))
        self.assertEqual(str(self.app.predict_button.cget("state")), "normal")

    def test_old_preferences_and_corrupt_settings_are_supported(self):
        path = self.folder / "settings.json"
        for value in ('broken json', '["en"]', '{"language":"fr"}'):
            path.write_text(value, encoding="utf-8")
            self.assertIn(read_language(path), LANGUAGES)
        path.write_text(json.dumps({"checkpoint": "missing.pt", "device": "cuda", "language": "en"}))
        self.assertEqual(read_language(path), "en")

    def test_close_waits_for_worker_without_showing_error(self):
        self.app.set_images([self.image])
        self.app.start_prediction()
        self.app.close()
        self.assertTrue(self.app.closing)
        self.assertFalse(self.app.closed)
        self.engine.release.set()
        deadline = time.monotonic()+5
        while not self.app.closed and time.monotonic()<deadline:
            self.root.update()
            time.sleep(.02)
        self.assertTrue(self.app.closed)
        self.assertFalse(self.errors)


if __name__ == "__main__":
    unittest.main()
