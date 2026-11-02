import csv
import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from PIL import Image

from streetgeo.core.checkpoints import save_checkpoint
from streetgeo.core.model import StreetGeoModel
from streetgeo.inference.osv_test import OSVTestSampler, attach_ground_truth, distance_km
from streetgeo.ui.prediction import PredictionApp
from test_model import tiny_config


def test_fixture(folder):
    root = Path(folder) / "test"
    images = root / "00"
    images.mkdir(parents=True)
    identifier = "9007199254740993"
    Image.new("RGB", (64, 48), (80, 120, 160)).save(images / f"{identifier}.jpg")
    with (root / "test.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "latitude", "longitude"])
        writer.writerow([identifier, -33.8688, 151.2093])
        writer.writerow(["missing", 0, 0])
    return root, identifier


class SamplerTests(unittest.TestCase):
    def test_only_matches_local_official_test_with_exact_string_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root, identifier = test_fixture(folder)
            train = root / "train"
            train.mkdir()
            Image.new("RGB", (20, 20)).save(train / "missing.jpg")
            sampler = OSVTestSampler()
            for _ in range(3):
                sample = sampler.pick(root)
                self.assertEqual(sample["osv5m_id"], identifier)
                self.assertEqual(sample["latitude"], -33.8688)
                self.assertEqual(sample["longitude"], 151.2093)
            Path(sample["image"]).unlink()
            with self.assertRaises(ValueError):
                sampler.pick(root)

    def test_huggingface_layout_and_invalid_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            root, identifier = test_fixture(folder)
            alternate = root.parent / "images" / "test"
            alternate.parent.mkdir()
            (root / "00").rename(alternate)
            self.assertEqual(OSVTestSampler().pick(root)["osv5m_id"], identifier)
            text = (root / "test.csv").read_text().replace("-33.8688", "nan")
            (root / "test.csv").write_text(text)
            with self.assertRaisesRegex(ValueError, "没有可测试"):
                OSVTestSampler().pick(root)

    def test_csv_without_extracted_images_gives_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _ = test_fixture(folder)
            next((root / "00").glob("*.jpg")).unlink()
            with self.assertRaisesRegex(ValueError, "解压"):
                OSVTestSampler().pick(root)

    def test_comparison_does_not_mutate_predictions_and_handles_date_line(self):
        original = {"candidates": [{"latitude": 0, "longitude": -179.9}]}
        result = attach_ground_truth(original, {"latitude": 0, "longitude": 179.9})
        self.assertAlmostEqual(result["top1_error_km"], 22.239, delta=.01)
        self.assertNotIn("error_km", original["candidates"][0])
        self.assertEqual(distance_km(0, 0, 0, 0), 0)


class RandomTestGuiTests(unittest.TestCase):
    def test_random_image_prediction_truth_markers_export_and_clear(self):
        with tempfile.TemporaryDirectory() as folder:
            root_path, identifier = test_fixture(folder)
            checkpoint = Path(folder) / "model.pt"
            save_checkpoint(StreetGeoModel(tiny_config()), checkpoint)
            root = tk.Tk()
            root.withdraw()
            app = PredictionApp(root, settings_path=Path(folder) / "settings.json")
            errors = []
            app.error = lambda error: errors.append(str(error))
            try:
                app.checkpoint.set(str(checkpoint))
                app.device.set("cpu")
                app.test_directory.set(str(root_path))
                original = app.engine.predict
                inputs = []
                def predict(checkpoint, images, **kwargs):
                    inputs.append((images, kwargs))
                    return original(checkpoint, images, **kwargs)
                app.engine.predict = predict
                app.random_test()
                self.assertTrue(app.busy)
                deadline = time.monotonic() + 30
                while app.busy and time.monotonic() < deadline:
                    root.update()
                    time.sleep(.01)
                self.assertFalse(app.busy)
                self.assertFalse(errors, errors)
                self.assertEqual(app.result["ground_truth"]["osv5m_id"], identifier)
                self.assertFalse(inputs[0][1].get("google_coverage_only", False))
                self.assertEqual(len(inputs[0][0]), 1)
                self.assertEqual(app.result["source"], "model")
                self.assertGreaterEqual(app.result["top1_error_km"], 0)
                root.deiconify(); root.update()
                app.map._draw()
                self.assertTrue(app.map.find_withtag("truth-marker"))
                self.assertTrue(app.map.candidates)
                self.assertTrue(app.map._world_overview)
                app.candidate_list.selection_clear(0, "end")
                app.candidate_list.selection_set(1)
                app.select_candidate()
                self.assertIn("候选 2", app.test_comparison.cget("text"))
                output = Path(folder) / "result.json"
                with patch("streetgeo.ui.prediction.filedialog.asksaveasfilename", return_value=str(output)):
                    app.export_result()
                exported = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(exported["ground_truth"]["osv5m_id"], identifier)
                self.assertIn("error_km", exported["candidates"][0])
                app.clear_images()
                app.map._draw()
                self.assertIsNone(app.test_sample)
                self.assertIsNone(app.map.ground_truth)
                self.assertFalse(app.map.find_withtag("truth-marker"))
            finally:
                app.close()


if __name__ == "__main__":
    unittest.main()
