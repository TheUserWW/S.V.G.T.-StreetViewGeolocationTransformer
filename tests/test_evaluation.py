import csv
import json
import math
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from streetgeo.core.checkpoints import load_model, save_checkpoint
from streetgeo.core.model import StreetGeoModel
from streetgeo.inference.evaluate import (describe, geoguessr_score, evaluate_test_set,
                                          format_evaluation_report, summarize_errors)
from streetgeo.inference.osv_test import distance_km, scan_test_samples
from streetgeo.inference.predict import predict_images
from streetgeo.ui.prediction import PredictionApp
from test_model import tiny_config


def fixture(folder, *, incomplete=False):
    folder = Path(folder)
    test = folder / "test"
    images = test / "00"
    images.mkdir(parents=True)
    identifiers = ["9007199254740993", "2", "3", "4", "5"]
    rows = []
    for index, identifier in enumerate(identifiers):
        Image.new("RGB", (48, 32), (index * 30, 100, 160)).save(images / f"{identifier}.jpg")
        rows.append([identifier, index * 10, index * 20, "AA" if index < 3 else "BB"])
    if incomplete:
        (images / "4.jpg").write_text("corrupt image")
        rows.extend([["missing", 0, 0, "AA"], ["invalid", "nan", 0, "AA"], rows[0]])
    with (test / "test.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "latitude", "longitude", "country"])
        writer.writerows(rows)
    checkpoint = folder / "tiny.pt"
    save_checkpoint(StreetGeoModel(tiny_config()), checkpoint)
    return checkpoint, test


class StatisticsTests(unittest.TestCase):
    def test_exact_metrics_even_median_quantiles_and_threshold_boundaries(self):
        stats = describe([100, 0, 25, 1])
        self.assertEqual(stats["mean"], 31.5)
        self.assertEqual(stats["median"], 13)
        self.assertEqual(stats["p25"], .75)
        self.assertEqual(stats["p75"], 43.75)
        self.assertAlmostEqual(stats["std"], math.sqrt(sum((x - 31.5)**2 for x in (0, 1, 25, 100)) / 4))
        summary = summarize_errors([0, 1, 25, 200, 750, 2500, 10000])
        self.assertEqual(summary["within_km"]["1"]["count"], 2)
        self.assertEqual(summary["within_km"]["2500"]["fraction"], 6 / 7)
        self.assertEqual(geoguessr_score(0), 5000)
        self.assertEqual(geoguessr_score(.025), 5000)
        self.assertEqual(geoguessr_score(1492.7), 1839)
        self.assertEqual(geoguessr_score(20015), 0)
        self.assertEqual(summary["score"]["median"], geoguessr_score(200))
        self.assertEqual(summary["total_score"], sum(geoguessr_score(x) for x in [0, 1, 25, 200, 750, 2500, 10000]))
        self.assertEqual(summary["perfect_scores"]["count"], 1)
        self.assertEqual(summary["maximum_total_score"], 35000)
        self.assertEqual(sum(summary["distance_bins_km"].values()), 7)
        for invalid in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                geoguessr_score(invalid)
        # These values distinguish half-up rounding from integer truncation.
        for raw, expected in ((1000.49, 1000), (1000.51, 1001)):
            self.assertEqual(geoguessr_score(-1492.7 * math.log(raw / 5000)), expected)
        self.assertIsNone(describe([])["median"])
        self.assertEqual(describe([3])["std"], 0)


class EvaluationTests(unittest.TestCase):
    def test_complete_batched_predictions_match_single_image_path_and_preserve_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, test = fixture(folder)
            original = checkpoint.read_bytes()
            model = load_model(checkpoint, "cpu")
            samples = scan_test_samples(test)["samples"]
            expected = [predict_images(model, [sample["image"]])[0] for sample in samples]
            calls = []
            real_predict = StreetGeoModel.predict

            def predict(model, images, **kwargs):
                calls.append((tuple(images.shape), kwargs))
                self.assertEqual(set(kwargs), {"top_k"})  # No coordinates or country labels enter inference.
                return real_predict(model, images, **kwargs)

            events = []
            with patch.object(StreetGeoModel, "predict", predict):
                report = evaluate_test_set(checkpoint, test, device="cpu", batch_size=2,
                                           output_root=Path(folder) / "reports", progress=events.append)
            self.assertEqual([shape[0] for shape, _ in calls], [2, 2, 1])
            self.assertTrue(all(shape[1] == 1 for shape, _ in calls))  # Independent samples, one view each.
            self.assertEqual(report["status"], "completed")
            self.assertTrue(report["full_test_complete"])
            self.assertEqual((report["successful"], report["failed"], report["not_run"]), (5, 0, 0))
            self.assertEqual(report["by_country"]["AA"]["error_km"]["count"], 3)
            self.assertEqual(events[-1]["processed"], 5)
            output = Path(report["output_directory"])
            with (output / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["osv5m_id"], "9007199254740993")
            for row, sample, prediction in zip(rows, samples, expected):
                self.assertAlmostEqual(float(row["predicted_latitude"]), prediction.latitude, delta=.00001)
                self.assertAlmostEqual(float(row["top1_error_km"]), distance_km(
                    sample["latitude"], sample["longitude"], prediction.latitude, prediction.longitude), delta=.001)
                self.assertEqual(len(json.loads(row["candidates_json"])), 5)
            self.assertEqual(report, json.loads((output / "summary.json").read_text(encoding="utf-8")))
            self.assertIn("GeoGuessr 全球地图口径", (output / "report.txt").read_text(encoding="utf-8"))
            self.assertTrue(all(row["top1_score"].isdigit() for row in rows))
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_missing_invalid_duplicates_and_corrupt_images_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, test = fixture(folder, incomplete=True)
            report = evaluate_test_set(checkpoint, test, device="cpu", batch_size=2, output_root=folder)
            self.assertEqual(report["status"], "completed")
            self.assertFalse(report["full_test_complete"])
            self.assertEqual((report["successful"], report["failed"], report["not_run"]), (4, 1, 0))
            self.assertEqual(report["coverage"]["metadata_rows"], 8)
            for reason in ("missing_image", "invalid_id_or_coordinates", "duplicate_id"):
                self.assertEqual(report["coverage"][reason], 1)
            self.assertEqual(report["top1"]["error_km"]["count"], 4)
            with (Path(report["output_directory"]) / "excluded.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            with (Path(report["output_directory"]) / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            failed = [row for row in rows if row["status"] == "failed"]
            self.assertEqual(failed[0]["osv5m_id"], "4")
            self.assertTrue(failed[0]["error"])

    def test_stop_and_model_failure_keep_partial_results(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, test = fixture(folder)
            stop = [False]

            def progress(event):
                if event["processed"] >= 2:
                    stop[0] = True

            report = evaluate_test_set(checkpoint, test, device="cpu", batch_size=2, output_root=folder,
                                       progress=progress, cancelled=lambda: stop[0])
            self.assertEqual(report["status"], "cancelled")
            self.assertEqual((report["successful"], report["not_run"]), (2, 3))
            self.assertFalse(report["full_test_complete"])
            real_predict = StreetGeoModel.predict
            calls = [0]

            def fail_second_batch(model, *args, **kwargs):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError("synthetic device failure")
                return real_predict(model, *args, **kwargs)

            with patch.object(StreetGeoModel, "predict", fail_second_batch):
                failed = evaluate_test_set(checkpoint, test, device="cpu", batch_size=2, output_root=folder)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual((failed["successful"], failed["not_run"]), (2, 3))
            self.assertIn("synthetic device failure", format_evaluation_report(failed))
            self.assertNotEqual(report["output_directory"], failed["output_directory"])

    def test_no_eligible_images_and_scan_cancel_do_not_load_model(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, test = fixture(folder)
            for image in (test / "00").glob("*"):
                image.unlink()
            with patch("streetgeo.core.checkpoints.load_model") as load:
                report = evaluate_test_set(checkpoint, test, device="cpu", output_root=folder)
                cancelled = evaluate_test_set(checkpoint, test, device="cpu", output_root=folder,
                                              cancelled=lambda: True)
            load.assert_not_called()
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["coverage"]["missing_image"], 5)
            self.assertIsNone(report["top1"]["score"]["mean"])
            self.assertEqual(cancelled["status"], "cancelled")


class EvaluationGuiTests(unittest.TestCase):
    def test_background_full_test_updates_statistics_and_saves_report(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, test = fixture(folder, incomplete=True)
            root = tk.Tk()
            root.withdraw()
            app = PredictionApp(root, settings_path=Path(folder) / "settings.json")
            errors = []
            app.error = errors.append
            original = evaluate_test_set

            def evaluate(*args, **kwargs):
                return original(*args, **kwargs, output_root=Path(folder) / "reports")

            try:
                app.checkpoint.set(str(checkpoint))
                app.device.set("cpu")
                app.test_directory.set(str(test))
                app.evaluation_batch.set("2")
                with patch("streetgeo.ui.prediction.evaluate_test_set", side_effect=evaluate):
                    app.start_evaluation()
                    self.assertTrue(app.busy)
                    self.assertEqual(app.stop_evaluation_button.cget("state"), "normal")
                    deadline = time.monotonic() + 30
                    ticks = 0
                    while app.busy and time.monotonic() < deadline:
                        root.update()
                        time.sleep(.01)
                        ticks += 1
                self.assertFalse(app.busy)
                self.assertGreater(ticks, 0)
                self.assertFalse(errors)
                self.assertEqual(app.evaluation_report["successful"], 4)
                self.assertIn("平均", app.evaluation_status.get())
                self.assertIn("中位", app.evaluation_status.get())
                self.assertTrue(app.evaluation_window.winfo_exists())
                self.assertEqual(app.stop_evaluation_button.cget("state"), "disabled")
                self.assertEqual(app.evaluation_report_button.cget("state"), "normal")
            finally:
                app.close()

    def test_close_requests_stop_and_waits_until_partial_report_is_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = tk.Tk()
            root.withdraw()
            app = PredictionApp(root, settings_path=Path(folder) / "settings.json")
            app.evaluating = True
            app._busy(True)
            app.close()
            self.assertTrue(app.evaluation_stop.is_set())
            self.assertTrue(root.winfo_exists())
            app.events.put(("evaluation_error", "stopped test worker"))
            app._poll()
            self.assertIsNone(app._poll_id)


if __name__ == "__main__":
    unittest.main()
