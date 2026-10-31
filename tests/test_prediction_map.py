from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from streetgeo.core.checkpoints import save_checkpoint
from streetgeo.maps.projection import ControlPoint, MAX_LATITUDE, ScreenshotCalibration, parse_coordinates, project, unproject
from streetgeo.core.model import StreetGeoModel
from streetgeo.inference.backend import InferenceEngine, TileCache
from test_model import tiny_config


def controls_for(places, transform):
    return [ControlPoint(lat, lon, *transform(*project(lat, lon))) for lat, lon in places]


class ProjectionTests(unittest.TestCase):
    def test_round_trip_and_coordinate_order(self):
        for lat, lon in ((34.0522, -118.2437), (-33.86, 151.2), (MAX_LATITUDE, 179.9), (-80, -179.9)):
            actual = unproject(*project(lat, lon))
            self.assertAlmostEqual(actual[0], lat)
            self.assertAlmostEqual(actual[1], lon)
        self.assertEqual(parse_coordinates("34.0522， -118.2437"), (34.0522, -118.2437))
        for text in ("91, 0", "0, 181", "nan, 12", "12", "1,2,3"):
            with self.assertRaises(ValueError):
                parse_coordinates(text)

    def test_affine_calibration_recovers_unseen_point(self):
        transform = lambda x, y: (7000*x + 1000*y - 900, -500*x + 6500*y - 1500)
        controls = controls_for(((47, -123), (47, -80), (27, -123), (27, -80)), transform)
        fit = ScreenshotCalibration(controls)
        expected = transform(*project(34.05, -118.24))
        actual = fit.locate(34.05, -118.24)
        self.assertAlmostEqual(actual[0], expected[0], places=8)
        self.assertAlmostEqual(actual[1], expected[1], places=8)
        self.assertLess(fit.rms_pixels, 1e-8)

    def test_date_line_calibration(self):
        def transform(lat, lon):
            x, y = project(lat, lon)
            return ((x if lon >= 0 else x + 1) * 8000 - 7400, y * 8000 - 1800)
        controls = [ControlPoint(lat, lon, *transform(lat, lon)) for lat, lon in
                    ((50, 170), (50, -170), (20, 170), (20, -170))]
        fit = ScreenshotCalibration(controls)
        for lat, lon in ((35, 179.5), (35, -179.5)):
            actual, expected = fit.locate(lat, lon), transform(lat, lon)
            self.assertAlmostEqual(actual[0], expected[0], places=7)
            self.assertAlmostEqual(actual[1], expected[1], places=7)

    def test_bad_controls_rejected_and_residual_reports_noise(self):
        with self.assertRaises(ValueError):
            ScreenshotCalibration([ControlPoint(0, lon, lon + 20, lon + 30) for lon in (0, 1, 2)])
        with self.assertRaises(ValueError):
            ScreenshotCalibration([ControlPoint(0, 0, 0, 0)] * 3)
        controls = controls_for(((45, -120), (45, -80), (25, -120), (25, -80)),
                                lambda x, y: (x*2000, y*2000))
        point = controls[-1]
        controls[-1] = ControlPoint(point.latitude, point.longitude, point.x + 20, point.y)
        self.assertGreater(ScreenshotCalibration(controls).rms_pixels, 2)
        with self.assertRaises(ValueError):
            ScreenshotCalibration([ControlPoint(86, 0, 0, 0), *controls])


class TileTests(unittest.TestCase):
    def test_cache_wraps_x_respects_freshness_and_requests_only_one_tile(self):
        buffer = BytesIO()
        Image.new("RGB", (256, 256), "blue").save(buffer, "PNG")
        calls = []
        now = [1000.0]

        class Response(BytesIO):
            headers = {"Cache-Control": "max-age=100"}

        def opener(request, timeout):
            calls.append(request)
            return Response(buffer.getvalue())

        with tempfile.TemporaryDirectory() as folder:
            cache = TileCache(folder, opener=opener, clock=lambda: now[0])
            self.assertEqual(cache.get(2, -1, 1).size, (256, 256))
            self.assertEqual(calls[0].full_url, "https://tile.openstreetmap.org/2/3/1.png")
            self.assertIn("StreetGeoLocalMap", calls[0].get_header("User-agent"))
            cache.get(2, 3, 1)
            TileCache(folder, opener=opener, clock=lambda: now[0]).get(2, 3, 1)
            self.assertEqual(len(calls), 1)
            now[0] += 101
            cache.get(2, 3, 1)
            self.assertEqual(len(calls), 2)

    def test_no_store_is_not_persisted(self):
        buffer = BytesIO()
        Image.new("RGB", (256, 256)).save(buffer, "PNG")

        class Response(BytesIO):
            headers = {"Cache-Control": "no-store"}

        with tempfile.TemporaryDirectory() as folder:
            cache = TileCache(folder, opener=lambda *args, **kwargs: Response(buffer.getvalue()))
            cache.get(0, 0, 0)
            self.assertEqual(list(Path(folder).rglob("*.png")), [])
            self.assertFalse(cache.memory)


class InferenceTests(unittest.TestCase):
    def test_real_checkpoint_prediction_reuses_model_and_reloads_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            checkpoint, image = folder / "模型.pt", folder / "街景.png"
            model = StreetGeoModel(tiny_config()).eval()
            save_checkpoint(model, checkpoint)
            Image.new("RGB", (48, 32), "orange").save(image)
            engine = InferenceEngine()
            first = engine.predict(checkpoint, [image], device="cpu")
            cached_model = engine.model
            second = engine.predict(checkpoint, [image], device="cpu")
            self.assertEqual(len(first["candidates"]), 5)
            self.assertEqual(first["candidates"], second["candidates"])
            self.assertFalse(first["reused_model"])
            self.assertTrue(second["reused_model"])
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(0.01)
            save_checkpoint(model, checkpoint)
            third = engine.predict(checkpoint, [image], device="cpu")
            self.assertFalse(third["reused_model"])
            self.assertIsNot(engine.model, cached_model)
            json.dumps(third, allow_nan=False)


class GuiIntegrationTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from streetgeo.ui.prediction import PredictionApp

        self.folder = tempfile.TemporaryDirectory()
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = PredictionApp(self.root, settings_path=Path(self.folder.name) / "settings.json")
        self.app.error = lambda error: self.fail(str(error))
        self.root.update_idletasks()

    def tearDown(self):
        self.app.close()
        self.folder.cleanup()

    def test_manual_result_and_screenshot_calibration_survive_preview_resize(self):
        app = self.app
        app.manual.set("34.0522, -118.2437")
        app.use_manual()
        self.assertEqual(app.result["source"], "manual")
        path = Path(self.folder.name) / "map.png"
        Image.new("RGB", (900, 600), "white").save(path)
        app.set_screenshot(path)
        self.root.update_idletasks()
        self.assertTrue(app.screenshot_visible())
        app.points = controls_for(((47, -125), (47, -75), (24, -125), (24, -75)),
                                  lambda x, y: (x * 5400 - 800, y * 3500 - 1150))
        app.refresh_calibration()
        self.assertIsNotNone(app.target)
        target = app.target
        app.screen.configure(width=370, height=250)
        self.root.update_idletasks()
        app.screen.draw()
        actual = app.screen.canvas_to_image(*app.screen.image_to_canvas(*target))
        self.assertAlmostEqual(actual[0], target[0])
        self.assertAlmostEqual(actual[1], target[1])
        app.manual.set("-33.8, 151.2")
        app.use_manual()
        self.assertIsNone(app.target)
        self.assertIn("范围", app.screen_info.cget("text"))
        app.set_screenshot(path)
        self.assertIsNone(app.fit)
        self.assertEqual(app.points, [])

    def test_export_restore_hash_validation_and_png_dimensions(self):
        app = self.app
        folder = Path(self.folder.name)
        path = folder / "map.png"
        Image.new("RGB", (900, 600), "white").save(path)
        app.set_screenshot(path)
        app.points = controls_for(((47, -125), (47, -75), (24, -125), (24, -75)),
                                  lambda x, y: (x * 5400 - 800, y * 3500 - 1150))
        app.refresh_calibration()
        app.manual.set("34.0522, -118.2437")
        app.use_manual()
        calibration = folder / "calibration.json"
        with patch("streetgeo.ui.prediction.filedialog.asksaveasfilename", return_value=str(calibration)):
            app.save_calibration()
        original = list(app.points)
        app.clear_points()
        with patch("streetgeo.ui.prediction.filedialog.askopenfilename", return_value=str(calibration)):
            app.load_calibration()
        self.assertEqual(app.points, original)
        overlay = folder / "hint.png"
        with patch("streetgeo.ui.prediction.filedialog.asksaveasfilename", return_value=str(overlay)):
            app.export_overlay()
        with Image.open(overlay) as image:
            self.assertEqual(image.size, (900, 600))
            pixels = np.asarray(image)
            self.assertGreater(int(((pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 100)).sum()), 30)
        replacement = folder / "other.png"
        Image.new("RGB", (900, 600), "blue").save(replacement)
        app.set_screenshot(replacement)
        errors = []
        app.error = errors.append
        with patch("streetgeo.ui.prediction.filedialog.askopenfilename", return_value=str(calibration)):
            app.load_calibration()
        self.assertIn("截图不同", errors[0])
        self.assertFalse(app.points)

    def test_background_prediction_keeps_ui_events_and_updates_candidates(self):
        app = self.app
        folder = Path(self.folder.name)
        checkpoint, image = folder / "tiny.pt", folder / "street.png"
        save_checkpoint(StreetGeoModel(tiny_config()), checkpoint)
        Image.new("RGB", (32, 32), "green").save(image)
        app.images = [str(image)]
        app.checkpoint.set(str(checkpoint))
        app.device.set("cpu")
        app.start_prediction()
        self.assertTrue(app.busy)
        ticks = 0
        deadline = time.monotonic() + 30
        while app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
            ticks += 1
        self.assertFalse(app.busy)
        self.assertGreater(ticks, 0)
        self.assertEqual(len(app.candidates), 5)
        self.assertEqual(app.result["source"], "model")


if __name__ == "__main__":
    unittest.main()
