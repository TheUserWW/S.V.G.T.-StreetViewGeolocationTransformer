from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps
import torch

from streetgeo.core.checkpoints import save_checkpoint
from streetgeo.core.model import StreetGeoModel
from streetgeo.datasets.images import load_image
from streetgeo.inference.explain import (AnalysisCancelled, analyze_attention, comparison_text,
                                         content_box, export_explanation, heatmap_image,
                                         joint_probability, occlude_region)
from test_model import tiny_config


class ExplanationFixture:
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name)
        self.images = [self.path / "wide.png", self.path / "portrait.png"]
        for index, (name, size) in enumerate(zip(self.images, ((113, 39), (31, 89)))):
            pixels = np.random.default_rng(123 + index).integers(0, 256, (*size[::-1], 3), dtype=np.uint8)
            Image.fromarray(pixels).save(name)
        torch.manual_seed(12)
        self.model = StreetGeoModel(tiny_config(dropout=0.2)).eval()

    def tearDown(self):
        self.folder.cleanup()

    def analysis(self):
        result = analyze_attention(self.model, self.images)
        result["checkpoint"] = "tiny.pt"
        return result


class ExplanationTests(ExplanationFixture, unittest.TestCase):
    def test_attention_matches_native_mha_and_preserves_predictions_weights_and_mode(self):
        arguments = []
        pool = self.model.encoder.pool_attention
        hook = pool.register_forward_pre_hook(lambda module, args: arguments.append(tuple(arg.clone() for arg in args)))
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        inputs = torch.stack([load_image(path, 32) for path in self.images]).unsqueeze(0)
        with torch.inference_mode():
            expected = self.model(inputs)
        self.model.train()
        result = self.analysis()
        self.assertTrue(self.model.training)
        hook.remove()
        self.model.eval()
        with torch.inference_mode():
            _, weights = pool(*arguments[-1], need_weights=True, average_attn_weights=True)
        np.testing.assert_allclose(result["maps"].reshape(2, 2, 16), weights.numpy(), rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(result["maps"].sum(axis=(-1, -2)), 1, atol=1e-6)
        self.assertEqual([item["cell_id"] for item in result["candidates"]], expected.candidate_cell_ids[0].tolist())
        np.testing.assert_allclose([item["cell_probability"] for item in result["candidates"]],
                                   expected.candidate_probabilities[0].numpy(), rtol=1e-5)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertFalse(pool._forward_hooks)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_cancel_and_forward_failure_always_remove_hook_and_restore_mode(self):
        stop = threading.Event()
        hook = self.model.encoder.pool_attention.register_forward_hook(lambda *args: stop.set())
        self.model.train()
        try:
            with self.assertRaises(AnalysisCancelled):
                analyze_attention(self.model, self.images, cancelled=stop.is_set)
            self.assertTrue(self.model.training)
            self.assertEqual(len(self.model.encoder.pool_attention._forward_hooks), 1)
        finally:
            hook.remove()
        with patch.object(self.model, "forward", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                self.analysis()
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.encoder.pool_attention._forward_hooks)

    def test_occlusion_changes_only_selected_view_content_and_compares_original_cells(self):
        analysis = self.analysis()
        seen = []
        hook = self.model.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        try:
            result = occlude_region(self.model, analysis, 0, (0.2, 0.1, 0.7, 0.8))
        finally:
            hook.remove()
        original, masked = analysis["inputs"], seen[0]
        torch.testing.assert_close(original[0, 1], masked[0, 1], rtol=0, atol=0)
        mask = torch.zeros_like(original, dtype=torch.bool)
        x0, y0, x1, y1 = result["input_box"]
        mask[0, 0, :, y0:y1, x0:x1] = True
        self.assertTrue(torch.equal(original[~mask], masked[~mask]))
        self.assertFalse(torch.equal(original[mask], masked[mask]))
        left, top, right, bottom = content_box(analysis["pictures"][0], 32)
        self.assertGreaterEqual(y0, top)
        self.assertLessEqual(y1, bottom)
        with torch.inference_mode():
            expected = self.model(masked)
        for item, old in zip(result["comparisons"], analysis["candidates"]):
            self.assertEqual(item["cell_id"], old["cell_id"])
            self.assertEqual(item["before"], old["cell_probability"])
            self.assertAlmostEqual(item["after"], joint_probability(expected, old["cell_id"], 4), places=7)
        self.assertIn("不是与真实位置的误差", comparison_text(analysis, result))
        self.assertFalse(any(p.grad is not None for p in self.model.parameters()))
        # Even when the new Top-5 IDs are different, comparisons retain the old IDs.
        expected.candidate_cell_ids = (expected.candidate_cell_ids + 17) % self.model.config.total_cells
        with patch.object(self.model, "forward", return_value=expected):
            changed = occlude_region(self.model, analysis, 1, (0, 0, 1, 1))
        self.assertEqual([x["cell_id"] for x in changed["comparisons"]],
                         [x["cell_id"] for x in analysis["candidates"]])

    def test_letterbox_mapping_orientation_and_export(self):
        for size in ((113, 39), (31, 89), (41, 41), (101, 24), (24, 101)):
            picture = Image.new("RGB", size, "white")
            padded = ImageOps.pad(picture, (32, 32), color="black")
            self.assertEqual(content_box(picture, 32), padded.getbbox())
        rotated = Image.new("RGB", (80, 40), "orange")
        exif = rotated.getexif()
        exif[274] = 6
        path = self.path / "rotated.jpg"
        rotated.save(path, exif=exif)
        result = analyze_attention(self.model, [path])
        self.assertEqual(result["pictures"][0].size, (40, 80))
        analysis = self.analysis()
        np.testing.assert_array_equal(np.asarray(heatmap_image(analysis, 0, None, 0)),
                                      np.asarray(analysis["pictures"][0]))
        masked = occlude_region(self.model, analysis, 0, (0.1, 0.1, 0.5, 0.5))
        output = self.path / "解释.png"
        export_explanation(analysis, 0, 1, 0.6, output, masked)
        with Image.open(output) as image:
            self.assertEqual(image.width, 1500)
            self.assertGreater(image.height, 900)
            self.assertIn("池化注意力", image.info["Description"])
        for bad in ((0, 0, 0, 1), (-1, 0, 1, 1), (0, 0, float("nan"), 1)):
            with self.assertRaises(ValueError):
                occlude_region(self.model, analysis, 0, bad)


class ExplanationGuiTests(ExplanationFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        import tkinter as tk
        from streetgeo.ui.prediction import PredictionApp
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = PredictionApp(self.root, settings_path=self.path / "settings.json")
        self.app.error = lambda error: self.fail(str(error))
        self.app.images = self.images
        checkpoint = self.path / "tiny.pt"
        save_checkpoint(self.model, checkpoint)
        self.app.checkpoint.set(str(checkpoint))
        self.app.device.set("cpu")

    def tearDown(self):
        import tkinter as tk
        try:
            if self.root.winfo_exists():
                self.app._destroy()
        except tk.TclError:
            pass
        super().tearDown()

    def wait(self):
        deadline = time.monotonic() + 15
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(self.app.busy, "worker did not finish")
        self.root.update()

    def test_worker_heatmaps_selection_occlusion_export_and_window_resize(self):
        self.app.open_explanation()
        window = self.app.explanation_window
        window.withdraw()
        window.show_error = lambda error: self.fail(str(error))
        self.assertTrue(self.app.busy)
        self.wait()
        self.assertIsNotNone(window.analysis)
        reused = self.app.engine.model
        window.deiconify()
        for geometry in ("920x720", "1450x950"):
            window.geometry(geometry)
            self.root.update()
            self.assertGreater(window.original.winfo_width(), 200)
            self.assertLessEqual(window.tabs.winfo_x() + window.tabs.winfo_width(), window.winfo_width())
            self.assertLessEqual(window.details.winfo_rooty() + window.details.winfo_height(),
                                 window.winfo_rooty() + window.winfo_height())
        canvas = window.original
        left, top, width, height = canvas.bounds
        canvas._press(SimpleNamespace(x=left + width * .1, y=top + height * .2))
        canvas._release(SimpleNamespace(x=left + width * .6, y=top + height * .8))
        np.testing.assert_allclose(window.rectangle, (.1, .2, .6, .8), atol=1e-6)
        window.run_occlusion()
        self.wait()
        self.assertIs(self.app.engine.model, reused)
        self.assertIsNotNone(window.occlusion)
        self.assertIn("固定比较原候选", window.details.get("1.0", "end"))
        output = self.path / "gui-export.png"
        self.app.start_explanation_job(window, "export", (0, None, .5, str(output)))
        self.wait()
        self.assertTrue(output.is_file())
        window.view_box.current(1)
        window.change_view()
        self.assertIsNone(window.rectangle)
        window.close()
        self.assertTrue(window.closed)
        self.root.update()

    def test_cancel_child_close_and_stale_checkpoint(self):
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
            window.close()
            self.assertTrue(self.app.explanation_stop.is_set())
            finish.set()
            self.wait()
        self.assertFalse(self.app.explaining)
        self.app.open_explanation()
        window = self.app.explanation_window
        window.withdraw()
        self.wait()
        messages = []
        window.show_error = messages.append
        self.app.engine.key = None
        window.select_region((0, 0, .5, .5))
        window.run_occlusion()
        self.wait()
        self.assertIn("模型已切换", messages[0])

    def test_import_error_releases_busy_and_parent_close_waits_for_worker(self):
        from streetgeo.ui.explanation import ExplanationWindow
        messages = []
        with patch.object(ExplanationWindow, "show_error", lambda window, text: messages.append(text)), \
                patch.dict(sys.modules, {"streetgeo.inference.explain": None}):
            self.app.open_explanation()
            self.app.explanation_window.withdraw()
            self.wait()
        self.assertTrue(messages)
        self.assertFalse(self.app.explaining)
        started, finish = threading.Event(), threading.Event()
        real = self.app.engine.predict

        def blocked(*args, **kwargs):
            started.set()
            finish.wait(5)
            return real(*args, **kwargs)

        with patch.object(self.app.engine, "predict", side_effect=blocked):
            self.app.open_explanation()
            window = self.app.explanation_window
            self.assertTrue(started.wait(5))
            self.app.close()
            self.assertTrue(self.app.closing)
            self.assertTrue(self.app.explanation_stop.is_set())
            self.assertTrue(self.root.winfo_exists())
            finish.set()
            deadline = time.monotonic() + 10
            while self.app.busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(.01)
            self.assertFalse(self.app.busy)
            self.assertTrue(window.closed)


if __name__ == "__main__":
    unittest.main()
