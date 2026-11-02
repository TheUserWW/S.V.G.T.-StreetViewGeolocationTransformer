import csv
import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg

from streetgeo.inference.evaluation_charts import (build_evaluation_figure, distribution_data,
                                                   export_evaluation_chart, load_evaluation_report, save_evaluation_charts)
from streetgeo.inference.evaluate import describe, evaluate_test_set, geoguessr_score, summarize_errors
from streetgeo.ui.evaluation import EvaluationReportWindow
from streetgeo.ui.prediction import PredictionApp
from test_evaluation import fixture


def example_report(folder, errors=None):
    errors = [0, .025, 1, 25, 200, 750, 2500, 10000] if errors is None else errors
    return {"status": "cancelled", "created_at": "2026-10-31T22:26:32+00:00", "checkpoint": "model.pt",
            "test_directory": str(folder), "output_directory": str(folder), "batch_size": 16,
            "device": "cpu", "elapsed_seconds": 1.5, "successful": len(errors), "failed": 1,
            "not_run": 92, "full_test_complete": False,
            "coverage": {"metadata_rows": 102, "available_samples": 101, "missing_image": 1,
                         "invalid_id_or_coordinates": 0, "duplicate_id": 0},
            "top1": summarize_errors(errors), "best_of_5_error_km": describe(errors),
            "visualization": distribution_data(errors, [geoguessr_score(d) for d in errors]),
            "by_country": {"AA": summarize_errors(errors[:3]), "BB": summarize_errors(errors[3:])} if errors else {}}


class DistributionTests(unittest.TestCase):
    def test_histogram_boundaries_and_tied_ecdf(self):
        data = distribution_data([0, 0, 1, 1, 10, 20], [0, 499, 500, 999, 4500, 5000])
        self.assertEqual(data["score_bin_counts"], [2, 2, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertEqual(data["error_cdf"]["error_km"], [0, 1, 10, 20])
        self.assertEqual(data["error_cdf"]["fraction"], [2 / 6, 4 / 6, 5 / 6, 1])
        self.assertEqual(distribution_data([], [])["score_bin_counts"], [0] * 10)
        self.assertEqual(distribution_data([3, 3], [1, 1])["error_cdf"]["fraction"], [1])
        with self.assertRaises(ValueError):
            distribution_data([0], [5000.2])

    def test_large_curve_is_bounded_and_keeps_true_endpoints_and_ranks(self):
        errors = list(range(210122))
        data = distribution_data(errors, [0] * len(errors))
        curve = data["error_cdf"]
        self.assertEqual(len(curve["error_km"]), 512)
        self.assertTrue(curve["downsampled"])
        self.assertEqual((curve["error_km"][0], curve["error_km"][-1]), (0, 210121))
        for error, fraction in zip(curve["error_km"], curve["fraction"]):
            self.assertEqual(fraction, (error + 1) / len(errors))


class ChartFileTests(unittest.TestCase):
    def test_export_has_fixed_geometry_and_preserves_data_zoom(self):
        with tempfile.TemporaryDirectory() as folder:
            report = example_report(folder)
            path = Path(folder) / "zoom.png"
            limits = [((0, 2500), (0, 3)), ((0, 5), (0, 5)), ((0, 4), (0, 100)), ((0, 1000), (0, 100))]
            from matplotlib.figure import Figure
            original = Figure.savefig
            saved_limits = []

            def record(figure, *args, **kwargs):
                saved_limits.extend((tuple(axis.get_xlim()), tuple(axis.get_ylim())) for axis in figure.axes)
                return original(figure, *args, **kwargs)

            with patch.object(Figure, "savefig", record):
                export_evaluation_chart(report, "overview", path, limits=limits)
            with Image.open(path) as image:
                self.assertEqual(image.size, (2340, 1620))
            self.assertEqual(saved_limits, limits)

    def test_old_report_uses_csv_without_inference_and_can_be_moved(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            report = example_report(folder)
            del report["visualization"]
            report["output_directory"] = "Z:/no-longer-exists"
            summary = folder / "summary.json"
            original = json.dumps(report)
            summary.write_text(original, encoding="utf-8")
            errors = [0, .025, 1, 25, 200, 750, 2500, 10000]
            with (folder / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=("status", "top1_error_km", "top1_score"))
                writer.writeheader()
                writer.writerow({"status": "failed"})
                writer.writerows({"status": "ok", "top1_error_km": d, "top1_score": geoguessr_score(d)} for d in errors)
            with patch("streetgeo.core.checkpoints.load_model") as load:
                loaded = load_evaluation_report(summary)
            load.assert_not_called()
            self.assertEqual(loaded["output_directory"], str(folder.resolve()))
            self.assertEqual(sum(loaded["visualization"]["score_bin_counts"]), 8)
            self.assertEqual(summary.read_text(encoding="utf-8"), original)
            report["successful"] += 1
            summary.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "数量不一致"):
                load_evaluation_report(summary)

    def test_standalone_images_svg_and_partial_labels_have_correct_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            report = example_report(folder)
            saved = save_evaluation_charts(report)
            self.assertEqual(len(saved), 4)
            for filename in saved.values():
                path = Path(folder) / filename
                self.assertGreater(path.stat().st_size, 1000)
                if path.suffix == ".png":
                    with Image.open(path) as image:
                        image.verify()
                else:
                    self.assertEqual(ET.parse(path).getroot().tag, "{http://www.w3.org/2000/svg}svg")
            figure = build_evaluation_figure(report)
            canvas = FigureCanvasAgg(figure)
            canvas.draw()
            self.assertIn("部分结果", figure._suptitle.get_text())
            self.assertEqual(sum(p.get_height() for p in figure.axes[0].patches), 8)
            self.assertEqual([p.get_height() for p in figure.axes[2].patches],
                             [100 * item["fraction"] for item in report["top1"]["within_km"].values()])
            for text in figure.texts:
                bounds = text.get_window_extent(canvas.get_renderer())
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertLessEqual(bounds.x1, figure.bbox.width)
                self.assertGreaterEqual(bounds.y0, 0)
                self.assertLessEqual(bounds.y1, figure.bbox.height)
            self.assertFalse(any("$" in text.get_text() for text in figure.axes[3].get_xticklabels()))
            figure.clear()

    def test_empty_result_renders_and_chart_failure_keeps_numeric_results(self):
        with tempfile.TemporaryDirectory() as folder:
            for view in ("overview", "countries"):
                figure = build_evaluation_figure(example_report(folder, []), view)
                FigureCanvasAgg(figure).draw()
                figure.clear()
            checkpoint, test = fixture(folder)
            with patch("streetgeo.inference.evaluate.save_evaluation_charts", side_effect=RuntimeError("plot failure")):
                report = evaluate_test_set(checkpoint, test, device="cpu", output_root=folder)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["successful"], 5)
            self.assertEqual(report["visualization_error"], "plot failure")
            self.assertTrue((Path(report["output_directory"]) / "results.csv").is_file())


class ChartGuiTests(unittest.TestCase):
    def test_resize_dpi_tabs_and_scroll_keep_chart_layout_and_toolbar_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = tk.Tk()
            root.withdraw()
            initial_scaling = root.tk.call("tk", "scaling")
            callbacks = []
            root.report_callback_exception = lambda kind, error, trace: callbacks.append(str(error))
            try:
                for scale in (1, 1.5, 2):
                    root.tk.call("tk", "scaling", scale * 96 / 72)
                    report = example_report(folder, [0, 25, 200, 2500, 10000] * 15)
                    report["by_country"] = {f"A{i:02}": summarize_errors([0, 25, 200, 2500, 10000]) for i in range(15)}
                    window = EvaluationReportWindow(root, report)
                    try:
                        for geometry in ("900x650", "1600x650", "1200x900"):
                            window.geometry(geometry)
                            for view in ("overview", "countries"):
                                window.notebook.select(window.frames[view])
                                root.update()
                                viewport = window.viewports[view]
                                chart = window.canvases[view]
                                widget = chart.get_tk_widget()
                                width, height = widget.winfo_width(), widget.winfo_height()
                                self.assertAlmostEqual(width / height, 13 / 9, delta=.01)
                                self.assertGreaterEqual(width, 11 * chart.figure.dpi - 1)
                                toolbar = window.toolbars[view]
                                # Compare root coordinates: scrolling content may
                                # have negative local coordinates inside its canvas.
                                self.assertLessEqual(viewport.winfo_rooty() + viewport.winfo_height(), toolbar.winfo_rooty())
                                self.assertLessEqual(toolbar.winfo_rooty() + toolbar.winfo_height(),
                                                     window.winfo_rooty() + window.winfo_height())
                                figure = chart.figure
                                renderer = chart.get_renderer()
                                for text in figure.texts:
                                    bounds = text.get_window_extent(renderer)
                                    self.assertGreaterEqual(bounds.x0, 0)
                                    self.assertLessEqual(bounds.x1, figure.bbox.width)
                                    self.assertGreaterEqual(bounds.y0, 0)
                                    self.assertLessEqual(bounds.y1, figure.bbox.height)
                                for left in figure.axes:
                                    bounds = left.get_tightbbox(renderer)
                                    self.assertGreaterEqual(bounds.x0, 0)
                                    self.assertLessEqual(bounds.x1, figure.bbox.width)
                                    self.assertGreaterEqual(bounds.y0, 0)
                                    self.assertLessEqual(bounds.y1, figure.bbox.height)
                                    for right in figure.axes:
                                        if left is not right:
                                            self.assertFalse(left.get_tightbbox(renderer).overlaps(right.get_tightbbox(renderer)))
                        window.geometry("900x650")
                        window.notebook.select(window.frames["overview"])
                        root.update()
                        viewport = window.viewports["overview"]
                        start = viewport.viewport.yview()[0]
                        self.assertEqual(viewport._wheel(SimpleNamespace(delta=-120, state=0)), "break")
                        self.assertGreater(viewport.viewport.yview()[0], start)
                        start = viewport.viewport.xview()[0]
                        viewport._wheel(SimpleNamespace(delta=-120, state=1))
                        self.assertGreater(viewport.viewport.xview()[0], start)
                        axis = window.figures["overview"].axes[0]
                        axis.set_xlim(100, 2000)
                        window.toolbars["overview"].zoom()
                        self.assertIsNone(viewport._wheel(SimpleNamespace(delta=-120, state=0)))
                        window.geometry("1300x720")
                        root.update()
                        self.assertEqual(tuple(axis.get_xlim()), (100, 2000))
                        output = Path(folder) / f"resized_{scale}.png"
                        with patch("streetgeo.ui.evaluation.filedialog.asksaveasfilename", return_value=str(output)):
                            window.toolbars["overview"].save_figure()
                        with Image.open(output) as image:
                            self.assertEqual(image.size, (2340, 1620))
                        viewport._schedule_resize()
                    finally:
                        window.destroy()
                        root.update()
                self.assertFalse(callbacks, callbacks)
            finally:
                root.tk.call("tk", "scaling", initial_scaling)
                root.destroy()

    def test_narrow_header_wraps_buttons_and_restores_wide_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = tk.Tk()
            root.withdraw()
            window = EvaluationReportWindow(root, example_report(folder))
            try:
                window.minsize(600, 400)
                window.heading.configure(font=("Microsoft YaHei UI", 26, "bold"))
                window.geometry("650x650")
                root.update()
                self.assertTrue(window._stacked_header)
                self.assertGreaterEqual(window.action_buttons.winfo_rooty(),
                                        window.heading.winfo_rooty() + window.heading.winfo_height())
                window.geometry("1800x900")
                root.update()
                self.assertFalse(window._stacked_header)
                self.assertEqual(window.action_buttons.grid_info()["row"], 0)
            finally:
                window.destroy()
                root.destroy()

    def test_chart_tabs_export_and_repeated_close(self):
        with tempfile.TemporaryDirectory() as folder:
            root = tk.Tk()
            root.withdraw()
            window = EvaluationReportWindow(root, example_report(folder))
            try:
                root.update()
                self.assertEqual(len(window.notebook.tabs()), 3)
                self.assertIn("overview", window.figures)
                window.notebook.select(window.frames["countries"])
                root.update()
                self.assertIn("countries", window.figures)
                output = Path(folder) / "export.png"
                with patch("streetgeo.ui.evaluation.filedialog.asksaveasfilename", return_value=str(output)):
                    window.export_chart()
                self.assertTrue(output.is_file())
                window.notebook.select(window.frames["details"])
                root.update()
                self.assertEqual(window.export_button.cget("state"), "disabled")
                window.destroy()
                second = EvaluationReportWindow(root, example_report(folder, []))
                root.update()
                second.destroy()
                root.update()
            finally:
                if window.winfo_exists():
                    window.destroy()
                root.destroy()

    def test_history_report_loader_is_background_only_and_does_not_run_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            report = example_report(folder)
            summary = folder / "summary.json"
            summary.write_text(json.dumps(report), encoding="utf-8")
            root = tk.Tk()
            root.withdraw()
            app = PredictionApp(root, settings_path=folder / "settings.json")
            errors = []
            app.error = errors.append
            try:
                with patch("streetgeo.ui.prediction.filedialog.askopenfilename", return_value=str(summary)), \
                        patch.object(app.engine, "predict") as predict:
                    app.open_evaluation_report()
                    self.assertTrue(app.busy)
                    deadline = time.monotonic() + 30
                    while app.busy and time.monotonic() < deadline:
                        root.update()
                        time.sleep(.01)
                predict.assert_not_called()
                self.assertFalse(app.busy)
                self.assertFalse(errors)
                self.assertEqual(app.evaluation_report["successful"], 8)
                self.assertTrue(app.evaluation_window.winfo_exists())
            finally:
                app.close()


if __name__ == "__main__":
    unittest.main()
