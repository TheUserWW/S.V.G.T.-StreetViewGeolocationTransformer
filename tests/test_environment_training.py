from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

from PIL import Image, ImageOps
import torch

from streetgeo.core.checkpoints import read_checkpoint, save_checkpoint
from streetgeo.core.model import StreetGeoModel
from streetgeo.datasets.images import IMAGE_FILL, IMAGE_MEAN, IMAGE_STD, StreetViewDataset, load_image
from streetgeo.datasets.road_mask import RoadMask
from streetgeo.training.environment import LEGACY_RECIPE, parameter_groups, train_mode, resolve_recipe
from streetgeo.training.runtime import AlternatingBatchSampler
from streetgeo.ui.backend import DEFAULTS, build_training
from test_model import tiny_config


def fixture(folder):
    rows = []
    for index in range(8):
        path = folder / f"{index}.png"
        image = Image.new("RGB", (80, 48), (index * 20, 100, 180))
        image.paste((220, index * 20, 40), (0, 32, 80, 48))
        image.save(path)
        rows.append({"image": path.name, "latitude": 10 + index * 5, "longitude": -120 + index * 8,
                     "mapillary_id": str(index), "source": "Mapillary"})
    for name, part in (("train", rows[:6]), ("val", rows[6:])):
        (folder / f"{name}.jsonl").write_text("\n".join(map(json.dumps, part)), encoding="utf-8")
    cfg = tiny_config(n_layer=4, dropout=.2)
    base = folder / "base.pt"
    torch.manual_seed(421)
    save_checkpoint(StreetGeoModel(cfg), base)
    return cfg, base


class EnvironmentInputTests(unittest.TestCase):
    def test_legacy_rectangle_is_bottom_third_of_source_content_before_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            for width, height in ((120, 60), (60, 120)):
                path = Path(folder) / f"{width}.png"
                Image.new("RGB", (width, height), (10, 80, 220)).save(path)
                original = path.read_bytes()
                full = load_image(path, 120)
                masked = load_image(path, 120, mask_bottom_third=True)
                x, top = (120 - width) // 2, (120 - height) // 2
                boundary = top + height * 2 // 3
                torch.testing.assert_close(full[:, :boundary], masked[:, :boundary], rtol=0, atol=0)
                fill = torch.tensor([(c / 255 - m) / s for c, m, s in zip(IMAGE_FILL, IMAGE_MEAN, IMAGE_STD)])
                torch.testing.assert_close(masked[:, boundary, x + width // 2], fill)
                self.assertFalse(torch.equal(full[:, boundary, x + width // 2], fill))
                self.assertEqual(path.read_bytes(), original)

    def test_mask_uses_exif_oriented_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path, oriented = folder / "rotated.jpg", folder / "oriented.png"
            image = Image.new("RGB", (90, 60), "blue")
            image.paste("orange", (30, 0, 90, 60))
            exif = image.getexif()
            exif[274] = 6
            image.save(path, exif=exif)
            with Image.open(path) as source:
                ImageOps.exif_transpose(source).convert("RGB").save(oriented)
            torch.testing.assert_close(load_image(path, 96, road_mask=RoadMask()),
                                       load_image(oriented, 96, road_mask=RoadMask()), rtol=0, atol=0)

    def test_trapezoid_masks_center_and_preserves_sides_before_letterboxing(self):
        mask = RoadMask()
        self.assertAlmostEqual(mask.area_fraction, .27)
        with tempfile.TemporaryDirectory() as temporary:
            for width, height in ((120, 60), (60, 120)):
                path = Path(temporary) / f"{width}.png"
                source = Image.new("RGB", (width, height), (10, 80, 220))
                source.save(path)
                masked = mask.apply(source)
                top = round((1 - mask.height) * (height - 1))
                self.assertEqual(masked.getpixel((width // 2, top)), IMAGE_FILL)
                self.assertEqual(masked.getpixel((0, top)), (10, 80, 220))
                self.assertEqual(masked.getpixel((width - 1, top)), (10, 80, 220))
                self.assertEqual(masked.getpixel((width // 2, height - 1)), IMAGE_FILL)
                self.assertEqual(masked.getpixel((width // 2, top - 1)), (10, 80, 220))
                self.assertEqual(source.getpixel((width // 2, height - 1)), (10, 80, 220))
                expected_path = Path(temporary) / "expected.png"
                masked.save(expected_path)
                torch.testing.assert_close(load_image(path, 120, road_mask=mask), load_image(expected_path, 120), rtol=0, atol=0)

    def test_mask_parameters_and_legacy_recipe_are_explicit(self):
        for args in ((.6, .4, .45), (.2, 1.1, .45), (.2, 1, 0), (.2, 1, float("nan"))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                RoadMask(*args)
        run = {"road_mask_top_width": .2, "road_mask_bottom_width": 1., "road_mask_height": .45}
        recipe, mask, legacy = resolve_recipe(run)
        self.assertEqual(recipe["version"], 2)
        self.assertFalse(legacy)
        old, _, legacy = resolve_recipe(run, {"environment_recipe": LEGACY_RECIPE})
        self.assertTrue(legacy)
        self.assertEqual(old["version"], 1)
        with self.assertRaisesRegex(ValueError, "--init-from"):
            resolve_recipe(run, {"environment_recipe": LEGACY_RECIPE}, explicit_geometry=True)
        with self.assertRaisesRegex(ValueError, "recipe changed"):
            resolve_recipe(dict(run, road_mask_top_width=.3), {"environment_recipe": recipe})

    def test_sampler_repeats_same_batch_and_resumes_either_phase(self):
        full = list(AlternatingBatchSampler(7, 3, 123, 0))
        self.assertEqual(len(full), 6)
        for first, second in zip(full[::2], full[1::2]):
            self.assertEqual([index for index, _ in first], [index for index, _ in second])
            self.assertTrue(all(phase == 0 for _, phase in first))
            self.assertTrue(all(phase == 1 for _, phase in second))
        for start in range(7):
            resumed = AlternatingBatchSampler(7, 3, 123, 0, start)
            self.assertEqual(list(resumed), full[start:])
            self.assertEqual(len(resumed), 6 - start)

    def test_frozen_prefix_stays_unchanged_and_attention_and_terminal_layers_learn(self):
        model = StreetGeoModel(tiny_config(n_layer=4, dropout=.2))
        groups = parameter_groups(model, unfreeze_last=2, learning_rate=2e-5,
                                  visual_learning_rate=5e-6, weight_decay=.1)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        train_mode(model, 2)
        self.assertFalse(model.encoder.visual_blocks[0].training)
        self.assertTrue(model.encoder.visual_blocks[2].training)
        optimizer = torch.optim.AdamW(groups)
        result = model(torch.randn(2, 1, 3, 32, 32), targets=torch.tensor([[35., -118.], [10., 120.]]))
        result.loss.backward()
        optimizer.step()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.encoder.pool_queries, before["encoder.pool_queries"]))
        self.assertFalse(torch.equal(model.encoder.visual_blocks[2].attention.qkv.weight,
                                     before["encoder.visual_blocks.2.attention.qkv.weight"]))


class EnvironmentPipelineTests(unittest.TestCase):
    def test_legacy_rectangle_checkpoint_resumes_without_switching_to_trapezoid(self):
        from streetgeo.__main__ import main
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _, base = fixture(folder)
            dataset = StreetViewDataset(folder / "train.jsonl", tiny_config(), mapillary_only=True, environment_specialist=True)
            del dataset.road_mask, dataset.legacy_environment_mask
            torch.testing.assert_close(dataset[0, 1]["images"][0], load_image(folder / "0.png", 32, mask_bottom_third=True))
            common = ["train", "--init-from", str(base), "--environment-specialist", "--train-manifest", str(folder / "train.jsonl"),
                      "--val-manifest", str(folder / "val.jsonl"), "--epochs", "1", "--batch-size", "2",
                      "--num-workers", "0", "--device", "cpu", "--amp", "off"]
            with patch("streetgeo.training.train.resolve_recipe", return_value=(LEGACY_RECIPE, RoadMask(), True)), redirect_stdout(io.StringIO()):
                main([*common, "--output", str(folder / "old_full.pt")])
                main([*common, "--output", str(folder / "old_partial.pt"), "--stop-after-steps", "1"])
            with redirect_stdout(io.StringIO()):
                main(["train", "--resume", str(folder / "old_partial.last.pt"), "--device", "cpu", "--num-workers", "0"])
            expected, actual = read_checkpoint(folder / "old_full.last.pt"), read_checkpoint(folder / "old_partial.last.pt")
            self.assertEqual(actual["training_state"]["environment_recipe"], LEGACY_RECIPE)
            for key, value in expected["model_state_dict"].items():
                torch.testing.assert_close(value, actual["model_state_dict"][key], rtol=0, atol=0)

    def test_alternation_two_validations_and_exact_resume_after_full_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            cfg, base = fixture(folder)
            original = base.read_bytes()
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUTF8="1")

            def run(args, success=True):
                result = subprocess.run([sys.executable, "-m", "streetgeo", "train", *args],
                                        capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
                return result

            common = ["--init-from", str(base), "--environment-specialist", "--unfreeze-last", "2",
                      "--train-manifest", str(folder / "train.jsonl"), "--val-manifest", str(folder / "val.jsonl"),
                      "--epochs", "1", "--batch-size", "2", "--device", "cpu", "--amp", "off",
                      "--num-workers", "0", "--log-every-steps", "1", "--save-every-steps", "1"]
            full = folder / "full.pt"
            first = run([*common, "--output", str(full)])
            events = [json.loads(line) for line in first.stdout.splitlines() if line.startswith("{")]
            updates = [event["input_view"] for event in events if "input_view" in event]
            self.assertEqual(updates, ["full", "environment"] * 3)
            self.assertTrue(any(event.get("event") == "baseline_evaluation" for event in events))
            metrics = next(event for event in events if "environment_validation" in event)
            self.assertIn("validation", metrics)
            best = read_checkpoint(full)
            baseline = next(event for event in events if event.get("event") == "baseline_evaluation")
            self.assertEqual(best["training_state"]["metrics"]["validation"]["median_km"],
                             min(baseline["full"]["median_km"], metrics["validation"]["median_km"]))
            partial = folder / "partial.pt"
            run([*common, "--output", str(partial), "--num-workers", "2", "--stop-after-steps", "1"])
            last = folder / "partial.last.pt"
            stopped = read_checkpoint(last)
            self.assertEqual(stopped["training_state"]["next_phase"], "environment")
            self.assertEqual(stopped["training_state"]["next_batch"], 1)
            resumed = run(["--resume", str(last), "--device", "cpu", "--num-workers", "2", "--log-every-steps", "1", "--stop-after-steps", "1"])
            resumed_events = [json.loads(line) for line in resumed.stdout.splitlines() if line.startswith("{")]
            self.assertEqual(next(event["input_view"] for event in resumed_events if "input_view" in event), "environment")
            self.assertFalse(any(event.get("event") == "baseline_evaluation" for event in resumed_events))
            self.assertEqual(read_checkpoint(last)["training_state"]["next_phase"], "full")
            resumed_again = run(["--resume", str(last), "--device", "cpu", "--num-workers", "0", "--log-every-steps", "1"])
            self.assertIn('"input_view": "full"', resumed_again.stdout)
            expected, actual = read_checkpoint(folder / "full.last.pt"), read_checkpoint(last)
            self.assertEqual(actual["training_state"]["global_step"], 6)
            self.assertEqual(actual["training_state"]["epoch"], 1)
            self.assertEqual(actual["training_state"]["environment_recipe"]["mask"], "oriented_source_bottom_trapezoid")
            self.assertEqual(actual["training_state"]["run_config"]["road_mask_top_width"], .2)
            self.assertEqual(actual["training_state"]["baseline_metrics"], expected["training_state"]["baseline_metrics"])
            for name, value in expected["model_state_dict"].items():
                torch.testing.assert_close(value, actual["model_state_dict"][name], rtol=0, atol=0)
            for key, values in expected["training_state"]["optimizer"]["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, actual["training_state"]["optimizer"]["state"][key][name], rtol=0, atol=0)
            initial = read_checkpoint(base)["model_state_dict"]
            for name in initial:
                if name.startswith(("encoder.patch_embedding.", "encoder.position_embedding", "encoder.visual_blocks.0.", "encoder.visual_blocks.1.")):
                    torch.testing.assert_close(actual["model_state_dict"][name], initial[name], rtol=0, atol=0)
            self.assertEqual(base.read_bytes(), original)
            rejected = run(["--resume", str(last), "--device", "cpu", "--no-environment-specialist"], success=False)
            self.assertIn("unchanged environment_specialist", rejected.stderr)
            rejected = run(["--resume", str(last), "--device", "cpu", "--road-mask-top-width", ".3"], success=False)
            self.assertIn("unchanged road_mask_top_width", rejected.stderr)

    def test_cli_rejects_missing_model_overlapping_validation_and_invalid_layer_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _, base = fixture(folder)
            command = [sys.executable, "-m", "streetgeo", "train", "--environment-specialist", "--device", "cpu",
                       "--train-manifest", str(folder / "train.jsonl"), "--output", str(folder / "rejected.pt")]
            cases = [([], "--init-from"), (["--init-from", str(base)], "验证清单"),
                     (["--init-from", str(base), "--val-manifest", str(folder / "train.jsonl")], "相同 Mapillary"),
                     (["--init-from", str(base), "--val-manifest", str(folder / "val.jsonl"), "--unfreeze-last", "99"], "unfreeze-last")]
            for args, message in cases:
                with self.subTest(message=message):
                    result = subprocess.run([*command, *args], capture_output=True, text=True, encoding="utf-8", timeout=90,
                                            env=dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertFalse((folder / "rejected.pt").exists())
                    self.assertFalse((folder / "rejected.last.pt").exists())

    def test_gui_mode_defaults_required_inputs_and_resume_locking(self):
        from streetgeo.ui.training import TrainingApp
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            cfg, base = fixture(folder)
            values = dict(DEFAULTS, environment_specialist="开启 · 全图/专项交替", init_from=str(base),
                          train_manifest=str(folder / "train.jsonl"), val_manifest=str(folder / "val.jsonl"),
                          output=str(folder / "new.pt"), model_config="", device="cpu", amp="off")
            args = build_training(values, folder / "stop")
            self.assertIn("--environment-specialist", args)
            self.assertIn("--visual-learning-rate", args)
            self.assertEqual(args[args.index("--road-mask-top-width") + 1], "0.2")
            self.assertEqual(args[args.index("--road-mask-bottom-width") + 1], "1.0")
            self.assertEqual(args[args.index("--road-mask-height") + 1], "0.45")
            for missing in ("init_from", "val_manifest"):
                with self.assertRaises(ValueError):
                    build_training(dict(values, **{missing: ""}), folder / "stop")
            root = tk.Tk()
            root.withdraw()
            app = TrainingApp(root, settings_path=folder / "gui.json", probe=False)
            try:
                app.values["environment_specialist"].set("开启 · 全图/专项交替")
                app._environment_changed()
                self.assertEqual(app.values["learning_rate"].get(), "0.00002")
                self.assertTrue(Path(app.values["output"].get()).name.startswith("environment"))
                self.assertEqual(str(app.editors["unfreeze_last"].cget("state")), "normal")
                with patch("streetgeo.ui.training.filedialog.askopenfilename", return_value=str(folder / "0.png")):
                    preview = app._preview_environment_mask()
                root.update_idletasks()
                self.assertTrue(preview.winfo_exists())
                self.assertEqual(preview.mask_config, RoadMask())
                app.values["road_mask_top_width"].set("30")
                self.assertEqual(preview.mask_config.top_width, .3)
                preview.destroy()
                self.assertIsNone(app.mask_preview_window)
                app.values["mode"].set("继续上次训练")
                app._mode_changed()
                self.assertEqual(str(app.editors["environment_specialist"].cget("state")), "disabled")
                self.assertEqual(str(app.editors["visual_learning_rate"].cget("state")), "disabled")
                self.assertEqual(str(app.editors["road_mask_top_width"].cget("state")), "disabled")
                app._structured({"event": "checkpoint_info", "resumable": True, "run_config": {
                                     "environment_specialist": True, "road_mask_top_width": .25,
                                     "road_mask_bottom_width": .9, "road_mask_height": .5},
                                 "environment_recipe": {"version": 2}, "step": 1, "epoch": 0, "next_batch": 1,
                                 "next_phase": "environment"})
                self.assertEqual(app.values["road_mask_top_width"].get(), "25")
                self.assertEqual(app.values["road_mask_height"].get(), "50")
                app._structured({"event": "validation_progress", "epoch": 1, "validation_view": "environment", "baseline": True,
                                 "batch": 1, "total_batches": 2, "samples": 1, "total_samples": 2})
                self.assertIn("基线", app.task_label.cget("text"))
                self.assertIn("专项", app.task_label.cget("text"))
            finally:
                for timer in root.tk.call("after", "info"):
                    root.after_cancel(timer)
                app._close()


if __name__ == "__main__":
    unittest.main()
