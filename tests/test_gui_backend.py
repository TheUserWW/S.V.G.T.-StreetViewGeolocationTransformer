from dataclasses import asdict
import io
import json
from pathlib import Path
import os
import queue
import tempfile
import time
import unittest
from contextlib import redirect_stdout

from PIL import Image

from streetgeo.core.checkpoints import read_checkpoint
from streetgeo.datasets.download_mapillary import collect
from streetgeo.ui.backend import (DEFAULTS, MODEL_PRESETS, ProcessRunner, build_prepare, build_gpu_check, build_training,
                         load_settings, model_preset_for_config, new_model_output, save_settings)
from unittest.mock import patch
from test_model import tiny_config
from test_mapillary import FakeBoundary, FakeClient, args_for
from streetgeo.training.runtime import StopRequest


class GuiBackendTests(unittest.TestCase):
    def test_gpu_check_uses_selected_model_and_batch_size(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "selected model.json"
            config.write_text("{}", encoding="utf-8")
            values = dict(DEFAULTS, model_config=str(config), batch_size="8")
            self.assertEqual(build_gpu_check(values), ["--batch-size", "8", "--amp", "bf16", "--model-config", str(config)])
            values["batch_size"] = "0"
            with self.assertRaises(ValueError):
                build_gpu_check(values)

    def test_previous_settings_infer_model_size_from_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            for name, preset in MODEL_PRESETS.items():
                path.write_text(json.dumps({"model_config": preset["config"]}), encoding="utf-8")
                self.assertEqual(load_settings(path)["model_preset"], name)

    def test_model_output_avoids_existing_weights_and_resume_state(self):
        with tempfile.TemporaryDirectory() as folder, patch("streetgeo.ui.backend.ROOT", Path(folder)):
            for name, preset in MODEL_PRESETS.items():
                path = new_model_output(preset["config"])
                self.assertEqual(path.parent.name, preset["directory"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.with_name("best.last.pt").touch()
                self.assertEqual(new_model_output(preset["config"]).name, "best_2.pt")

    @unittest.skipUnless(os.name == "nt", "Windows naming rules")
    def test_invalid_output_rejected_before_manifest_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            values = dict(DEFAULTS, output=str(Path(folder) / "SVGT:4B.pt"), train_manifest="missing")
            with patch("streetgeo.datasets.policy.check_osv_manifest_header") as checked:
                with self.assertRaisesRegex(ValueError, "Windows"):
                    build_training(values, Path(folder) / "stop")
                checked.assert_not_called()

    def test_gui_switches_scales_without_changing_resume_architecture(self):
        import tkinter as tk
        from streetgeo.ui.training import TrainingApp
        with tempfile.TemporaryDirectory() as folder:
            root = tk.Tk()
            root.withdraw()
            app = TrainingApp(root, settings_path=Path(folder) / "settings.json", probe=False)
            try:
                for name, preset in MODEL_PRESETS.items():
                    app.values["init_from"].set("previous.pt")
                    app.values["model_preset"].set(name)
                    app._model_preset_changed()
                    self.assertEqual(app.values["model_config"].get(), preset["config"])
                    self.assertEqual(app.values["batch_size"].get(), preset["batch_size"])
                    self.assertEqual(model_preset_for_config(app.values["model_config"].get()), name)
                    self.assertEqual(Path(app.values["output"].get()).parent.name, preset["directory"])
                    self.assertEqual(app.values["init_from"].get(), "")
                app.values["mode"].set("继续上次训练")
                app._mode_changed()
                self.assertEqual(str(app.editors["model_preset"].cget("state")), "disabled")
                original_name, original = next(iter(MODEL_PRESETS.items()))
                original_config = json.loads((Path(__file__).resolve().parents[1] / original["config"]).read_text())
                app._structured({"event": "checkpoint_info", "resumable": True,
                                 "model_config": original_config, "run_config": {"batch_size": 128},
                                 "step": 9, "epoch": 0, "next_batch": 9})
                self.assertEqual(app.values["model_preset"].get(), original_name)
                self.assertEqual(app.values["model_config"].get(), original["config"])
                self.assertEqual(app.values["batch_size"].get(), "128")
                app._structured({"event": "validation_progress", "epoch": 2, "batch": 5,
                                 "total_batches": 10, "samples": 320, "total_samples": 640})
                self.assertIn("验证中", app.task_label.cget("text"))
                self.assertEqual(float(app.progress.cget("value")), 50)
                app._structured({"event": "training_phase", "phase": "saving"})
                self.assertIn("保存", app.task_label.cget("text"))
                app.progress.stop()
                app.values["mode"].set("新建训练")
                app._mode_changed()
                self.assertEqual(str(app.editors["model_preset"].cget("state")), "readonly")
                app.values["model_config"].set(str(Path(folder) / "custom.json"))
                self.assertNotIn(app.values["model_preset"].get(), MODEL_PRESETS)
            finally:
                for timer in root.tk.call("after", "info"):
                    root.after_cancel(timer)
                app._close()

    def test_preferences_never_store_access_token(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            values = dict(DEFAULTS, token="SECRET", MAPILLARY_ACCESS_TOKEN="SECRET", epochs="8")
            save_settings(path, values)
            self.assertNotIn("SECRET", path.read_text(encoding="utf-8"))
            self.assertEqual(load_settings(path)["epochs"], "8")

    def test_resume_uses_checkpoint_settings_without_overriding_schedule(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "model.last.pt"
            checkpoint.touch()
            values = dict(DEFAULTS, mode="继续上次训练", resume=str(checkpoint), epochs="invalid", batch_size="invalid")
            args = build_training(values, Path(folder) / "stop")
            self.assertIn("--resume", args)
            for flag in ("--epochs", "--batch-size", "--amp", "--train-manifest", "--output"):
                self.assertNotIn(flag, args)

    def test_file_paths_with_spaces_are_passed_as_single_arguments(self):
        with tempfile.TemporaryDirectory(prefix="训练 资料 ") as folder:
            manifest = Path(folder) / "训练 清单.jsonl"
            manifest.write_text(json.dumps({"image": "1.png", "osv5m_id": "1", "source": "OpenStreetView-5M", "latitude": 0, "longitude": 0}), encoding="utf-8")
            values = dict(DEFAULTS, train_manifest=str(manifest), val_manifest="", output=str(Path(folder) / "新模型.pt"),
                          device="cpu", amp="off")
            arguments = build_training(values, Path(folder) / "stop")
            self.assertEqual(arguments[arguments.index("--train-manifest") + 1], str(manifest))

    def test_invalid_config_rejected_before_starting_process(self):
        with self.assertRaises(ValueError):
            build_prepare(dict(DEFAULTS, data_source="Mapillary / JSONL", split_manifest="nonexistent.jsonl"))

    def test_worker_output_is_redacted_and_stop_file_is_cleaned(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            script = folder / "runner_test_worker.py"
            script.write_text(
                "import os,sys,time\nfrom pathlib import Path\n"
                "print(os.environ['MAPILLARY_ACCESS_TOKEN'],flush=True)\n"
                "while not Path(sys.argv[1]).exists(): time.sleep(0.02)\n"
                "print('STOPPED',flush=True)\n", encoding="utf-8")
            runner = ProcessRunner(workspace=folder / "work")
            stop = runner.new_stop_file()
            runner.start("test", "runner_test_worker", [str(stop)], stop_file=stop,
                         extra_env={"MAPILLARY_ACCESS_TOKEN": "DUMMY_SECRET", "PYTHONPATH": str(folder)})
            deadline = time.monotonic() + 15
            output, finished = [], None
            while time.monotonic() < deadline:
                event = runner.events.get(timeout=10)
                if event.kind == "line":
                    output.append(event.data)
                    if "令牌已隐藏" in event.data:
                        runner.request_stop()
                if event.kind == "finished":
                    finished = event.data
                    break
            runner.thread.join(timeout=5)
            self.assertEqual(finished["returncode"], 0)
            self.assertTrue(finished["stopped"])
            self.assertIn("STOPPED", output)
            self.assertNotIn("DUMMY_SECRET", runner.log_path.read_text(encoding="utf-8"))
            self.assertFalse(stop.exists())

    def test_download_stops_between_batches_and_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            stop = Path(folder) / "stop"
            stop.touch()
            client = FakeClient()
            collect(args_for(folder, stop_file=str(stop)), client, FakeBoundary())
            self.assertEqual(client.pages, [])
            self.assertTrue((Path(folder) / "all.jsonl").exists())

    def test_stop_request_polls_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stop"
            request = StopRequest(path)
            self.assertFalse(request.requested)
            path.touch()
            self.assertTrue(request.requested)

    def test_gui_training_stop_and_resume_reach_real_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = []
            for index in range(6):
                Image.new("RGB", (32, 32), (index * 30, 100, 140)).save(folder / f"{index}.png")
                rows.append({"image": f"{index}.png", "latitude": 30 + index, "longitude": -120 + index,
                             "source": "OpenStreetView-5M", "osv5m_id": str(index)})
            manifest = folder / "train.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            model_config = folder / "model.json"
            model_config.write_text(json.dumps(asdict(tiny_config(dropout=0.1))), encoding="utf-8")
            output = folder / "best.pt"
            settings = dict(DEFAULTS, train_manifest=str(manifest), val_manifest="", output=str(output),
                            model_config=str(model_config), epochs="200", batch_size="2", num_workers="0", device="cpu", amp="off")
            runner = ProcessRunner(workspace=folder / "work")
            stop = runner.new_stop_file()
            runner.start("train", "streetgeo.training.train", build_training(settings, stop), stop_file=stop)
            deadline, completed = time.monotonic() + 60, None
            while time.monotonic() < deadline:
                event = runner.events.get(timeout=40)
                if event.kind == "line":
                    try:
                        data = json.loads(event.data)
                    except ValueError:
                        continue
                    if "loss" in data and "step" in data:
                        runner.request_stop()
                if event.kind == "finished":
                    completed = event.data
                    break
            runner.thread.join(timeout=5)
            self.assertEqual(completed["returncode"], 0, runner.log_path.read_text(encoding="utf-8"))
            last = folder / "best.last.pt"
            state = read_checkpoint(last)["training_state"]
            self.assertEqual(state["reason"], "graceful_stop")
            self.assertGreaterEqual(state["global_step"], 1)
            self.assertLess(state["global_step"], 600)
            settings.update(mode="继续上次训练", resume=str(last))
            stop = runner.new_stop_file()
            runner.start("train", "streetgeo.training.train", build_training(settings, stop) + ["--stop-after-steps", "2"], stop_file=stop)
            runner.thread.join(timeout=60)
            self.assertFalse(runner.busy)
            resumed = read_checkpoint(last)["training_state"]
            self.assertEqual(resumed["global_step"], state["global_step"] + 2)


if __name__ == "__main__":
    unittest.main()
