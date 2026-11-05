from dataclasses import asdict
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from streetgeo.core.checkpoints import read_checkpoint
from streetgeo.datasets.images import StreetViewDataset
from streetgeo.datasets.prepare import prepare_datasets
from streetgeo.datasets.sources import PreparationCancelled, iter_manifest, resolve_source
from streetgeo.ui.backend import DATA_SOURCES, DEFAULTS, build_prepare, build_download, load_settings, save_settings
from test_model import tiny_config


def mapillary_fixture(folder):
    root = Path(folder) / "Mapillary 数据"
    (root / "images").mkdir(parents=True)
    rows = []
    for index in range(12):
        Image.new("RGB", (32, 32), (index * 15, 100, 150)).save(root / "images" / f"{index}.jpg")
        rows.append({"image": f"images/{index}.jpg", "latitude": 10 + index,
                     "longitude": -70 + index, "sequence_id": str(index // 2),
                     "mapillary_id": str(index), "source": "Mapillary", "headings": [None]})
    (root / "all.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return root


class DatasetSourceTests(unittest.TestCase):
    def test_split_command_reuses_frozen_pipeline_and_preserves_download_manifest(self):
        from streetgeo.datasets.split import prepare
        with tempfile.TemporaryDirectory() as folder:
            root = mapillary_fixture(folder)
            manifest = root / "all.jsonl"
            original = manifest.read_bytes()
            with redirect_stdout(io.StringIO()):
                report = prepare(manifest, block_degrees=.02)
                self.assertEqual(prepare(manifest, block_degrees=.02), report)
                with self.assertRaisesRegex(ValueError, "已有其他"):
                    prepare(manifest, block_degrees=.02, seed=42)
            self.assertEqual(manifest.read_bytes(), original)
            expected = prepare_datasets(roots=[root], output_dir=Path(folder) / "prepared", block_degrees=.02)
            self.assertEqual(report["preparation_fingerprint"], expected["preparation_fingerprint"])
            self.assertEqual(report["counts"], expected["counts"])
            self.assertEqual(report["source_counts"], {"Mapillary": 12})
            self.assertTrue((root / "splits/train.jsonl").is_file())

    def test_prepare_mapillary_is_frozen_relative_and_sequence_isolated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = mapillary_fixture(folder)
            output = Path(folder) / "prepared"
            report = prepare_datasets(roots=[root], output_dir=output)
            self.assertEqual(report["source_counts"], {"Mapillary": 12})
            self.assertEqual(prepare_datasets(roots=[root], output_dir=output), report)
            self.assertEqual(resolve_source(root), ("mapillary", root.resolve() / "all.jsonl"))
            owners = {}
            for name in ("train", "val", "test"):
                for row in map(json.loads, (output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()):
                    self.assertTrue((output / row["image"]).is_file())
                    self.assertEqual(owners.setdefault(row["sequence_id"], name), name)
            with (root / "all.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("\n" + json.dumps(dict(next(iter_manifest(root / "all.jsonl")), latitude=0)))
            with self.assertRaisesRegex(ValueError, "冲突"):
                prepare_datasets(roots=[root], output_dir=output)

    def test_prepare_rejects_osv_gsv_generic_and_mixed_sources_before_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            root = mapillary_fixture(folder)
            unsupported = Path(folder) / "unsupported"
            unsupported.mkdir()
            (unsupported / "train.csv").write_text("id,latitude,longitude", encoding="utf-8")
            (unsupported / "Images").mkdir()
            (unsupported / "Dataframes").mkdir()
            output = Path(folder) / "rejected"
            for roots in ([unsupported], [root, unsupported]):
                with self.assertRaisesRegex(ValueError, "Mapillary"):
                    prepare_datasets(roots=roots, output_dir=output)
                self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "Mapillary"):
                prepare_datasets(manifests=[root / "all.jsonl"], output_dir=output)
            rows = list(iter_manifest(root / "all.jsonl"))
            rows[-1]["source"] = "OpenStreetView-5M"
            (root / "all.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Mapillary"):
                prepare_datasets(roots=[root], output_dir=output)
            self.assertFalse(output.exists())

    def test_cancel_and_missing_images_do_not_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            root = mapillary_fixture(folder)
            output = Path(folder) / "prepared"
            with self.assertRaises(PreparationCancelled):
                prepare_datasets(roots=[root], output_dir=output, cancelled=lambda: True)
            (root / "images/0.jpg").unlink()
            with self.assertRaisesRegex(ValueError, "图片不存在"):
                prepare_datasets(roots=[root], output_dir=output)
            self.assertFalse(output.exists())

    def test_training_loader_checks_source_and_identity_at_every_line(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = list(iter_manifest(mapillary_fixture(folder) / "all.jsonl"))
            manifest = Path(folder) / "mixed.jsonl"
            for invalid in (dict(rows[1], source="GSV-Cities"), dict(rows[1], source="OpenStreetView-5M"),
                            dict(rows[1], mapillary_id="wrong"), dict(rows[1], osv5m_id="1"),
                            {k: v for k, v in rows[1].items() if k != "source"}):
                manifest.write_text("\n".join(map(json.dumps, [rows[0], invalid])), encoding="utf-8")
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Mapillary"):
                    StreetViewDataset(manifest, tiny_config(), mapillary_only=True)

    def test_native_mapillary_training_resume_and_rejected_osv_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = mapillary_fixture(folder)
            folder = Path(folder)
            config = folder / "tiny.json"
            config.write_text(json.dumps(asdict(tiny_config(dropout=.1))), encoding="utf-8")
            output = folder / "world.pt"
            command = [sys.executable, "-m", "streetgeo", "train", "--dataset-root", str(root),
                       "--data-output", str(folder / "data"), "--model-config", str(config),
                       "--output", str(output), "--epochs", "3", "--batch-size", "2", "--num-workers", "2",
                       "--device", "cpu", "--amp", "off", "--stop-after-steps", "2"]
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUTF8="1")
            first = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            checkpoint = folder / "world.last.pt"
            self.assertEqual(read_checkpoint(checkpoint)["training_state"]["global_step"], 2)
            resumed = subprocess.run([sys.executable, "-m", "streetgeo", "train", "--resume", str(checkpoint),
                                      "--device", "cpu", "--num-workers", "2", "--stop-after-steps", "2"],
                                     capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            saved = read_checkpoint(checkpoint)
            self.assertEqual(saved["training_state"]["global_step"], 4)
            self.assertEqual(saved["training_state"]["dataset_source"], "Mapillary")
            import torch
            saved["training_state"]["dataset_source"] = "OpenStreetView-5M"
            legacy = folder / "osv.last.pt"
            torch.save(saved, legacy)
            before = checkpoint.read_bytes()
            rejected = subprocess.run([sys.executable, "-m", "streetgeo", "train", "--resume", str(legacy), "--device", "cpu"],
                                      capture_output=True, text=True, encoding="utf-8", env=env, timeout=90)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Mapillary", rejected.stderr)
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_validation_cannot_bypass_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = list(iter_manifest(mapillary_fixture(folder) / "all.jsonl"))
            folder = Path(folder)
            train, val = folder / "train.jsonl", folder / "val.jsonl"
            train.write_text(json.dumps(rows[0]), encoding="utf-8")
            val.write_text(json.dumps(dict(rows[1], source="OpenStreetView-5M")), encoding="utf-8")
            config = folder / "tiny.json"
            config.write_text(json.dumps(asdict(tiny_config())), encoding="utf-8")
            output = folder / "rejected.pt"
            result = subprocess.run([sys.executable, "-m", "streetgeo", "train", "--train-manifest", str(train),
                                     "--val-manifest", str(val), "--model-config", str(config), "--device", "cpu", "--output", str(output)],
                                    capture_output=True, text=True, encoding="utf-8", timeout=90,
                                    env=dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="2"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Mapillary", result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name("rejected.last.pt").exists())

    def test_gui_selection_token_settings_and_resume_controls(self):
        import tkinter as tk
        from streetgeo.ui.training import TrainingApp
        with tempfile.TemporaryDirectory() as folder:
            root = tk.Tk()
            root.withdraw()
            settings = Path(folder) / "gui.json"
            app = TrainingApp(root, settings_path=settings, probe=False)
            try:
                self.assertEqual(set(app.pages), {"download", "split", "train"})
                self.assertNotIn("osv_root", app.editors)
                app.region_tree.selection_set(("continent:asia", "country:FRA", "region:US-CA"))
                app._region_selection_changed()
                self.assertEqual(app.values["download_continents"].get(), "asia")
                self.assertEqual(app.values["download_countries"].get(), "FRA")
                self.assertEqual(app.values["download_regions"].get(), "US-CA")
                app.values["token"].set("TEST_SECRET")
                app._persist()
                self.assertNotIn("TEST_SECRET", settings.read_text(encoding="utf-8"))
                self.assertEqual(load_settings(settings)["download_countries"], "FRA")
                app.values["mode"].set("继续上次训练")
                app._lock_controls(True)
                self.assertTrue(all(str(editor.cget("state")) == "disabled" for editor in app.all_editors))
                app._lock_controls(False)
                self.assertEqual(str(app.editors["epochs"].cget("state")), "disabled")
                self.assertEqual(str(app.editors["mapillary_root"].cget("state")), "normal")
            finally:
                for timer in root.tk.call("after", "info"):
                    root.after_cancel(timer)
                app._close()

    def test_download_options_are_validated_and_token_is_not_an_argument(self):
        args = build_download(dict(DEFAULTS, token="SECRET", download_countries="CHN,JPN", download_bbox="130 30 145 45"))
        self.assertEqual(args[args.index("--countries") + 1], "CHN,JPN")
        self.assertNotIn("SECRET", str(args))
        for values in (dict(DEFAULTS, download_countries="INVALID"), dict(DEFAULTS, download_bbox="nan 0 1 1"),
                       dict(DEFAULTS, max_gb="nan"), dict(DEFAULTS, download_countries="")):
            with self.assertRaises(ValueError):
                build_download(values)

    def test_old_osv_preferences_migrate_to_new_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps({"data_source": "OpenStreetView-5M", "osv_root": "old", "split_output": "old", "mode": "继续上次训练"}), encoding="utf-8")
            saved = load_settings(path)
            self.assertEqual(saved["split_output"], DEFAULTS["split_output"])
            self.assertEqual(saved["mode"], "新建训练")
            self.assertNotIn("osv_root", saved)
            for source in ("OpenStreetView-5M", "GSV-Cities", "custom"):
                with self.assertRaisesRegex(ValueError, "Mapillary"):
                    build_prepare(dict(DEFAULTS, data_source=source))


if __name__ == "__main__":
    unittest.main()
