from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from streetgeo.core.checkpoints import read_checkpoint
from streetgeo.datasets.images import StreetViewDataset
from streetgeo.ui.backend import DATA_SOURCES, DEFAULTS, build_prepare
from streetgeo.datasets.prepare import prepare_datasets
from test_model import tiny_config


def gsv_fixture(folder):
    root = Path(folder) / "街景 GSV"
    (root / "Dataframes").mkdir(parents=True)
    for city, initial_lat, initial_lon in (("North_City", 30, -120), ("South", -40, 140)):
        directory = root / "Images" / city
        directory.mkdir(parents=True)
        rows = []
        for place in range(6):
            for historical in range(2):
                lat, lon = initial_lat + place * .8 + historical * .00001, initial_lon + place
                heading = 725 if historical == 0 else -7
                panoid = f"pan_o_{city}_{place}_{historical}"
                # Different decimal formatting is intentional: float-to-string
                # reconstruction from CSV must not be used to locate files.
                name = f"{city}_{place:07d}_{2016+historical}_05_{heading:03d}_{lat}_{lon}_{panoid}.JPG"
                Image.new("RGB", (48, 32), (50 + place * 20, 90, 140)).save(directory / name)
                rows.append({"place_id": place, "year": 2016 + historical, "month": 5,
                             "northdeg": heading, "city_id": city, "lat": f"{lat:.10f}",
                             "lon": f"{lon:.10f}", "panoid": panoid})
        with (root / "Dataframes" / f"{city}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return root


def mapillary_fixture(folder):
    root = Path(folder) / "Mapillary 数据"
    (root / "images").mkdir(parents=True)
    rows = []
    for index in range(6):
        Image.new("RGB", (32, 32), "orange").save(root / "images" / f"{index}.jpg")
        rows.append({"image": f"images/{index}.jpg", "latitude": 10 + index,
                     "longitude": -70 + index, "sequence_id": str(index // 2),
                     "mapillary_id": str(index), "source": "Mapillary", "headings": [None]})
    (root / "all.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return root


class DatasetSourceTests(unittest.TestCase):
    def test_prepare_rejects_gsv_mapillary_jsonl_and_mixed_sources(self):
        from test_osv5m import osv_fixture
        with tempfile.TemporaryDirectory() as folder:
            gsv, mapillary, osv = gsv_fixture(folder), mapillary_fixture(folder), osv_fixture(folder)
            output = Path(folder) / "rejected"
            for roots in ([gsv], [mapillary], [osv, gsv], [osv, mapillary]):
                with self.subTest(roots=roots), self.assertRaises(ValueError):
                    prepare_datasets(roots=roots, output_dir=output)
                self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "OpenStreetView-5M"):
                prepare_datasets(manifests=[mapillary / "all.jsonl"], output_dir=output)

    def test_gui_rejects_old_source_selections(self):
        for source in ("GSV-Cities（全球）", "Mapillary / JSONL", "混合数据（全球）"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "OpenStreetView-5M"):
                build_prepare(dict(DEFAULTS, data_source=source))
        self.assertEqual(len(DATA_SOURCES), 1)

    def test_training_loader_rejects_non_osv_records_at_any_line(self):
        from streetgeo.datasets.osv5m import iter_osv5m
        from test_osv5m import osv_fixture
        with tempfile.TemporaryDirectory() as folder:
            rows = list(iter_osv5m(osv_fixture(folder)))
            manifest = Path(folder) / "mixed.jsonl"
            for invalid in (dict(rows[1], source="GSV-Cities"), dict(rows[1], source="Mapillary"),
                            dict(rows[1], osv5m_id="wrong"), {k:v for k,v in rows[1].items() if k != "source"}):
                manifest.write_text("\n".join(json.dumps(row) for row in [rows[0], invalid]), encoding="utf-8")
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "OpenStreetView-5M"):
                    StreetViewDataset(manifest, tiny_config(), osv5m_only=True)

    def test_native_osv_training_resume_and_rejected_legacy_checkpoint(self):
        from test_osv5m import osv_fixture
        from streetgeo.core.checkpoints import save_checkpoint
        from streetgeo.core.model import StreetGeoModel
        from streetgeo.training.runtime import manifest_fingerprint
        with tempfile.TemporaryDirectory() as folder:
            osv = osv_fixture(folder)
            folder = Path(folder)
            config = folder / "tiny.json"
            cfg = tiny_config(dropout=.1)
            config.write_text(json.dumps(asdict(cfg)), encoding="utf-8")
            output = folder / "world.pt"
            command = [sys.executable, "-m", "streetgeo", "train", "--dataset-root", str(osv),
                       "--data-output", str(folder / "data"), "--model-config", str(config),
                       "--output", str(output), "--epochs", "3", "--batch-size", "2", "--num-workers", "2",
                       "--device", "cpu", "--amp", "off", "--stop-after-steps", "2"]
            environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUTF8="1")
            first = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=environment, timeout=90)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            checkpoint = folder / "world.last.pt"
            self.assertEqual(read_checkpoint(checkpoint)["training_state"]["global_step"], 2)
            resumed = subprocess.run([sys.executable, "-m", "streetgeo", "train", "--resume", str(checkpoint), "--device", "cpu",
                                      "--num-workers", "2", "--stop-after-steps", "2"], capture_output=True,
                                     text=True, encoding="utf-8", env=environment, timeout=90)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            saved = read_checkpoint(checkpoint)
            self.assertEqual(saved["training_state"]["global_step"], 4)
            self.assertEqual(saved["training_state"]["dataset_source"], "OpenStreetView-5M")
            state = saved["training_state"]
            rows = [json.loads(line) for line in Path(state["train_manifest"]).read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row["image"] = str((Path(state["train_manifest"]).parent / row["image"]).resolve())
                row["source"] = "Mapillary"
            legacy_manifest = folder / "old_mapillary.jsonl"
            legacy_manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            state["train_manifest"] = str(legacy_manifest)
            state["train_fingerprint"] = manifest_fingerprint(legacy_manifest)
            state.pop("dataset_source", None)
            legacy = folder / "old.last.pt"
            model = StreetGeoModel(cfg)
            model.load_state_dict(saved["model_state_dict"])
            save_checkpoint(model, legacy, training_state=state)
            before = checkpoint.read_bytes()
            rejected = subprocess.run([sys.executable, "-m", "streetgeo", "train", "--resume", str(legacy), "--device", "cpu"],
                                      capture_output=True, text=True, encoding="utf-8", env=environment, timeout=90)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("OpenStreetView-5M", rejected.stderr)
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_validation_manifest_cannot_bypass_dataset_policy(self):
        from streetgeo.datasets.osv5m import iter_osv5m
        from test_osv5m import osv_fixture
        with tempfile.TemporaryDirectory() as folder:
            rows = list(iter_osv5m(osv_fixture(folder)))
            folder = Path(folder)
            train = folder / "train.jsonl"
            validation = folder / "val.jsonl"
            train.write_text(json.dumps(rows[0]), encoding="utf-8")
            validation.write_text(json.dumps(dict(rows[1], source="Mapillary")), encoding="utf-8")
            config = folder / "tiny.json"
            config.write_text(json.dumps(asdict(tiny_config())), encoding="utf-8")
            output = folder / "rejected.pt"
            result = subprocess.run([sys.executable, "-m", "streetgeo", "train", "--train-manifest", str(train),
                                     "--val-manifest", str(validation), "--model-config", str(config),
                                     "--device", "cpu", "--output", str(output)], capture_output=True,
                                    text=True, encoding="utf-8", timeout=60,
                                    env=dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="2"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("OpenStreetView-5M", result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name("rejected.last.pt").exists())

    def test_gui_has_only_osv_import_and_preserves_resume_controls(self):
        import tkinter as tk
        from streetgeo.ui.training import TrainingApp
        with tempfile.TemporaryDirectory() as folder:
            root = tk.Tk()
            root.withdraw()
            app = TrainingApp(root, settings_path=Path(folder) / "gui.json", probe=False)
            try:
                self.assertEqual(set(app.pages), {"split", "train"})
                self.assertNotIn("gsv_root", app.editors)
                self.assertNotIn("split_manifest", app.editors)
                app.values["mode"].set("继续上次训练")
                app._lock_controls(True)
                app._lock_controls(False)
                self.assertEqual(str(app.editors["epochs"].cget("state")), "disabled")
                self.assertEqual(str(app.editors["osv_root"].cget("state")), "normal")
            finally:
                app._close()


if __name__ == "__main__":
    unittest.main()
