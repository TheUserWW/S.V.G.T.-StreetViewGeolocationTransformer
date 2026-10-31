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
from streetgeo.datasets.images import StreetViewDataset, collate_streetviews
from streetgeo.datasets.sources import PreparationCancelled, iter_gsv_cities
from streetgeo.ui.backend import DATA_SOURCES, DEFAULTS, build_prepare
from streetgeo.datasets.prepare import file_digest, prepare_datasets
from streetgeo.datasets.split import create_splits
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
    def test_historical_coordinate_jitter_does_not_merge_an_entire_city(self):
        rows = []
        for place in range(10):
            for offset in (-.0001, .0001):
                rows.append({"latitude": 40, "longitude": -120 + place * .02 + offset,
                             "source": "GSV-Cities", "city": "City", "place_id": str(place),
                             "group_id": f"gsv:City:{place}"})
        splits, report = create_splits(rows, block_degrees=.02, guard_m=0, stratify_cities=True)
        self.assertEqual(report["groups"], 10)
        self.assertTrue(all(splits.values()))
        owners = {}
        for name, part in splits.items():
            for row in part:
                self.assertEqual(owners.setdefault(row["group_id"], name), name)

    def test_city_stratification_represents_each_city_in_every_split(self):
        rows = []
        for city, latitude in (("North", 40), ("South", -30)):
            for place in range(20):
                rows.append({"latitude": latitude, "longitude": place,
                             "source": "GSV-Cities", "city": city, "group_id": f"{city}:{place}"})
        splits, _ = create_splits(rows, block_degrees=.02, guard_m=0, stratify_cities=True)
        for part in splits.values():
            self.assertEqual({row["city"] for row in part}, {"North", "South"})

    def test_gsv_handles_global_places_historical_views_and_filename_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = gsv_fixture(folder)
            report = {}
            rows = list(iter_gsv_cities(root, report=report))
            self.assertEqual(len(rows), 24)
            self.assertEqual(report["places"], 12)
            self.assertTrue(any(row["latitude"] < 0 and row["longitude"] > 0 for row in rows))
            self.assertEqual({row["headings"][0] for row in rows}, {5, 353})
            self.assertTrue(all("image" in row and "images" not in row for row in rows))
            self.assertTrue(all(Path(row["image"]).is_file() for row in rows))
            north = [row for row in rows if row["city"] == "North_City" and row["place_id"] == "0"]
            south = [row for row in rows if row["city"] == "South" and row["place_id"] == "0"]
            self.assertEqual(north[0]["group_id"], north[1]["group_id"])
            self.assertNotEqual(north[0]["group_id"], south[0]["group_id"])

    def test_incomplete_gsv_does_not_silently_skip_records(self):
        with tempfile.TemporaryDirectory() as folder:
            root = gsv_fixture(folder)
            next((root / "Images" / "North_City").glob("*.JPG")).unlink()
            with self.assertRaisesRegex(ValueError, "找不到对应图片"):
                list(iter_gsv_cities(root))

    def test_mixed_sources_keep_place_ids_together_and_rebase_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            gsv, mapillary = gsv_fixture(folder), mapillary_fixture(folder)
            output = Path(folder) / "混合 splits"
            report = prepare_datasets(roots=[gsv, mapillary], output_dir=output)
            self.assertEqual(report["source_counts"], {"GSV-Cities": 24, "Mapillary": 6})
            self.assertEqual(report["samples"], 30)
            owners = {}
            for name in ("train", "val", "test"):
                data = StreetViewDataset(output / f"{name}.jsonl", tiny_config())
                collated = collate_streetviews([data[0]])
                self.assertEqual(collated["images"].shape[-2:], (32, 32))
                for line in (output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    self.assertFalse(Path(row["image"]).is_absolute())
                    self.assertTrue((output / row["image"]).is_file())
                    key = row.get("group_id", f"mapillary:{row.get('sequence_id')}")
                    self.assertEqual(owners.setdefault(key, name), name)
            self.assertTrue(all(report["counts"].values()))

    def test_repeated_preparation_preserves_frozen_manifests_and_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = mapillary_fixture(folder)
            output = Path(folder) / "prepared"
            first = prepare_datasets(roots=[root], output_dir=output)
            stamp = (output / "train.jsonl").stat().st_mtime_ns
            second = prepare_datasets(roots=[root], output_dir=output, guard_m=250.0)
            self.assertEqual(first, second)
            self.assertEqual(stamp, (output / "train.jsonl").stat().st_mtime_ns)
            manifest = root / "all.jsonl"
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            rows[0]["latitude"] += .01
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "已有其他"):
                prepare_datasets(roots=[root], output_dir=output)
            self.assertEqual(file_digest(output / "train.jsonl"), first["manifest_sha256"]["train.jsonl"])

    def test_cancel_does_not_publish_partial_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = gsv_fixture(folder)
            output = Path(folder) / "cancelled"
            with self.assertRaises(PreparationCancelled):
                prepare_datasets(roots=[root], output_dir=output, cancelled=lambda: True)
            self.assertFalse(output.exists())

    def test_gui_builders_support_each_source_without_requiring_the_other(self):
        with tempfile.TemporaryDirectory() as folder:
            root, mapillary = gsv_fixture(folder), mapillary_fixture(folder)
            values = dict(DEFAULTS, gsv_root=str(root), split_output=str(Path(folder) / "out"),
                          split_manifest=str(mapillary / "all.jsonl"))
            gsv = build_prepare(dict(values, data_source=DATA_SOURCES[0], split_manifest="missing"), "stop")
            self.assertIn("--dataset-root", gsv)
            self.assertNotIn("--manifest", gsv)
            old = build_prepare(dict(values, data_source=DATA_SOURCES[1], gsv_root="missing"))
            self.assertIn("--manifest", old)
            self.assertNotIn("--dataset-root", old)
            mixed = build_prepare(dict(values, data_source=DATA_SOURCES[2]))
            self.assertIn("--dataset-root", mixed)
            self.assertIn("--manifest", mixed)

    def test_native_mixed_directory_training_and_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root, mapillary = gsv_fixture(folder), mapillary_fixture(folder)
            from test_osv5m import osv_fixture
            osv = osv_fixture(folder)
            folder = Path(folder)
            config = folder / "tiny.json"
            config.write_text(json.dumps(asdict(tiny_config(dropout=.1))), encoding="utf-8")
            output = folder / "world.pt"
            command = [sys.executable, "-m", "streetgeo", "train", "--dataset-root", str(root), "--dataset-root", str(mapillary), "--dataset-root", str(osv),
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
            self.assertEqual(read_checkpoint(checkpoint)["training_state"]["global_step"], 4)

    def test_gui_source_switch_preserves_resume_controls(self):
        import tkinter as tk
        from streetgeo.ui.training import TrainingApp

        with tempfile.TemporaryDirectory() as folder:
            root = tk.Tk()
            root.withdraw()
            app = TrainingApp(root, settings_path=Path(folder) / "gui.json", probe=False)
            try:
                app.values["data_source"].set(DATA_SOURCES[0])
                app._source_changed(reset_output=True)
                self.assertEqual(app.values["split_output"].get(), "data/world_gsv")
                self.assertEqual(str(app.editors["split_manifest"].cget("state")), "disabled")
                app.values["mode"].set("继续上次训练")
                app._lock_controls(True)
                app._lock_controls(False)
                self.assertEqual(str(app.editors["epochs"].cget("state")), "disabled")
                self.assertEqual(str(app.editors["gsv_root"].cget("state")), "normal")
                self.assertEqual(str(app.editors["split_manifest"].cget("state")), "disabled")
            finally:
                app._close()


if __name__ == "__main__":
    unittest.main()
