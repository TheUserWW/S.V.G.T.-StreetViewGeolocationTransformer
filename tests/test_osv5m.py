from dataclasses import asdict
import csv
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from streetgeo.datasets.osv5m import iter_osv5m, osv_layout
from streetgeo.datasets.prepare import prepare_datasets
from streetgeo.datasets.sources import PreparationCancelled, resolve_source
from streetgeo.ui.backend import DATA_SOURCES, DEFAULTS, build_prepare


def osv_fixture(folder, nested=True):
    root = Path(folder) / "OSV 部分数据"
    train = root / "train" if nested else root
    train.mkdir(parents=True)
    rows = []
    for index in range(14):
        identifier = str(9007199254740992 + index)
        rows.append({"id": identifier, "latitude": -25 + index * 3, "longitude": 20 + index * 4,
                     "sequence": f"sequence_{index // 2}", "country": "XY", "city": "Example"})
        if index < 12:
            shard = train / f"{index // 4:02d}"
            shard.mkdir(exist_ok=True)
            Image.new("RGB", (32, 32), (index * 10, 100, 200)).save(shard / f"{identifier}.JPG")
    with (train / "train.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    test = root / "test"
    test.mkdir()
    Image.new("RGB", (32, 32)).save(test / "official_test_only.jpg")
    return root


class PartialOSVTests(unittest.TestCase):
    def test_partial_download_matches_exact_string_ids_and_excludes_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = osv_fixture(folder)
            report = {}
            rows = list(iter_osv5m(root, report=report))
            self.assertEqual(len(rows), 12)
            self.assertEqual(report["metadata_rows"], 14)
            self.assertEqual(report["missing_images"], 2)
            self.assertEqual(report["local_images"], 12)
            self.assertEqual(rows[0]["osv5m_id"], "9007199254740992")
            self.assertEqual(rows[1]["osv5m_id"], "9007199254740993")
            self.assertEqual(rows[0]["sequence_id"], rows[1]["sequence_id"])
            self.assertTrue(all(Path(row["image"]).is_file() for row in rows))
            self.assertFalse(any("official_test" in row["image"] for row in rows))
            self.assertEqual(resolve_source(root), ("osv5m", root.resolve()))

    def test_root_csv_and_huggingface_image_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = osv_fixture(folder, nested=False)
            images = root / "images" / "train"
            images.mkdir(parents=True)
            for shard in ("00", "01", "02"):
                (root / shard).rename(images / shard)
            self.assertEqual(osv_layout(root)[1], images.resolve())
            self.assertEqual(len(list(iter_osv5m(root))), 12)

    def test_available_image_with_invalid_coordinate_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = osv_fixture(folder)
            path = root / "train" / "train.csv"
            text = path.read_text(encoding="utf-8").replace("9007199254740992,-25", "9007199254740992,nan")
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "经纬度"):
                list(iter_osv5m(root))

    def test_duplicate_local_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = osv_fixture(folder)
            Image.new("RGB", (32, 32)).save(root / "train" / "01" / "9007199254740992.jpg")
            with self.assertRaisesRegex(ValueError, "多个本地文件"):
                list(iter_osv5m(root))

    def test_cancel_and_zero_match_do_not_publish_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = osv_fixture(folder)
            output = Path(folder) / "prepared"
            with self.assertRaises(PreparationCancelled):
                prepare_datasets(roots=[root], output_dir=output, cancelled=lambda: True)
            self.assertFalse(output.exists())
            csv_path = root / "train" / "train.csv"
            csv_path.write_text("id,latitude,longitude\nabsent,0,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "没有匹配"):
                prepare_datasets(roots=[root], output_dir=output)
            self.assertFalse(output.exists())

    def test_sequence_isolation_report_and_new_shard_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = osv_fixture(folder)
            output = Path(folder) / "prepared"
            report = prepare_datasets(roots=[root], output_dir=output)
            self.assertEqual(report["samples"], 12)
            self.assertEqual(report["source_details"][0]["missing_images"], 2)
            owners = {}
            for name in ("train", "val", "test"):
                for line in (output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    self.assertTrue((output / row["image"]).is_file())
                    self.assertEqual(owners.setdefault(row["sequence_id"], name), name)
            self.assertEqual(prepare_datasets(roots=[root], output_dir=output), report)
            Image.new("RGB", (32, 32)).save(root / "train" / "02" / "9007199254741004.jpg")
            with self.assertRaisesRegex(ValueError, "已有其他"):
                prepare_datasets(roots=[root], output_dir=output)

    def test_gui_osv_source_does_not_require_other_datasets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = osv_fixture(folder)
            values = dict(DEFAULTS, data_source=DATA_SOURCES[0], osv_root=str(root),
                          gsv_root="missing", split_manifest="missing", split_output=str(Path(folder) / "out"))
            args = build_prepare(values)
            self.assertEqual(args[args.index("--dataset-root") + 1], str(root.resolve()))
            self.assertNotIn("--manifest", args)

    def test_gui_source_controls_and_partial_import_progress(self):
        import tkinter as tk
        from streetgeo.ui.training import TrainingApp
        with tempfile.TemporaryDirectory() as folder:
            root = tk.Tk()
            root.withdraw()
            app = TrainingApp(root, settings_path=Path(folder) / "settings.json", probe=False)
            try:
                app.values["data_source"].set(DATA_SOURCES[0])
                self.assertEqual(str(app.editors["osv_root"].cget("state")), "normal")
                self.assertNotIn("gsv_root", app.editors)
                self.assertNotIn("split_manifest", app.editors)
                self.assertNotIn("download", app.pages)
                self.assertEqual(app.values["split_output"].get(), "data/world_osv_514m")
                app._structured({"event": "dataset_import_progress", "stage": "metadata", "metadata_rows": 100,
                                 "local_images": 12, "samples": 8, "missing_images": 92})
                self.assertIn("92", app.detail_label.cget("text"))
                app._structured({"event": "dataset_import_complete", "samples": 12, "missing_images": 88})
                self.assertIn("88", app.split_summary.cget("text"))
            finally:
                app._close()


if __name__ == "__main__":
    unittest.main()
