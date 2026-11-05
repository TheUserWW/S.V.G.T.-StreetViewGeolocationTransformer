from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from streetgeo.__main__ import main
from streetgeo.core.checkpoints import read_checkpoint
from test_model import tiny_config


class ApiStub:
    def search(self, bbox, fields, after=None):
        west, south, east, north = bbox
        return [{"id": str(index), "camera_type": "perspective", "is_pano": False,
                 "computed_geometry": {"type": "Point", "coordinates": [west + (east - west) * (index + 1) / 14,
                                                                              south + (north - south) * (index + 1) / 14]},
                 "sequence": f"sequence-{index}", "creator": {"id": "1"}}
                for index in range(12)], None

    def image_bytes(self, identifier, resolution):
        stream = io.BytesIO()
        Image.new("RGB", (300, 250), "orange").save(stream, format="JPEG")
        return stream.getvalue()


class DownloadTrainingWorkflowTests(unittest.TestCase):
    def test_download_split_train_and_saved_source_in_one_command(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            config = folder / "tiny.json"
            config.write_text(json.dumps(asdict(tiny_config())), encoding="utf-8")
            output = folder / "trained.pt"
            downloaded, prepared = folder / "download", folder / "prepared"
            command = ["train", "--download", "--countries", "USA", "--bbox", "-118.5", "34", "-118", "34.5",
                       "--download-output", str(downloaded), "--data-output", str(prepared), "--max-images", "12",
                       "--workers", "1", "--min-free-gb", "0", "--model-config", str(config),
                       "--output", str(output), "--epochs", "1", "--batch-size", "2", "--num-workers", "0",
                       "--device", "cpu", "--amp", "off"]
            with patch.dict(os.environ, {"MAPILLARY_ACCESS_TOKEN": "WORKFLOW_SECRET"}), \
                    patch("streetgeo.datasets.download_mapillary.MapillaryClient", return_value=ApiStub()), \
                    redirect_stdout(io.StringIO()) as captured:
                main(command)
            self.assertNotIn("WORKFLOW_SECRET", captured.getvalue())
            self.assertNotIn("WORKFLOW_SECRET", (downloaded / "all.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(len(list((downloaded / "images").glob("*.jpg"))), 12)
            self.assertTrue(output.is_file())
            state = read_checkpoint(folder / "trained.last.pt")["training_state"]
            self.assertEqual(state["dataset_source"], "Mapillary")
            self.assertEqual(Path(state["train_manifest"]), prepared / "train.jsonl")
            self.assertIn('"status": "target_reached"', captured.getvalue())

    def test_paused_collection_does_not_prepare_or_train(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "unstarted.pt"
            with patch("streetgeo.datasets.download_mapillary.run_collection", return_value="paused"), \
                    patch("streetgeo.datasets.prepare.prepare_datasets") as prepare, redirect_stdout(io.StringIO()):
                main(["train", "--download", "--countries", "JPN", "--device", "cpu", "--output", str(output)])
            prepare.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
