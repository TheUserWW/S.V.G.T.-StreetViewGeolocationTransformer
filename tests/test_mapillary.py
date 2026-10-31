import argparse
from contextlib import closing, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

from PIL import Image

from streetgeo.datasets.download_mapillary import DownloadStore, ImageTooSmallError, collect, download_one, normalize_metadata
from streetgeo.datasets.mapillary_client import DownloadError, MapillaryClient, SafeRedirect, ThumbnailUnavailable
from streetgeo.datasets.split import create_splits, remove_nearby
from streetgeo.datasets.us_boundaries import USBoundary


class FakeBoundary:
    fingerprint = "test-boundary"

    def contains(self, lon, lat):
        return -125 < lon < -66 and 24 < lat < 49

    def tiles(self, *args):
        return [(-118.3, 34, -118.22, 34.08)]


def metadata(identifier, lon=-118.25, sequence=None):
    return {"id": str(identifier), "camera_type": "perspective", "is_pano": False,
            "computed_geometry": {"type": "Point", "coordinates": [lon, 34.05]},
            "compass_angle": 0, "sequence": sequence or f"seq{identifier}",
            "creator": {"id": "99", "username": "tester"}}


class FakeClient:
    def __init__(self):
        self.downloads = []
        self.pages = []
        self.fail_once = True
        buffer = io.BytesIO()
        Image.new("RGB", (300, 250), (20, 80, 120)).save(buffer, format="JPEG")
        self.content = buffer.getvalue()

    def search(self, bbox, fields, after=None):
        self.pages.append(after)
        if after is None:
            return [metadata(1), metadata(2), metadata(3, lon=-99), metadata(88, lon=-130)], "second"
        return [metadata(1), metadata(4)], None

    def image_bytes(self, identifier, resolution):
        self.downloads.append(identifier)
        if identifier == "2" and self.fail_once:
            self.fail_once = False
            raise DownloadError("simulated temporary network failure")
        return self.content


def args_for(folder, **overrides):
    values = dict(output=str(folder), region="conus", states=None, bbox=None,
                  tile_size=0.08, seed=1337, resolution=1024, min_side=224,
                  max_per_tile=30, max_per_sequence=50, max_images=2,
                  max_pages=20, workers=1, max_gb=2, min_free_gb=0)
    values.update(overrides)
    return argparse.Namespace(**values)


class DownloadTests(unittest.TestCase):
    @staticmethod
    def image_bytes(size):
        stream = io.BytesIO()
        Image.new("RGB", size, (30, 80, 120)).save(stream, format="JPEG")
        return stream.getvalue()

    def test_256_landscape_automatically_downloads_1024(self):
        small = self.image_bytes((256, 144))
        large = self.image_bytes((1024, 576))
        client = FakeClient()
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "images").mkdir()
            with patch.object(client, "image_bytes", side_effect=[small, large]) as fetch:
                result = download_one(client, "1", folder, 256, 224)
            self.assertEqual([call.args[1] for call in fetch.call_args_list], [256, 1024])
            self.assertEqual(result, len(large))
            with Image.open(Path(folder) / "images" / "1.jpg") as image:
                self.assertEqual(image.size, (1024, 576))
            # The same validated image is reused on restart, without network I/O.
            with patch.object(client, "image_bytes", side_effect=AssertionError("unnecessary download")):
                download_one(client, "1", folder, 256, 224)

    def test_missing_intermediate_thumbnail_can_fall_back_to_2048(self):
        client = FakeClient()
        responses = [self.image_bytes((256, 144)), ThumbnailUnavailable("missing variant"),
                     self.image_bytes((2048, 1152))]
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "images").mkdir()
            with patch.object(client, "image_bytes", side_effect=responses) as fetch:
                download_one(client, "1", folder, 256, 224)
            self.assertEqual([call.args[1] for call in fetch.call_args_list], [256, 1024, 2048])

    def test_network_failure_during_upgrade_is_not_a_permanent_skip(self):
        client = FakeClient()
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "images").mkdir()
            with patch.object(client, "image_bytes", side_effect=[self.image_bytes((256, 144)), DownloadError("offline")]):
                with self.assertRaises(DownloadError):
                    download_one(client, "1", folder, 256, 224)
            self.assertFalse((Path(folder) / "images" / "1.jpg").exists())

    def test_small_images_are_recorded_as_skipped_without_repeated_downloads(self):
        client = FakeClient()
        client.fail_once = False
        client.content = self.image_bytes((100, 75))
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as output:
            collect(args_for(folder, resolution=256), client, FakeBoundary())
            initial_count = len(client.downloads)
            self.assertGreater(initial_count, 0)
            collect(args_for(folder, resolution=256), client, FakeBoundary())
            self.assertEqual(len(client.downloads), initial_count)
            import sqlite3
            with closing(sqlite3.connect(Path(folder) / "downloads.sqlite3")) as database:
                self.assertEqual(database.execute("SELECT COUNT(*) FROM images WHERE status=2").fetchone()[0], 4)
                self.assertEqual(database.execute("SELECT COUNT(*) FROM images WHERE status=-1").fetchone()[0], 0)
            self.assertIn("100×75", output.getvalue())
            self.assertIn('"skipped": 4', output.getvalue())
            self.assertEqual((Path(folder) / "all.jsonl").read_text(), "")

    def test_legacy_min_side_failures_resume_at_larger_resolution(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            args = args_for(folder, resolution=256, max_images=1)
            settings = {name: getattr(args, name) for name in ("region", "states", "bbox", "tile_size", "seed",
                        "resolution", "min_side", "max_per_tile", "max_per_sequence")}
            settings["boundary_fingerprint"] = FakeBoundary.fingerprint
            store = DownloadStore(folder, settings, FakeBoundary().tiles())
            normalized = normalize_metadata(metadata(1), FakeBoundary())
            with store.db:
                store.db.execute("INSERT INTO images(id,metadata,sequence_id,status,error) VALUES (?,?,?,?,?)",
                                 ("1", json.dumps(normalized), "seq1", -1, "Image is smaller than min-side"))
            store.close()
            client = FakeClient()
            client.fail_once = False
            with patch.object(client, "image_bytes", return_value=self.image_bytes((1024, 576))) as fetch:
                collect(args, client, FakeBoundary())
            self.assertEqual(fetch.call_args.args, ("1", 1024))
            exported = json.loads((Path(folder) / "all.jsonl").read_text())
            self.assertNotIn("_resolution_hint", exported)

    def test_resumable_discovery_failures_duplicates_and_manifests(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            folder = Path(folder)
            client = FakeClient()
            collect(args_for(folder), client, FakeBoundary())
            rows = [json.loads(line) for line in (folder / "all.jsonl").read_text().splitlines()]
            self.assertEqual({row["mapillary_id"] for row in rows}, {"1", "3"})
            collect(args_for(folder, max_images=4), client, FakeBoundary())
            rows = [json.loads(line) for line in (folder / "all.jsonl").read_text().splitlines()]
            self.assertEqual({row["mapillary_id"] for row in rows}, {"1", "2", "3", "4"})
            self.assertEqual(client.pages, [None, "second"])
            self.assertEqual(client.downloads.count("1"), 1)
            self.assertEqual(client.downloads.count("2"), 2)
            self.assertEqual(len((folder / "attribution.jsonl").read_text().splitlines()), 4)
            # Repair a deleted local file without rediscovering metadata.
            (folder / "images" / "1.jpg").unlink()
            collect(args_for(folder, max_images=4), client, FakeBoundary())
            self.assertTrue((folder / "images" / "1.jpg").exists())
            self.assertEqual(client.downloads.count("1"), 2)

    def test_download_configuration_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            collect(args_for(folder), FakeClient(), FakeBoundary())
            with self.assertRaisesRegex(ValueError, "settings changed"):
                collect(args_for(folder, resolution=2048), FakeClient(), FakeBoundary())

    def test_panoramas_and_foreign_points_filtered(self):
        row = metadata(1)
        row["camera_type"] = "spherical"
        self.assertIsNone(normalize_metadata(row, FakeBoundary()))
        self.assertIsNone(normalize_metadata(metadata(2, lon=-130), FakeBoundary()))
        self.assertIsNotNone(normalize_metadata(metadata(3), FakeBoundary()))

    def test_truncated_image_is_not_committed(self):
        client = FakeClient()
        client.content = client.content[:-120]
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "images").mkdir()
            with self.assertRaises(OSError):
                download_one(client, "1", folder, 1024, 224)
            self.assertFalse((Path(folder) / "images" / "1.jpg").exists())

    def test_token_only_in_authorization_header_and_retry_after(self):
        client = MapillaryClient("TOP_SECRET", requests_per_second=100000)
        headers = {"Retry-After": "0"}
        error = HTTPError("https://graph.mapillary.com/images", 429, "limited", headers, None)
        response = io.BytesIO(b'{"data":[]}')
        with patch("streetgeo.datasets.mapillary_client.urlopen", side_effect=[error, response]) as opened, patch("streetgeo.datasets.mapillary_client.time.sleep"):
            client.search((-118.3, 34, -118.22, 34.08), "id")
        request = opened.call_args.args[0]
        self.assertNotIn("TOP_SECRET", request.full_url)
        self.assertEqual(request.get_header("Authorization"), "OAuth TOP_SECRET")
        error = HTTPError("https://graph.mapillary.com/?access_token=TOP_SECRET", 401, "no", {}, None)
        with patch("streetgeo.datasets.mapillary_client.urlopen", side_effect=error):
            with self.assertRaises(DownloadError) as caught:
                client.get("images")
        self.assertNotIn("TOP_SECRET", str(caught.exception))

    def test_expired_thumbnail_is_refreshed(self):
        client = MapillaryClient("secret")
        with patch.object(client, "get", side_effect=[{"thumb_1024_url": "https://cdn/old"},
                                                     {"thumb_1024_url": "https://cdn/new"}]), \
             patch.object(client, "_read", side_effect=[DownloadError("HTTP 403 from cdn"), b"image"]):
            self.assertEqual(client.image_bytes("1", 1024), b"image")

    def test_authorization_cannot_follow_a_cross_host_redirect(self):
        request = Request("https://graph.mapillary.com/images", headers={"Authorization": "OAuth SECRET"})
        with self.assertRaises(DownloadError):
            SafeRedirect().redirect_request(request, None, 302, "", {}, "https://another-host.example/")


class SplitTests(unittest.TestCase):
    def test_sequence_and_spatial_blocks_remain_in_one_split(self):
        rows = [{"mapillary_id": str(index), "sequence_id": f"s{index // 2}",
                 "latitude": 28 + (index // 2) * 0.5, "longitude": -120 + index * 0.001}
                for index in range(40)]
        splits, report = create_splits(rows)
        owners = {}
        for name, partition in splits.items():
            for row in partition:
                self.assertIn(owners.setdefault(row["sequence_id"], name), (name,))
        self.assertTrue(all(report["counts"].values()))
        self.assertEqual(create_splits(rows)[1], report)

    def test_guard_removes_close_samples_across_antimeridian(self):
        reference = [{"latitude": 0, "longitude": 179.999}]
        candidates = [{"latitude": 0, "longitude": -179.999}, {"latitude": 0, "longitude": 170}]
        self.assertEqual(remove_nearby(candidates, reference, 250), candidates[1:])

    def test_boundary_country_filter_and_tiles(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "boundary.json"
            path.write_text(json.dumps({"features": [{"properties": {"STUSAB": "CA"},
                "geometry": {"type": "Polygon", "coordinates": [
                    [[-118.3, 34], [-118.2, 34], [-118.2, 34.1], [-118.3, 34.1], [-118.3, 34]]
                ]}}]}))
            boundary = USBoundary(path)
            self.assertTrue(boundary.contains(-118.25, 34.05))
            self.assertFalse(boundary.contains(-130, 34.05))
            self.assertTrue(boundary.tiles())


if __name__ == "__main__":
    unittest.main()
