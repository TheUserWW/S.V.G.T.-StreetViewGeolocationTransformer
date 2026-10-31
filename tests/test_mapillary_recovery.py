from contextlib import closing, redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from streetgeo.datasets.download_mapillary import DownloadStore, collect, normalize_metadata
from streetgeo.datasets.mapillary_client import MapillaryClient, MapillaryHTTPError, RetryableDownloadError
from test_mapillary import FakeBoundary, FakeClient, args_for, metadata


def settings_for(args):
    result = {name: getattr(args, name) for name in (
        "region", "states", "bbox", "tile_size", "seed", "resolution", "min_side", "max_per_tile", "max_per_sequence")}
    result["boundary_fingerprint"] = FakeBoundary.fingerprint
    return result


class RecoveryTests(unittest.TestCase):
    def test_500_query_is_subdivided_and_children_run_before_other_roots(self):
        class Boundary(FakeBoundary):
            def tiles(self, *args):
                return [(-118.3, 34, -118.22, 34.08), (-118.4, 34, -118.32, 34.08)]

        class Client(FakeClient):
            def search(self, bbox, fields, after=None):
                self.pages.append((bbox, after))
                if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > 0.002:
                    raise MapillaryHTTPError(500, query_too_large=True)
                return [metadata(1), metadata(2)], None

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as output:
            client = Client()
            client.fail_once = False
            collect(args_for(folder), client, Boundary())
            self.assertEqual(len(client.pages), 2)
            bounds = client.pages[1][0]
            self.assertAlmostEqual(bounds[2] - bounds[0], 0.04)
            self.assertAlmostEqual(bounds[3] - bounds[1], 0.04)
            self.assertEqual(json.loads(output.getvalue().splitlines()[-1])["status"], "target_reached")
            with closing(sqlite3.connect(Path(folder) / "downloads.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT done FROM tiles WHERE id=0").fetchone()[0], 2)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM tiles WHERE depth=1 AND root_id=0").fetchone()[0], 4)

    def test_subdivision_preserves_previous_page_and_shares_parent_sample_cap(self):
        class Client(FakeClient):
            def search(self, bbox, fields, after=None):
                self.pages.append((bbox, after))
                if bbox == list(FakeBoundary().tiles()[0]):
                    if after is None:
                        return [metadata(1)], "page-two"
                    raise MapillaryHTTPError(500)
                return [metadata(1), metadata(2), metadata(3)], None

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            client = Client()
            client.fail_once = False
            collect(args_for(folder, max_images=10, max_per_tile=2), client, FakeBoundary())
            rows = [json.loads(line) for line in (Path(folder) / "all.jsonl").read_text().splitlines()]
            self.assertEqual({row["mapillary_id"] for row in rows}, {"1", "2"})
            self.assertEqual(client.downloads.count("1"), 1)
            self.assertEqual(client.pages[1][1], "page-two")
            self.assertTrue(all(after is None for _, after in client.pages[2:]))
            with closing(sqlite3.connect(Path(folder) / "downloads.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT SUM(selected) FROM tiles WHERE root_id=0").fetchone()[0], 2)
                self.assertEqual(db.execute("SELECT cursor FROM tiles WHERE id=0").fetchone()[0], "page-two")

    def test_deferred_cursor_survives_restart_and_other_regions_continue(self):
        class Boundary(FakeBoundary):
            def tiles(self, *args):
                return [(-118.3, 34, -118.295, 34.005), (-118.28, 34, -118.275, 34.005)]

        class Client(FakeClient):
            recovered = False

            def search(self, bbox, fields, after=None):
                self.pages.append((bbox, after))
                if bbox == list(Boundary().tiles()[0]):
                    if after is None:
                        return [metadata(1)], "keep-this-cursor"
                    if not self.recovered:
                        raise RetryableDownloadError("temporary connection failure")
                    return [metadata(3)], None
                return [metadata(2)], None

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            client = Client()
            client.fail_once = False
            collect(args_for(folder), client, Boundary())
            with closing(sqlite3.connect(Path(folder) / "downloads.sqlite3")) as db:
                cursor, done, retry_at = db.execute("SELECT cursor,done,retry_at FROM tiles WHERE id=0").fetchone()
                self.assertEqual(cursor, "keep-this-cursor")
                self.assertEqual(done, 0)
                self.assertGreater(retry_at, time.time())
            client.recovered = True
            with patch("streetgeo.datasets.download_mapillary.time.time", return_value=time.time() + 1000):
                collect(args_for(folder, max_images=3), client, Boundary())
            self.assertEqual(client.pages[-1][1], "keep-this-cursor")
            self.assertEqual(client.downloads.count("1"), 1)
            self.assertEqual(len((Path(folder) / "all.jsonl").read_text().splitlines()), 3)

    def test_service_outage_pauses_after_bounded_failures_without_losing_work(self):
        class Boundary(FakeBoundary):
            def tiles(self, *args):
                return [(-118.3 + i * 0.08, 34, -118.22 + i * 0.08, 34.08) for i in range(8)]

        class Client(FakeClient):
            def search(self, bbox, fields, after=None):
                self.pages.append(bbox)
                raise MapillaryHTTPError(503)

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as output:
            client = Client()
            collect(args_for(folder), client, Boundary())
            self.assertEqual(len(client.pages), 6)
            final = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual(final["status"], "paused")
            self.assertEqual(final["deferred_regions"], 6)
            with closing(sqlite3.connect(Path(folder) / "downloads.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM tiles WHERE done=0").fetchone()[0], 8)

    def test_rate_limit_defers_without_subdividing_or_immediately_retrying(self):
        class Client(FakeClient):
            def search(self, bbox, fields, after=None):
                self.pages.append(bbox)
                raise MapillaryHTTPError(429)

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            client = Client()
            collect(args_for(folder), client, FakeBoundary())
            collect(args_for(folder), client, FakeBoundary())
            self.assertEqual(len(client.pages), 1)
            with closing(sqlite3.connect(Path(folder) / "downloads.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM tiles").fetchone()[0], 1)

    def test_auth_failure_is_not_hidden_as_a_retryable_region(self):
        class Client(FakeClient):
            def search(self, bbox, fields, after=None):
                raise MapillaryHTTPError(401)

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            with self.assertRaises(MapillaryHTTPError):
                collect(args_for(folder), Client(), FakeBoundary())
            with closing(sqlite3.connect(Path(folder) / "downloads.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT failures,done FROM tiles").fetchone(), (0, 0))

    def test_legacy_database_migrates_without_losing_images_or_cursor(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            args = args_for(folder)
            (folder / "images").mkdir()
            image = folder / "images" / "1.jpg"
            content = FakeClient().content
            image.write_bytes(content)
            with closing(sqlite3.connect(folder / "downloads.sqlite3")) as db:
                db.executescript("""
                    CREATE TABLE settings(id INTEGER PRIMARY KEY,value TEXT NOT NULL);
                    CREATE TABLE tiles(id INTEGER PRIMARY KEY,bbox TEXT NOT NULL,cursor TEXT,
                                       done INTEGER DEFAULT 0,selected INTEGER DEFAULT 0);
                    CREATE TABLE images(id TEXT PRIMARY KEY,metadata TEXT NOT NULL,sequence_id TEXT NOT NULL,
                                        status INTEGER DEFAULT 0,bytes INTEGER DEFAULT 0,last_run TEXT,error TEXT);
                """)
                db.execute("INSERT INTO settings VALUES (1,?)", (json.dumps(settings_for(args), sort_keys=True),))
                db.execute("INSERT INTO tiles VALUES (17,?,?,0,1)", (json.dumps(FakeBoundary().tiles()[0]), "saved-page"))
                db.execute("INSERT INTO images(id,metadata,sequence_id,status,bytes) VALUES (?,?,?,1,?)",
                           ("1", json.dumps(normalize_metadata(metadata(1), FakeBoundary())), "seq1", len(content)))
                db.commit()
            store = DownloadStore(folder, settings_for(args), [])
            try:
                self.assertEqual(store.stats()["downloaded"], 1)
                self.assertEqual(store.db.execute("SELECT cursor,root_id,depth FROM tiles").fetchone(), ("saved-page", 17, 0))
                self.assertEqual(image.read_bytes(), content)
            finally:
                store.close()


class HttpRetryTests(unittest.TestCase):
    def test_query_size_error_is_classified_without_repeating_the_same_request(self):
        body = io.BytesIO(b'{"error":{"message":"Please reduce the amount of data; SECRET"}}')
        error = HTTPError("https://graph.mapillary.com/images?access_token=SECRET", 500, "failed", {}, body)
        client = MapillaryClient("SECRET", retries=5)
        with patch("streetgeo.datasets.mapillary_client.urlopen", side_effect=error) as opened:
            with self.assertRaises(MapillaryHTTPError) as caught:
                client.search((-118.3, 34, -118.22, 34.08), "id")
        self.assertEqual(opened.call_count, 1)
        self.assertTrue(caught.exception.query_too_large)
        self.assertNotIn("SECRET", str(caught.exception))

    def test_transient_http_retries_are_bounded_and_reported(self):
        reports = []
        client = MapillaryClient("SECRET", requests_per_second=100000, retries=2, on_retry=reports.append)
        errors = [HTTPError("https://graph.mapillary.com/images", 500, "failed", {}, io.BytesIO(b"{}")) for _ in range(3)]
        with patch("streetgeo.datasets.mapillary_client.urlopen", side_effect=errors) as opened, patch("streetgeo.datasets.mapillary_client.time.sleep"):
            with self.assertRaises(MapillaryHTTPError) as caught:
                client.search((-118.3, 34, -118.22, 34.08), "id")
        self.assertEqual(opened.call_count, 3)
        self.assertEqual(len(reports), 2)
        self.assertTrue(caught.exception.retryable)
        self.assertFalse(caught.exception.query_too_large)
        self.assertNotIn("SECRET", json.dumps(reports))


if __name__ == "__main__":
    unittest.main()
