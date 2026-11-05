import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streetgeo.datasets.download_mapillary import DownloadStore, add_collection_arguments, run_collection
from streetgeo.datasets.regions import SelectionBoundary, country_catalog
from streetgeo.datasets.mapillary_client import MapillaryHTTPError


class MapillaryRegionTests(unittest.TestCase):
    def test_country_continent_and_union_selection(self):
        self.assertGreater(len(country_catalog()), 200)
        japan = SelectionBoundary(countries=["JPN"])
        self.assertTrue(japan.contains(139.69, 35.68))
        self.assertFalse(japan.contains(2.35, 48.86))
        both = SelectionBoundary(continents=["asia"], countries=["FRA"])
        self.assertTrue(both.contains(139.69, 35.68))
        self.assertTrue(both.contains(2.35, 48.86))
        self.assertFalse(both.contains(-118.25, 34.05))

    def test_administrative_regions_bbox_geojson_and_islands(self):
        california = SelectionBoundary(regions=["US-CA"])
        self.assertTrue(california.contains(-118.25, 34.05))
        self.assertFalse(california.contains(-74.0, 40.71))
        bbox = (-119, 33, -118, 35)
        clipped = SelectionBoundary(countries=["USA"], bbox=bbox)
        self.assertTrue(clipped.contains(-118.25, 34.05))
        self.assertFalse(clipped.contains(-122.42, 37.77))
        usa = SelectionBoundary(countries=["USA"])
        self.assertTrue(usa.contains(-157.86, 21.31))  # Hawaii is a separate country feature.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "region.geojson"
            path.write_text(json.dumps({"type": "Polygon", "coordinates": [[[-119, 33], [-118, 33], [-118, 35], [-119, 35], [-119, 33]]]}))
            custom = SelectionBoundary(countries=["USA"], geojson=path)
            self.assertTrue(custom.contains(-118.25, 34.05))
            self.assertFalse(custom.contains(-122.42, 37.77))
            path.write_text('{"type":"Point","coordinates":[0,0]}')
            with self.assertRaisesRegex(ValueError, "Polygon"):
                SelectionBoundary(geojson=path)
        with self.assertRaises(ValueError):
            SelectionBoundary(countries=["JPN"], bbox=bbox)

    def test_coarse_tiles_expand_inside_selection_with_separate_fine_quotas(self):
        boundary = SelectionBoundary(regions=["US-CA"], bbox=(-119, 33, -118, 35))
        with tempfile.TemporaryDirectory() as folder:
            store = DownloadStore(folder, {}, boundary.tiles())
            try:
                tile = store.next_tile("run", 2, .08, boundary)
                bounds = json.loads(tile[1])
                self.assertLessEqual(max(bounds[2] - bounds[0], bounds[3] - bounds[1]), .08)
                self.assertTrue(boundary.intersects(bounds))
                root = store.db.execute("SELECT root_id FROM tiles WHERE id=?", (tile[0],)).fetchone()[0]
                self.assertEqual(root, tile[0])
                self.assertLess(store.db.execute("SELECT COUNT(*) FROM tiles").fetchone()[0], 100)
                with store.db:
                    store.db.execute("UPDATE tiles SET selected=2 WHERE id=?", (tile[0],))
                next_tile = store.next_tile("run", 2, .08, boundary)
                self.assertIsNotNone(next_tile)
                self.assertNotEqual(next_tile[0], tile[0])
            finally:
                store.close()

    def test_selection_changes_require_new_download_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            store = DownloadStore(folder, {"countries": ["JPN"]}, [(139, 35, 139.08, 35.08)])
            store.close()
            with self.assertRaisesRegex(ValueError, "settings changed"):
                DownloadStore(folder, {"countries": ["USA"]}, [])

    def test_coarse_groups_rotate_and_recovery_still_subdivides_deep_planning_tiles(self):
        boundary = SelectionBoundary(countries=["USA"])
        with tempfile.TemporaryDirectory() as folder:
            store = DownloadStore(folder, {}, [(-120, 34, -118, 36), (-100, 34, -98, 36)])
            try:
                first = store.next_tile("run", 30, .08, boundary)
                first_group = store.db.execute("SELECT queue_group FROM tiles WHERE id=?", (first[0],)).fetchone()[0]
                self.assertEqual(store.recover_query(first, MapillaryHTTPError(500), "run"), "split")
                second = store.next_tile("run", 30, .08, boundary)
                second_group = store.db.execute("SELECT queue_group FROM tiles WHERE id=?", (second[0],)).fetchone()[0]
                self.assertNotEqual(first_group, second_group)
                # Scheduling turns survive closing/reopening the persistent queue.
                store.close()
                store = DownloadStore(folder, {}, [])
                third = store.next_tile("next-run", 30, .08, boundary)
                third_group = store.db.execute("SELECT queue_group FROM tiles WHERE id=?", (third[0],)).fetchone()[0]
                self.assertEqual(third_group, first_group)
            finally:
                store.close()

    def test_list_regions_and_invalid_selection_do_not_need_network_or_token(self):
        parser = argparse.ArgumentParser()
        add_collection_arguments(parser)
        with patch("streetgeo.datasets.download_mapillary.MapillaryClient") as client:
            with redirect_stdout(io.StringIO()) as output:
                run_collection(parser.parse_args(["--list-regions"]))
            self.assertGreater(len(json.loads(output.getvalue())["countries"]), 200)
            with self.assertRaises(ValueError):
                run_collection(parser.parse_args(["--countries", "BAD"]))
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
