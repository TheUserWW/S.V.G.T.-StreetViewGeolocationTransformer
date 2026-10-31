"""Regression checks for interactive rendering, label collisions, and layer reuse."""
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from streetgeo.maps.basemap import display_geometry, load_shapes
from streetgeo.maps.widgets import WorldMap
from streetgeo.ui.common import apply_dark_theme


class MapRenderingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = tk.Tk()
        apply_dark_theme(self.root)
        self.root.geometry("980x650")
        self.map = WorldMap(self.root, cache_folder=Path(self.folder.name))
        self.map.pack(fill="both", expand=True)
        self.root.update()
        self.map.focus(30, 10, 2)
        self.map._draw()

    def tearDown(self):
        self.map.close()
        self.root.destroy()
        self.folder.cleanup()

    def test_small_pan_reuses_geometry_and_preserves_coordinate_round_trip(self):
        widget = self.map
        items = widget.find_withtag("geography")
        original = widget.coords(items[0])
        center = widget.center
        widget.center = ((center[0] + 20 / widget.world_size) % 1, center[1])
        widget._draw()
        self.assertEqual(items, widget.find_withtag("geography"))
        self.assertAlmostEqual(widget.coords(items[0])[0], original[0] - 20)
        for lat, lon in ((34.05, -118.24), (0, 179.5), (0, -179.5)):
            actual = widget.from_screen(*widget.to_screen(lat, lon))
            self.assertAlmostEqual(actual[0], lat)
            self.assertAlmostEqual(actual[1], lon)

    def test_world_overview_fits_both_axes_and_resizes_without_losing_candidates(self):
        widget = self.map
        candidates = [{"latitude": lat, "longitude": lon} for lat, lon in
                      ((89.9, 179.9), (-89.9, -179.9), (-33.8, 151.2), (35, 139))]
        widget.candidates = candidates
        for width, height in ((980, 650), (400, 600), (1100, 380)):
            self.root.geometry(f"{width}x{height}")
            widget.show_world()
            self.root.update()
            self.assertTrue(widget._world_overview)
            for candidate in candidates:
                x, y = widget.to_screen(candidate["latitude"], candidate["longitude"])
                self.assertTrue(0 < x < widget.winfo_width())
                self.assertTrue(0 < y < widget.winfo_height() - 44)
            self.assertEqual(widget.candidates, candidates)

    def test_labels_use_actual_bounds_and_leave_space_between_names(self):
        widget = self.map
        for lat, lon, zoom in ((30, 10, 2), (48, 10, 4), (38, -98, 5)):
            widget.focus(lat, lon, zoom)
            widget._draw()
            boxes = [widget.bbox(item) for item in widget.find_withtag("map-label")]
            self.assertTrue(boxes)
            for index, (x0, y0, x1, y1) in enumerate(boxes):
                self.assertGreaterEqual(x0, 8)
                self.assertLessEqual(x1, widget.winfo_width() - 8)
                for a, b, c, d in boxes[index + 1:]:
                    self.assertFalse(x0 - 10 < c and x1 + 10 > a and y0 - 7 < d and y1 + 7 > b)

    def test_date_line_pan_reuses_shapes_without_a_world_width_jump(self):
        widget = self.map
        widget.focus(0, 179, 3)
        widget._draw()
        items = widget.find_withtag("geography")
        original = widget.coords(items[0])
        widget.focus(0, -179, 3)
        widget._draw()
        self.assertEqual(items, widget.find_withtag("geography"))
        self.assertAlmostEqual(widget.coords(items[0])[0] - original[0], -2 / 360 * widget.world_size)

    def test_drag_hides_labels_then_restores_and_relayouts_around_candidate(self):
        widget = self.map
        item = widget.find_withtag("map-label")[0]
        x, y = widget.coords(item)
        lat, lon = widget.from_screen(x, y)
        widget.candidates = [{"latitude": lat, "longitude": lon}]
        widget._draw()
        for other in widget.find_withtag("map-label"):
            a, b, c, d = widget.bbox(other)
            self.assertFalse(a < x + 24 and c > x - 24 and b < y + 24 and d > y - 24)
        widget._drag = (0, 0, widget.center, True)
        widget._draw()
        self.assertTrue(all(widget.itemcget(i, "state") == "hidden" for i in widget.find_withtag("map-label")))
        widget._drag = None
        widget._draw()
        self.assertTrue(widget.find_withtag("map-label"))
        self.assertTrue(all(widget.itemcget(i, "state") != "hidden" for i in widget.find_withtag("map-label")))

    def test_online_tile_arrival_and_toggle_preserve_geography(self):
        widget = self.map
        widget._cache = Mock(get=Mock(return_value=None))
        items = widget.find_withtag("geography")
        with patch.object(widget._requests, "put_nowait"):
            widget.set_online(True)
            widget._draw()
            self.assertEqual(items, widget.find_withtag("geography"))
            self.assertTrue(all(widget.itemcget(i, "state") == "hidden" for i in widget.find_withtag("map-label")))
            key = next(iter(widget._needed))
            widget._results.put((key, Image.new("RGB", (256, 256), "gray"), None))
            widget.after_cancel(widget._poll_id)
            widget._poll_id = None
            widget._poll()
            widget._draw()
            self.assertEqual(items, widget.find_withtag("geography"))
            tile_ids = widget.find_withtag("tile")
            self.assertTrue(tile_ids)
            widget._draw()
            self.assertEqual(tile_ids, widget.find_withtag("tile"))
            widget.set_online(False)
            widget._draw()
            self.assertFalse(widget.find_withtag("tile"))
            self.assertTrue(widget.find_withtag("map-label"))

    def test_zoom_rebuilds_geometry_and_keeps_special_boundary_dashes(self):
        old = self.map.find_withtag("geography")
        self.map.focus(35, 90, 4)
        self.map._draw()
        self.assertNotEqual(old, self.map.find_withtag("geography"))
        items = self.map.find_withtag("boundary-3")
        self.assertTrue(items)
        self.assertTrue(all(self.map.itemcget(i, "dash") for i in items))


class DisplayGeometryTests(unittest.TestCase):
    def test_display_cache_reduces_vertices_without_mutating_source(self):
        source = load_shapes("un_countries.geojson")
        before = sum(len(points) for points, _, _ in source)
        low = display_geometry("countries", 2)
        high = display_geometry("countries", 10)
        self.assertIs(low, display_geometry("countries", 2))
        self.assertLess(sum(len(p) for p, _, _ in low), before / 2)
        self.assertGreater(sum(len(p) for p, _, _ in high), sum(len(p) for p, _, _ in low))
        self.assertEqual(before, sum(len(points) for points, _, _ in source))


if __name__ == "__main__":
    unittest.main()
