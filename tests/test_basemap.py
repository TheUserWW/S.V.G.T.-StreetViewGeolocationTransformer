import hashlib
import json
import math
import unittest

from streetgeo.maps.basemap import boundary_style, load_boundaries, load_shapes, projected_segments
from streetgeo.paths import MAP_ASSETS


class ReferenceMapTests(unittest.TestCase):
    def test_bundled_source_files_match_recorded_downloads(self):
        source = json.loads((MAP_ASSETS / "un_map_sources.json").read_text(encoding="utf-8"))
        self.assertTrue(source["service"].startswith("https://gis.unocha.org/"))
        for name, expected in source["files"].items():
            payload = (MAP_ASSETS / name).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected["sha256"])
            self.assertEqual(len(json.loads(payload)["features"]), expected["features"])
        countries = json.loads((MAP_ASSETS / "un_countries.geojson").read_text(encoding="utf-8"))
        codes = {feature["properties"]["iso3cd"] for feature in countries["features"]}
        self.assertTrue({"CHN", "HKG", "MAC", "TWN"} <= codes)

    def test_source_blank_labels_are_not_replaced_with_country_names(self):
        original = json.loads((MAP_ASSETS / "un_countries.geojson").read_text(encoding="utf-8"))
        expected = {feature["properties"]["objectid"]: feature["properties"]["lbl_en"] or ""
                    for feature in original["features"]}
        shapes = load_shapes("un_countries.geojson")
        self.assertTrue(shapes)
        for _, _, props in shapes:
            self.assertEqual(props["name"], expected[props["objectid"]])

    def test_special_boundaries_keep_source_dash_styles(self):
        source = json.loads((MAP_ASSETS / "un_map_sources.json").read_text(encoding="utf-8"))
        self.assertFalse(boundary_style(1, source["boundary_styles"])["dash"])
        self.assertTrue(boundary_style(2, source["boundary_styles"])["dash"])
        self.assertTrue(boundary_style(3, source["boundary_styles"])["dash"])
        with self.assertRaises(KeyError):
            boundary_style(123456, source["boundary_styles"])
        lines = load_boundaries()
        self.assertTrue(any(props["bdytyp"] == 3 and props["style"]["dash"] for _, _, props in lines))
        self.assertTrue(any("CHN" in (props.get("iso3cd") or "") for _, _, props in lines))
        for line, _, _ in lines:
            self.assertTrue(all(math.isfinite(v) for point in line for v in point))
            self.assertTrue(all(abs(a[0] - b[0]) <= .5 for a, b in zip(line, line[1:])))

    def test_date_line_crossing_is_clipped_at_both_world_edges(self):
        parts = list(projected_segments([(170, 30), (-170, 30)]))
        self.assertEqual(len(parts), 2)
        self.assertAlmostEqual(parts[0][-1][0], 1)
        self.assertAlmostEqual(parts[1][0][0], 0)
        self.assertAlmostEqual(parts[0][-1][1], parts[1][0][1])


if __name__ == "__main__":
    unittest.main()
