"""Country filtering against cached U.S. Census state polygons (50 states + DC)."""

import hashlib
import json
import math
import os
import random
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from shapely.geometry import Point, box, shape
from shapely import make_valid
from shapely.prepared import prep
from shapely.ops import unary_union

CENSUS_LAYER = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/0"


def fetch_boundaries(path):
    """Download public boundary metadata, not imagery. Cached until removed."""
    path = Path(path)
    if path.exists():
        return
    params = {"where": "STATE < '60'", "outFields": "STATE,STUSAB,NAME", "outSR": 4326,
              "f": "geojson", "geometryPrecision": 5, "maxAllowableOffset": 0.0001}
    with urlopen(CENSUS_LAYER + "/query?" + urlencode(params), timeout=120) as response:
        content = response.read(32 * 1024 * 1024)
    data = json.loads(content)
    codes = {feature["properties"]["STUSAB"] for feature in data.get("features", [])}
    if data.get("exceededTransferLimit") or len(codes) != 51:
        raise ValueError("Census response must contain all 50 states and DC")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


class USBoundary:
    def __init__(self, path, region="conus", states=None):
        content = Path(path).read_bytes()
        self.fingerprint = hashlib.sha256(content).hexdigest()
        features = json.loads(content).get("features", [])
        available = {feature["properties"]["STUSAB"] for feature in features}
        selected = set(states or available)
        if selected - available:
            raise ValueError(f"Unknown state codes: {sorted(selected - available)}")
        if region == "conus":
            selected -= {"AK", "HI"}
        # Coordinate rounding can create self-intersections in coastal rings.
        # Repair the authoritative polygons before the union, preserving holes.
        geometries = [make_valid(shape(feature["geometry"])) for feature in features
                      if feature["properties"]["STUSAB"] in selected]
        if not geometries:
            raise ValueError("No states selected")
        self.geometry = unary_union(geometries)
        self.prepared = prep(self.geometry)

    def contains(self, longitude, latitude):
        return self.prepared.covers(Point(longitude, latitude))

    def tiles(self, size=0.08, seed=1337, bbox=None):
        if not 0 < size <= 0.09:
            raise ValueError("tile size must be in (0, 0.09] degrees")
        geometry = self.geometry if bbox is None else self.geometry.intersection(box(*bbox))
        if geometry.is_empty:
            raise ValueError("Requested bbox does not intersect the selected U.S. states")
        parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
        tiles = set()
        # Component bounds avoid crossing the full longitude span of Alaska.
        for part in parts:
            west, south, east, north = part.bounds
            for col in range(math.floor(west / size), math.ceil(east / size)):
                for row in range(math.floor(south / size), math.ceil(north / size)):
                    bounds = (max(-180, col * size), max(-90, row * size),
                              min(180, (col + 1) * size), min(90, (row + 1) * size))
                    if part.intersects(box(*bounds)):
                        tiles.add(tuple(round(value, 7) for value in bounds))
        tiles = sorted(tiles)
        random.Random(seed).shuffle(tiles)
        return tiles
