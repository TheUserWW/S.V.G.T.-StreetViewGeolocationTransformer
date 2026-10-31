"""Offline reference geography. Boundary geometry and styles come from UN OCHA."""
from functools import lru_cache
import json

from streetgeo.maps.projection import project
from streetgeo.paths import MAP_ASSETS


def _bounds(points):
    return (min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points))


def _point(coordinate):
    lon, lat = coordinate[:2]
    return project(max(-90, min(90, lat)), max(-180, min(180, lon)))


@lru_cache(maxsize=4)
def load_shapes(filename):
    """Load land fills/labels; an explicitly blank source label stays blank."""
    from shapely.geometry import shape

    path = MAP_ASSETS / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for feature in data["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        props = dict(feature["properties"])
        if "lbl_en" in props:
            props["name"] = props["lbl_en"] or ""
            point = shape(geometry).representative_point()
            props["label"] = [point.x, point.y]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            if not polygon:
                continue
            ring = [_point(coordinate) for coordinate in polygon[0]]
            if len(ring) >= 3:
                result.append((ring, _bounds(ring), props))
    return result


def projected_segments(coordinates):
    """Split at the date line instead of drawing a line across the whole map."""
    segment = []
    for coordinate in coordinates:
        point = _point(coordinate)
        if segment and abs(point[0] - segment[-1][0]) > 0.5:
            x, y = segment[-1]
            shifted_x = point[0] + (1 if point[0] < x else -1)
            seam = 1 if shifted_x >= 1 else 0
            fraction = (seam - x) / (shifted_x - x) if shifted_x != x else 0
            seam_y = y + fraction * (point[1] - y)
            segment.append((seam, seam_y))
            yield segment
            segment = [(1 - seam, seam_y)]
        segment.append(point)
    if len(segment) >= 2:
        yield segment


def boundary_style(code, source_styles):
    """Translate the source's ArcGIS line symbols to Tk line options."""
    info = source_styles[str(code)]  # Unknown types must not silently become solid borders.
    symbol = info["symbol"]
    color = "#" + "".join(f"{int(channel):02x}" for channel in symbol["color"][:3])
    dash = {"esriSLSSolid": (), "esriSLSDash": (5, 3), "esriSLSDot": (1, 3),
            "esriSLSDashDot": (5, 3, 1, 3)}[symbol["style"]]
    return {"fill": color, "width": max(0.5, float(symbol["width"])), "dash": dash}


@lru_cache(maxsize=1)
def load_boundaries():
    source = json.loads((MAP_ASSETS / "un_map_sources.json").read_text(encoding="utf-8"))
    data = json.loads((MAP_ASSETS / "un_boundaries.geojson").read_text(encoding="utf-8"))
    result = []
    for feature in data["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        props = dict(feature["properties"])
        props["style"] = boundary_style(props["bdytyp"], source["boundary_styles"])
        lines = geometry["coordinates"] if geometry["type"] == "MultiLineString" else [geometry["coordinates"]]
        for line in lines:
            for segment in projected_segments(line):
                result.append((segment, _bounds(segment), props))
    return result
