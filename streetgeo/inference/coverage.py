"""Offline country/region approximation of Google-operated Street View coverage.

This is an allowlist, not a road-level coverage map or an API availability check.
Geometries retain holes and separate territories from the bundled UN boundaries.
"""

from functools import lru_cache
import json

from streetgeo.paths import MAP_ASSETS


@lru_cache(maxsize=1)
def coverage_metadata():
    data = json.loads((MAP_ASSETS / "google_streetview_regions.json").read_text(encoding="utf-8"))
    if data.get("scope") != "country_region_approximation" or not data.get("iso3_codes"):
        raise ValueError("Invalid Google Street View region allowlist")
    return data


@lru_cache(maxsize=1)
def _coverage_geometry():
    from shapely import make_valid, prepare, union_all
    from shapely.geometry import shape

    allowed = set(coverage_metadata()["iso3_codes"])
    source = json.loads((MAP_ASSETS / "un_countries.geojson").read_text(encoding="utf-8"))
    geometries, found = [], set()
    for feature in source["features"]:
        code = feature["properties"].get("iso3cd")
        if code in allowed and feature.get("geometry"):
            geometries.append(make_valid(shape(feature["geometry"])))
            found.add(code)
    if allowed - found:
        raise ValueError("Missing Street View region boundaries: " + ", ".join(sorted(allowed - found)))
    geometry = union_all(geometries)
    if geometry.is_empty:
        raise ValueError("Empty Google Street View region coverage")
    prepare(geometry)
    return geometry


def google_coverage_mask(coordinates):
    """Boolean array for (..., latitude/longitude); boundary points are included."""
    import numpy as np
    from shapely import intersects_xy

    values = np.asarray(coordinates, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("coordinates must end in latitude, longitude")
    lat, lon = values[..., 0], values[..., 1]
    valid = np.isfinite(values).all(axis=-1) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    return valid & intersects_xy(_coverage_geometry(), lon, lat)


def coverage_details():
    data = coverage_metadata()
    return {"scope": data["scope"], "version": data["version"],
            "region_count": len(data["iso3_codes"]), "road_level_verified": False}
