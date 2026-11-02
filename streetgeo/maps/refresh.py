"""Download the public UN OCHA reference map for offline display (manual tool)."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from streetgeo.paths import ROOT

SERVICE = "https://gis.unocha.org/server/rest/services/Hosted/Global_AB_25M_simplified_fs_gray/FeatureServer"
ITEM = "https://gis.unocha.org/portal/sharing/rest/content/items/455567f87cfc43718978b6b2662770a8"
DISCLAIMER = ("The boundaries and names shown and the designations used on this map do not imply "
              "official endorsement or acceptance by the United Nations.")


def get_json(url, **parameters):
    if parameters:
        url += "?" + urlencode(parameters)
    request = Request(url, headers={"User-Agent": "SVGTOfflineMap/1.0"})
    with urlopen(request, timeout=90) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(f"Map service returned an error: {result['error']}")
    return result


def download_layer(layer, fields):
    url = f"{SERVICE}/{layer}"
    metadata = get_json(url, f="json")
    expected = get_json(url + "/query", where="1=1", returnCountOnly="true", f="json")["count"]
    features, identifiers = [], set()
    while len(features) < expected:
        page = get_json(url + "/query", where="1=1", outFields=fields,
                        outSR="4326", f="geojson", orderByFields="objectid ASC",
                        resultOffset=len(features), resultRecordCount=1000)
        rows = page.get("features", [])
        if not rows:
            raise RuntimeError(f"Incomplete layer {layer}: {len(features)} of {expected}")
        for row in rows:
            identifier = row["properties"]["objectid"]
            if identifier in identifiers:
                raise RuntimeError(f"Duplicate feature {identifier}; download aborted")
            identifiers.add(identifier)
        features.extend(rows)
        print(json.dumps({"layer": layer, "features": len(features), "expected": expected}), flush=True)
    if len(features) != expected:
        raise RuntimeError("Layer changed during download; retry")
    return {"type": "FeatureCollection", "features": features}, metadata


def write_json(path, content):
    payload = (json.dumps(content, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
            "features": len(content.get("features", []))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets" / "maps")
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    countries, country_meta = download_layer(18, "objectid,iso3cd,nam_en,lbl_en,stscod")
    boundaries, boundary_meta = download_layer(12, "objectid,iso3cd,bdytyp")
    item_error = None
    try:
        item = get_json(ITEM, f="json")
    except (OSError, RuntimeError) as error:
        # The public layer remains queryable when its portal item is unavailable.
        item, item_error = {}, str(error)
    renderer = boundary_meta["drawingInfo"]["renderer"]
    styles = {}
    for info in renderer["uniqueValueInfos"]:
        styles[info["value"]] = {"label": info["label"], "symbol": info["symbol"]}
    manifest = {
        "provider": "United Nations / OCHA", "service": SERVICE, "item": ITEM,
        "downloaded_at": datetime.now(timezone.utc).isoformat(), "crs": "EPSG:4326",
        "scale": "1:25,000,000 (source simplified reference layers)",
        "disclaimer": DISCLAIMER, "boundary_styles": styles,
        "credits": item.get("accessInformation", ""), "license_info": item.get("licenseInfo", ""),
        "source_modified_ms": item.get("modified"), "item_metadata_error": item_error, "files": {},
    }
    manifest["files"]["un_countries.geojson"] = write_json(folder / "un_countries.geojson", countries)
    manifest["files"]["un_boundaries.geojson"] = write_json(folder / "un_boundaries.geojson", boundaries)
    write_json(folder / "un_map_sources.json", manifest)
    print(json.dumps({"output": str(folder), "files": manifest["files"],
                      "boundary_types": dict(Counter(str(f["properties"]["bdytyp"]) for f in boundaries["features"]))}))


if __name__ == "__main__":
    main()
