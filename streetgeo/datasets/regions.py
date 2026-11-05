"""Offline geographic selection for worldwide Mapillary collection.

Country polygons are the bundled UN reference geography, not image sources.
Selections are combined by union; a custom GeoJSON and bbox further restrict it.
"""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random

from shapely import make_valid
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from streetgeo.paths import MAP_ASSETS

CONTINENTS = {
    "africa": ("非洲 / Africa", "AGO BDI BEN BFA BWA CAF CIV CMR COD COG COM CPV DJI DZA EGY ERI ESH ETH GAB GHA GIN GMB GNB GNQ KEN LBR LBY LSO MAR MDG MLI MOZ MRT MUS MWI MYT NAM NER NGA REU RWA SDN SEN SHN SLE SOM SSD STP SWZ SYC TCD TGO TUN TZA UGA ZAF ZMB ZWE"),
    "asia": ("亚洲 / Asia", "AFG ARE ARM AZE BGD BHR BRN BTN CHN CYP GEO HKG IDN IND IRN IRQ ISR JOR JPN KAZ KGZ KHM KOR KWT LAO LBN LKA MAC MDV MMR MNG MYS NPL OMN PAK PHL PRK PSE QAT SAU SGP SYR THA TJK TKM TLS TUR TWN UZB VNM YEM XAC XAP XJK"),
    "europe": ("欧洲 / Europe", "ALA ALB AND AUT BEL BGR BIH BLR CHE CZE DEU DNK ESP EST FIN FRA FRO GBR GGY GIB GRC HRV HUN IMN IRL ISL ITA JEY LIE LTU LUX LVA MCO MDA MKD MLT MNE NLD NOR POL PRT ROU RUS SJM SMR SRB SVK SVN SWE UKR VAT"),
    "north-america": ("北美洲 / North America", "ABW AIA ATG BES BHS BLM BLZ BMU BRB CAN CRI CUB CUW CYM DMA DOM GLP GRD GRL GTM HND HTI JAM KNA LCA MAF MEX MSR MTQ NIC PAN PRI SLV SPM SXM TCA TTO USA VCT VGB VIR"),
    "south-america": ("南美洲 / South America", "ARG BOL BRA CHL COL ECU FLK GUF GUY PER PRY SUR URY VEN"),
    "oceania": ("大洋洲 / Oceania", "ASM AUS CCK COK CXR FJI FSM GUM KIR MHL MNP NCL NFK NIU NRU NZL PCN PLW PNG PYF SLB TKL TON TUV VUT WSM"),
    "antarctica": ("南极洲 / Antarctica", "ATA ATF BVT HMD SGS"),
}
COUNTRY_CONTINENT = {code: continent for continent, (_, codes) in CONTINENTS.items() for code in codes.split()}


@lru_cache(maxsize=1)
def country_catalog():
    data = json.loads((MAP_ASSETS / "un_countries.geojson").read_text(encoding="utf-8"))
    countries = {}
    for feature in data["features"]:
        props = feature["properties"]
        code = props["iso3cd"].upper()
        countries[code] = {"code": code, "name": props.get("nam_en") or props.get("lbl_en") or code,
                           "continent": COUNTRY_CONTINENT[code]}
    return sorted(countries.values(), key=lambda item: item["name"])


def region_catalog():
    data = json.loads((MAP_ASSETS / "us_states.geojson").read_text(encoding="utf-8"))
    return [{"code": "US-" + feature["properties"]["name"], "country": "USA"}
            for feature in data["features"]]


def parse_codes(value):
    return sorted(set(part.strip().upper() for part in value.split(",") if part.strip()))


def validate_selection(continents=(), countries=(), regions=()):
    available = {item["code"] for item in country_catalog()}
    for selected, known, name in ((continents, CONTINENTS, "大洲"), (countries, available, "国家/地区"),
                                  (regions, {item["code"] for item in region_catalog()}, "行政区")):
        unknown = set(selected) - set(known)
        if unknown:
            raise ValueError(f"未知{name}代码：{', '.join(sorted(unknown))}")


class SelectionBoundary:
    spread_tiles = True

    def __init__(self, *, continents=(), countries=(), regions=(), geojson=None, bbox=None):
        validate_selection(continents, countries, regions)
        selected = set(countries)
        selected.update(code for code, continent in COUNTRY_CONTINENT.items() if continent in continents)
        content = (MAP_ASSETS / "un_countries.geojson").read_bytes()
        data = json.loads(content)
        pieces = []
        # With no named selection, a bbox or GeoJSON defines the requested area.
        for feature in data["features"]:
            code = feature["properties"]["iso3cd"].upper()
            if code in selected or not (continents or countries or regions):
                if feature.get("geometry"):
                    pieces.append(make_valid(shape(feature["geometry"])))
        digest = hashlib.sha256(content)
        if regions:
            content = (MAP_ASSETS / "us_states.geojson").read_bytes()
            digest.update(content)
            for feature in json.loads(content)["features"]:
                if "US-" + feature["properties"]["name"] in regions:
                    pieces.append(make_valid(shape(feature["geometry"])))
        geometry = unary_union(pieces)
        if geojson:
            content = Path(geojson).read_bytes()
            digest.update(content)
            data = json.loads(content)
            features = data.get("features", [data]) if data.get("type") == "FeatureCollection" else [data]
            geometries = [item.get("geometry", item) for item in features]
            if not geometries or any(item.get("type") not in ("Polygon", "MultiPolygon") for item in geometries):
                raise ValueError("区域 GeoJSON 必须只包含 Polygon / MultiPolygon（WGS84 经纬度）")
            custom = unary_union([make_valid(shape(item)) for item in geometries])
            geometry = geometry.intersection(custom)
        if bbox:
            geometry = geometry.intersection(box(*bbox))
        if geometry.is_empty or geometry.area <= 0:
            raise ValueError("所选区域没有交集，请检查国家、行政区、GeoJSON 和经纬度范围")
        west, south, east, north = geometry.bounds
        if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
            raise ValueError("区域坐标必须使用 WGS84 经度 [-180,180]、纬度 [-90,90]")
        self.geometry = geometry
        self.prepared = prep(geometry)
        self.fingerprint = digest.hexdigest()

    def contains(self, longitude, latitude):
        return self.prepared.covers(Point(longitude, latitude))

    def intersects(self, bounds):
        return self.prepared.intersects(box(*bounds))

    def tiles(self, size=0.08, seed=1337, bbox=None):
        """Queue coarse cells first; the store expands them lazily before API calls.

        This avoids allocating millions of fine cells for continent/world jobs.
        """
        if not math.isfinite(size) or not 0 < size <= 0.09:
            raise ValueError("tile size must be in (0, 0.09] degrees")
        geometry = self.geometry if bbox is None else self.geometry.intersection(box(*bbox))
        if geometry.is_empty:
            raise ValueError("请求范围与所选区域不相交")
        tiles = set()
        parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
        for part in parts:
            west, south, east, north = part.bounds
            for col in range(math.floor(west / 2), math.ceil(east / 2)):
                for row in range(math.floor(south / 2), math.ceil(north / 2)):
                    bounds = (max(-180, col * 2), max(-90, row * 2), min(180, (col + 1) * 2), min(90, (row + 1) * 2))
                    if part.intersects(box(*bounds)):
                        tiles.add(bounds)
        tiles = sorted(tiles)
        random.Random(seed).shuffle(tiles)
        return tiles
