"""Small dependency-free geography helpers; no model or browser state."""

import hashlib
import math
import statistics


def coordinate(latitude, longitude):
    if isinstance(latitude, bool) or isinstance(longitude, bool):
        raise ValueError("坐标不能是布尔值")
    lat, lon = float(latitude), float(longitude)
    if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("无效经纬度")
    return lat, (lon + 180) % 360 - 180


def distance_km(a, b):
    lat1, lon1 = map(math.radians, coordinate(*a))
    lat2, lon2 = map(math.radians, coordinate(*b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1, max(0, h))))


def spatial_split(lat, lon, *, degrees=1, fraction=.2):
    lat, lon = coordinate(lat, lon)
    block = f"svgt-autoplay-v1:{math.floor((lat + 90) / degrees)}:{math.floor((lon + 180) / degrees)}"
    number = int.from_bytes(hashlib.sha256(block.encode()).digest()[:8], "big") / 2 ** 64
    return "validation" if number < fraction else "train"


def metrics(errors):
    if not errors or any(not math.isfinite(e) or e < 0 for e in errors):
        raise ValueError("验证距离为空或无效")
    return {"count": len(errors), "median_km": statistics.median(errors), "mean_km": statistics.mean(errors),
            "within_25km": sum(e <= 25 for e in errors) / len(errors),
            "within_200km": sum(e <= 200 for e in errors) / len(errors),
            "over_2000km": sum(e > 2000 for e in errors) / len(errors)}


def promotion_allowed(before, after, improvement=.01):
    return (before["count"] == after["count"] and before["count"] > 0
            and after["median_km"] < before["median_km"] - improvement
            and after["mean_km"] <= before["mean_km"]
            and after["over_2000km"] <= before["over_2000km"])
