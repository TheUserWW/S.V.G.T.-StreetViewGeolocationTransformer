"""Local-only OSV-5M official-test indexing and distance measurements."""
import csv
import math
import os
from pathlib import Path
import random


def distance_km(latitude, longitude, other_latitude, other_longitude):
    a, b = math.radians(latitude), math.radians(other_latitude)
    dlat, dlon = b - a, math.radians(other_longitude - longitude)
    value = math.sin(dlat / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(max(0, min(1, value))))


def attach_ground_truth(result, sample):
    if sample is None:
        return result
    result = dict(result, ground_truth=dict(sample))
    result["candidates"] = [dict(candidate, error_km=distance_km(
        sample["latitude"], sample["longitude"], candidate["latitude"], candidate["longitude"]))
        for candidate in result["candidates"]]
    result["top1_error_km"] = result["candidates"][0]["error_km"]
    return result


class OSVTestSampler:
    def __init__(self):
        self.key, self.samples = None, []

    def pick(self, directory):
        folder = Path(directory).expanduser().resolve()
        metadata = folder / "test.csv"
        if not metadata.is_file():
            raise ValueError(f"{folder}: 找不到 test.csv，请选择 OSV-5M test 目录")
        stamp = metadata.stat()
        key = (str(folder), stamp.st_mtime_ns, stamp.st_size)
        if key != self.key:
            scan = scan_test_samples(folder)
            if not scan["samples"]:
                raise ValueError("没有可测试的本地图片。请解压 test 图片分片并保留 test.csv；程序不会从 train 抽样或自动下载。")
            self.key, self.samples = key, scan["samples"]
        while self.samples:
            sample = random.choice(self.samples)
            if Path(sample["image"]).is_file():
                return dict(sample)
            self.samples.remove(sample)
        self.key = None
        raise ValueError("缓存中的测试图片已不存在，请重新选择 test 目录后再试")


class TestScanCancelled(Exception):
    """The user stopped an index scan before its coverage could be established."""


def scan_test_samples(directory, *, cancelled=lambda: False):
    """Fresh deterministic scan, retaining coverage and excluded CSV rows.

    IDs remain strings. Only test images matched by test.csv are eligible;
    metadata (including country) is never passed to the model.
    """
    folder = Path(directory).expanduser().resolve()
    metadata = folder / "test.csv"
    if not metadata.is_file():
        raise ValueError(f"{folder}: 找不到 test.csv，请选择 OSV-5M test 目录")
    index, excluded, samples, seen = {}, [], [], {}
    roots = [folder]
    alternate = folder.parent / "images" / "test"
    if alternate.is_dir() and alternate.resolve() != folder:
        roots.append(alternate)
    for root in roots:
        for current, dirs, files in os.walk(root):
            if cancelled():
                raise TestScanCancelled()
            dirs[:] = sorted(d for d in dirs if d.lower() not in ("train", ".cache", ".git", "__macosx"))
            for name in sorted(files):
                if cancelled():
                    raise TestScanCancelled()
                path = Path(current) / name
                if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                    continue
                identifier = path.stem
                if identifier in index:
                    raise ValueError(f"测试图片 ID 对应多个本地文件：{identifier}")
                index[identifier] = str(path)
    rows, matched = 0, set()
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "latitude", "longitude"}.issubset(reader.fieldnames or []):
            raise ValueError("test.csv 缺少 id、latitude 或 longitude 字段")
        for line, item in enumerate(reader, 2):
            if cancelled():
                raise TestScanCancelled()
            rows += 1
            identifier = (item.get("id") or "").strip()
            reason = ""
            try:
                lat, lon = float(item["latitude"]), float(item["longitude"])
                if not identifier or not (math.isfinite(lat) and math.isfinite(lon)
                                         and -90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError()
            except (TypeError, ValueError):
                reason = "invalid_id_or_coordinates"
            else:
                if identifier in seen:
                    if seen[identifier] != (lat, lon):
                        raise ValueError(f"test.csv 同一 ID 存在冲突坐标：{identifier}")
                    reason = "duplicate_id"
                else:
                    seen[identifier] = (lat, lon)
                    if identifier not in index:
                        reason = "missing_image"
                    else:
                        matched.add(identifier)
                        samples.append({"image": index[identifier], "osv5m_id": identifier,
                                        "latitude": lat, "longitude": lon, "split": "official_test",
                                        "country": (item.get("country") or "").strip()})
            if reason:
                excluded.append({"csv_line": line, "osv5m_id": identifier, "reason": reason})
    counts = {reason: sum(row["reason"] == reason for row in excluded)
              for reason in ("missing_image", "invalid_id_or_coordinates", "duplicate_id")}
    return {"samples": samples, "excluded": excluded,
            "coverage": {"metadata_rows": rows, "available_samples": len(samples),
                         "local_image_files": len(index), "unmatched_local_images": len(index) - len(matched),
                         **counts}}
