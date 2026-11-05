"""Local Mapillary manifest import. Other imagery sources are not supported."""

from collections import Counter
import math
from pathlib import Path

from streetgeo.datasets.policy import require_mapillary_row


class PreparationCancelled(Exception):
    pass


def check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise PreparationCancelled("数据准备已取消，可以重新执行")


def validate_row(row, *, context="sample"):
    if not isinstance(row, dict):
        raise ValueError(f"{context}: 每条记录必须是 JSON 对象")
    if "image" in row and "images" in row:
        raise ValueError(f"{context}: image 与 images 不能同时出现")
    paths = row.get("images", [row["image"]] if "image" in row else [])
    if not isinstance(paths, list) or not paths or not all(isinstance(path, str) and path for path in paths):
        raise ValueError(f"{context}: 缺少有效图片路径")
    lat, lon = float(row["latitude"]), float(row["longitude"])
    if not math.isfinite(lat) or not -90 <= lat <= 90 or not math.isfinite(lon) or not -180 <= lon <= 180:
        raise ValueError(f"{context}: 经纬度无效")
    headings = row.get("headings", [None] * len(paths))
    if not isinstance(headings, list) or len(headings) != len(paths):
        raise ValueError(f"{context}: 朝向数量与图片不匹配")
    if any(value is not None and not math.isfinite(float(value)) for value in headings):
        raise ValueError(f"{context}: 朝向必须是有限数字或 null")
    return paths


def iter_manifest(path, *, report=None, progress=None, cancelled=None):
    import json

    path = Path(path).expanduser().resolve()
    counts = Counter()
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if number % 512 == 1:
                check_cancel(cancelled)
            if not line.strip():
                continue
            row = json.loads(line)
            require_mapillary_row(row)
            paths = validate_row(row, context=f"{path}:{number}")
            resolved = [(path.parent / name).resolve() for name in paths]
            if not all(image.is_file() for image in resolved):
                raise ValueError(f"{path}:{number}: 图片不存在，请补齐文件后再准备数据")
            row = dict(row)
            row["image"] = str(resolved[0])
            row["latitude"], row["longitude"] = float(row["latitude"]), float(row["longitude"])
            counts[row["source"]] += 1
            if progress and sum(counts.values()) % 5000 == 0:
                progress({"event": "dataset_import_progress", "source": "Mapillary", "samples": sum(counts.values())})
            yield row
    if report is not None:
        report.update({"type": "jsonl", "path": str(path), "samples": sum(counts.values()),
                       "sources": dict(counts)})
    if progress:
        progress({"event": "dataset_import_complete", "source": "Mapillary", "samples": sum(counts.values())})


def resolve_source(root):
    root = Path(root).expanduser().resolve()
    manifest = root / "all.jsonl"
    if not manifest.is_file():
        raise ValueError(f"{root}: 仅接受 Mapillary 下载目录（需包含 all.jsonl 与 images/）")
    from streetgeo.datasets.policy import check_mapillary_manifest_header
    check_mapillary_manifest_header(manifest)
    return "mapillary", manifest
