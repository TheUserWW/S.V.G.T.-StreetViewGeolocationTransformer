"""Offline adapters for global street-view datasets and existing JSONL files."""

from collections import Counter
import csv
import math
import os
from pathlib import Path


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


def iter_manifest(path, *, report=None, cancelled=None):
    import json

    path = Path(path).expanduser().resolve()
    counts = Counter()
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if number % 512 == 0:
                check_cancel(cancelled)
            if not line.strip():
                continue
            row = json.loads(line)
            paths = validate_row(row, context=f"{path}:{number}")
            resolved = [(path.parent / name).resolve() for name in paths]
            if not all(image.is_file() for image in resolved):
                raise ValueError(f"{path}:{number}: 图片不存在，请补齐文件后再准备数据")
            row = dict(row)
            if "images" in row:
                row["images"] = [str(image) for image in resolved]
            else:
                row["image"] = str(resolved[0])
            row["latitude"], row["longitude"] = float(row["latitude"]), float(row["longitude"])
            row.setdefault("source", "Mapillary" if "mapillary_id" in row else "custom")
            counts[row["source"]] += 1
            yield row
    if report is not None:
        report.update({"type": "jsonl", "path": str(path), "samples": sum(counts.values()),
                       "sources": dict(counts)})


def _image_key(parts):
    place, year, month, bearing, latitude, longitude, panoid = parts
    angle = float(bearing)
    if not math.isfinite(angle):
        raise ValueError("invalid bearing")
    return int(place), int(year), int(month), angle, panoid


def iter_gsv_cities(root, *, report=None, progress=None, cancelled=None):
    """One dated view per sample; place identity is used for splitting, not fusion."""
    root = Path(root).expanduser().resolve()
    dataframes, images = root / "Dataframes", root / "Images"
    csv_paths = sorted(dataframes.glob("*.csv"))
    if not images.is_dir() or not csv_paths:
        raise ValueError(f"{root}: 需要 GSV-Cities 的 Images 与 Dataframes/*.csv")
    total, places, cities = 0, set(), []
    for csv_path in csv_paths:
        check_cancel(cancelled)
        city = csv_path.stem
        folder = images / city
        if not folder.is_dir():
            raise ValueError(f"缺少城市图片目录：{folder}，请确认解压已完成")
        # Match the official filename fields without reformatting floating-point
        # coordinates. panoid can itself contain underscores.
        index = {}
        for number, entry in enumerate(os.scandir(folder)):
            if number % 512 == 0:
                check_cancel(cancelled)
            if not entry.is_file() or Path(entry.name).suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            stem = Path(entry.name).stem
            if not stem.startswith(city + "_"):
                continue
            parts = stem[len(city) + 1:].split("_", 6)
            if len(parts) != 7:
                continue
            try:
                key = _image_key(parts)
                coordinate = (float(parts[4]), float(parts[5]))
            except (ValueError, OverflowError):
                continue
            if key in index:
                raise ValueError(f"{folder}: 同一记录匹配到多张图片：{entry.name}")
            # folder is already absolute/resolved; resolving every one of half
            # a million files adds unnecessary filesystem calls on Windows.
            index[key] = (entry.path, coordinate)
        used, count = set(), 0
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{csv_path}: 缺少字段 {sorted(required - set(reader.fieldnames or []))}")
            for number, item in enumerate(reader, 2):
                if number % 512 == 0:
                    check_cancel(cancelled)
                try:
                    if item["city_id"] != city:
                        raise ValueError("CSV city_id 与目录不一致")
                    key = (int(item["place_id"]), int(item["year"]), int(item["month"]),
                           float(item["northdeg"]), item["panoid"])
                    if not 1 <= key[2] <= 12 or not math.isfinite(key[3]):
                        raise ValueError("拍摄日期或朝向无效")
                    if key not in index:
                        raise ValueError(f"找不到对应图片：place={key[0]}, panoid={key[4]}")
                    image, file_coordinate = index[key]
                    latitude, longitude = float(item["lat"]), float(item["lon"])
                    if not all(math.isfinite(value) for value in (*file_coordinate, latitude, longitude)):
                        raise ValueError("经纬度不是有限数字")
                    if max(abs(latitude - file_coordinate[0]), abs(longitude - file_coordinate[1])) > 1e-5:
                        raise ValueError("CSV 与图片文件名中的经纬度不一致")
                    group = f"gsv_cities:{city}:{key[0]}"
                    row = {"image": image, "latitude": latitude, "longitude": longitude,
                           "headings": [key[3] % 360], "source": "GSV-Cities", "city": city,
                           "place_id": str(key[0]), "group_id": group,
                           "panorama_id": item["panoid"], "year": key[1], "month": key[2],
                           "northdeg_raw": key[3]}
                    validate_row(row, context=f"{csv_path}:{number}")
                except (KeyError, ValueError, TypeError, OverflowError) as error:
                    raise ValueError(f"{csv_path}:{number}: {error}") from error
                if key in used:
                    continue
                used.add(key)
                places.add(group)
                count += 1
                total += 1
                yield row
        cities.append({"city": city, "samples": count, "image_files": len(index),
                       "unreferenced_images": len(index) - len(used)})
        if progress:
            progress({"event": "dataset_import", "source": "GSV-Cities", "city": city,
                      "city_samples": count, "samples": total, "cities_done": len(cities), "cities_total": len(csv_paths)})
    if report is not None:
        report.update({"type": "gsv_cities", "path": str(root), "samples": total,
                       "places": len(places), "cities": cities, "scope": "world"})


def resolve_source(root):
    root = Path(root).expanduser().resolve()
    if (root / "Images").is_dir() and (root / "Dataframes").is_dir():
        return "gsv_cities", root
    if (root / "train.csv").is_file() or (root / "train" / "train.csv").is_file():
        return "osv5m", root
    if (root / "all.jsonl").is_file():
        return "jsonl", root / "all.jsonl"
    raise ValueError(f"{root}: 需要 GSV-Cities Images/Dataframes、OSV-5M train.csv 或 Mapillary all.jsonl")
