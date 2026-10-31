"""Stream OSV-5M annotations and join only the locally extracted training images."""
import csv
import os
from pathlib import Path

from streetgeo.datasets.sources import check_cancel, validate_row


def osv_layout(root):
    root = Path(root).expanduser().resolve()
    if (root / "train.csv").is_file():
        metadata = root / "train.csv"
        folder = next((p for p in (root / "images" / "train", root / "train", root / "images") if p.is_dir()), root)
    elif (root / "train" / "train.csv").is_file():
        metadata = root / "train" / "train.csv"
        folder = root / "images" / "train"
        if not folder.is_dir():
            folder = root / "train"
    else:
        raise ValueError(f"{root}: 找不到 train.csv 或 train/train.csv；请同时保留坐标表和已解压训练图片")
    return metadata, folder


def _value(item, key):
    value = (item.get(key) or "").strip()
    return "" if value.lower() in ("nan", "none", "null") else value


def iter_osv5m(root, *, report=None, progress=None, cancelled=None):
    metadata, folder = osv_layout(root)
    index = {}
    for current, directories, files in os.walk(folder):
        check_cancel(cancelled)
        directories[:] = sorted(d for d in directories if d.lower() not in ("test", ".cache", "__macosx", ".git"))
        for number, name in enumerate(files):
            if number % 512 == 0:
                check_cancel(cancelled)
            image = Path(name)
            if image.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            if image.stem in index:
                raise ValueError(f"同一 OSV-5M 图片 ID 对应多个本地文件：{image.stem}")
            index[image.stem] = str(Path(current) / name)
        if progress:
            progress({"event": "dataset_import_progress", "source": "OpenStreetView-5M",
                      "stage": "index", "local_images": len(index), "samples": 0})
    if not index:
        raise ValueError(f"{folder}: 没有已解压的训练图片；ZIP 文件需先解压，test 图片不参与训练")
    seen = {}
    rows = missing = duplicates = 0
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "latitude", "longitude"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{metadata}: 缺少字段 {sorted(required - set(reader.fieldnames or []))}")
        for number, item in enumerate(reader, 2):
            rows += 1
            if rows % 2048 == 0:
                check_cancel(cancelled)
            if progress and rows % 50000 == 0:
                progress({"event": "dataset_import_progress", "source": "OpenStreetView-5M", "stage": "metadata",
                          "metadata_rows": rows, "local_images": len(index), "samples": len(seen), "missing_images": missing})
            identifier = _value(item, "id")  # Never convert long identifiers through floats.
            image = index.get(identifier)
            if image is None:
                missing += 1
                continue
            try:
                latitude, longitude = float(item["latitude"]), float(item["longitude"])
                sequence = _value(item, "sequence")
                row = {"image": image, "latitude": latitude, "longitude": longitude,
                       "source": "OpenStreetView-5M", "osv5m_id": identifier,
                       "group_id": f"osv5m:{identifier}", "country": _value(item, "country"),
                       "city": _value(item, "unique_city") or _value(item, "city") or _value(item, "country") or "unspecified"}
                if sequence:
                    row["sequence_id"] = sequence
                captured = _value(item, "captured_at")
                if captured:
                    row["captured_at"] = captured
                validate_row(row, context=f"{metadata}:{number}")
                signature = (latitude, longitude, sequence)
                if identifier in seen:
                    if seen[identifier] != signature:
                        raise ValueError(f"图片 ID {identifier} 存在冲突的坐标或采集序列")
                    duplicates += 1
                    continue
                seen[identifier] = signature
            except (KeyError, ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"{metadata}:{number}: {error}") from error
            yield row
    check_cancel(cancelled)
    if not seen:
        raise ValueError(f"{metadata}: CSV 中没有匹配的本地训练图片；请检查文件名 ID 和图片目录")
    detail = {"type": "osv5m", "path": str(Path(root).resolve()), "metadata_csv": str(metadata),
              "image_folder": str(folder), "metadata_rows": rows, "local_images": len(index),
              "samples": len(seen), "missing_images": missing, "duplicate_rows": duplicates,
              "images_without_metadata": len(index) - len(seen), "scope": "world", "official_split": "train"}
    if report is not None:
        report.update(detail)
    if progress:
        progress({"event": "dataset_import_complete", "source": "OpenStreetView-5M", **detail})
