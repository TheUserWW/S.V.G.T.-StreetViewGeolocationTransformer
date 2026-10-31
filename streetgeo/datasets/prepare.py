"""Prepare GSV-Cities, partial OSV-5M, Mapillary or joint datasets without copying images."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

from streetgeo.datasets.sources import (PreparationCancelled, check_cancel, iter_gsv_cities,
                             iter_manifest, resolve_source)
from streetgeo.datasets.split import create_splits
from streetgeo.datasets.osv5m import iter_osv5m

PREPARATION_VERSION = 2
OUTPUT_FILES = ("all.jsonl", "train.jsonl", "val.jsonl", "test.jsonl")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_row(row, output):
    row = dict(row)
    def relative(path):
        try:
            return os.path.relpath(path, output).replace("\\", "/")
        except ValueError:  # Windows datasets and output can be on different drives.
            return str(Path(path)).replace("\\", "/")
    if "images" in row:
        row["images"] = [relative(path) for path in row["images"]]
    else:
        row["image"] = relative(row["image"])
    return row


def prepare_datasets(*, roots=(), manifests=(), output_dir="data/world_training", seed=1337,
                     block_degrees=0.02, guard_m=250, progress=None, cancelled=None):
    """An identical rerun reuses verified frozen files. Changed data needs a new folder."""
    seed, block_degrees, guard_m = int(seed), float(block_degrees), float(guard_m)
    sources = [resolve_source(path) for path in roots]
    sources.extend(("jsonl", Path(path).expanduser().resolve()) for path in manifests)
    sources = list(dict.fromkeys(sources))
    if not sources:
        raise ValueError("请提供 GSV-Cities / OSV-5M 数据目录或 Mapillary JSONL 清单")
    output = Path(output_dir).expanduser().resolve()
    config = {"preparation_version": PREPARATION_VERSION, "seed": seed,
              "block_degrees": block_degrees, "guard_m": guard_m,
              "sources": [[kind, str(path)] for kind, path in sources]}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8"))
    rows, details, seen = [], [], {}
    duplicate_count = 0
    for kind, path in sources:
        check_cancel(cancelled)
        detail = {}
        if kind == "gsv_cities":
            iterator = iter_gsv_cities(path, report=detail, progress=progress, cancelled=cancelled)
        elif kind == "osv5m":
            iterator = iter_osv5m(path, report=detail, progress=progress, cancelled=cancelled)
        else:
            iterator = iter_manifest(path, report=detail, cancelled=cancelled)
        for row in iterator:
            paths = row.get("images", [row["image"]] if "image" in row else [])
            identity = tuple(os.path.normcase(path) for path in paths)
            if identity in seen:
                previous = seen[identity]
                if (abs(previous[0] - row["latitude"]) > 1e-5 or
                        abs(previous[1] - row["longitude"]) > 1e-5):
                    raise ValueError(f"同一图片存在冲突坐标：{paths[0]}")
                duplicate_count += 1
                continue
            seen[identity] = (row["latitude"], row["longitude"])
            digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
            rows.append(row)
        details.append(detail)
    del seen
    if not rows:
        raise ValueError("没有可用于训练的图片记录")
    fingerprint = digest.hexdigest()
    report_path = output / "split_report.json"
    if report_path.is_file():
        try:
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            reusable = (saved.get("preparation_fingerprint") == fingerprint and
                        all((output / name).is_file() and file_digest(output / name) == saved["manifest_sha256"][name]
                            for name in OUTPUT_FILES))
        except (ValueError, KeyError, TypeError, OSError):
            reusable = False
        if reusable:
            if progress:
                progress({"event": "dataset_ready", "output": str(output), "reused": True,
                          "samples": len(rows), "counts": saved["counts"], "source_counts": saved["source_counts"]})
            return saved
        raise ValueError("该目录已有其他或发生变化的数据划分；请换用新的输出目录，以保留现有续训清单")
    if any((output / name).exists() for name in OUTPUT_FILES):
        raise ValueError("输出目录已有清单；请换用新的数据划分目录，避免覆盖已有训练数据")
    check_cancel(cancelled)
    if progress:
        progress({"event": "dataset_splitting", "samples": len(rows), "output": str(output)})
    splits, report = create_splits(rows, seed=seed, block_degrees=block_degrees,
                                  guard_m=guard_m, cancelled=cancelled, stratify_cities=True)
    report.update(config)
    report.update({"preparation_fingerprint": fingerprint, "source_details": details,
                   "samples": len(rows), "duplicates_removed": duplicate_count,
                   "source_counts": dict(Counter(row["source"] for row in rows)),
                   "city_counts": dict(Counter(row.get("city", "unspecified") for row in rows)),
                   "split_source_counts": {name: dict(Counter(row["source"] for row in part)) for name, part in splits.items()},
                   "split_city_counts": {name: dict(Counter(row.get("city", "unspecified") for row in part)) for name, part in splits.items()},
                   "scope": "world", "manifest_sha256": {}})
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage the whole result before publishing any file; originals stay read-only.
    with tempfile.TemporaryDirectory(prefix=output.name + ".build.", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for name, records in (("all", rows), *splits.items()):
            filename = name + ".jsonl"
            checksum = hashlib.sha256()
            with (staging / filename).open("wb") as stream:
                for index, row in enumerate(records):
                    if index % 512 == 0:
                        check_cancel(cancelled)
                    content = (json.dumps(_relative_row(row, output), ensure_ascii=False, allow_nan=False,
                                          separators=(",", ":")) + "\n").encode("utf-8")
                    stream.write(content)
                    checksum.update(content)
                stream.flush()
                os.fsync(stream.fileno())
            report["manifest_sha256"][filename] = checksum.hexdigest()
        (staging / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        check_cancel(cancelled)
        if not output.exists():
            os.replace(staging, output)
        else:
            # A user-selected empty/existing directory may contain unrelated files.
            # Refuse conflicting generated files, and publish the commit report last.
            for name in (*OUTPUT_FILES, "split_report.json"):
                if (output / name).exists():
                    raise ValueError("输出目录在准备期间发生变化，请使用新的输出目录")
            for name in (*OUTPUT_FILES, "split_report.json"):
                os.replace(staging / name, output / name)
    if progress:
        progress({"event": "dataset_ready", "output": str(output), "reused": False,
                  "samples": len(rows), "counts": report["counts"], "source_counts": report["source_counts"]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", action="append", default=[], help="GSV-Cities, OSV-5M or Mapillary directory; repeat to mix")
    parser.add_argument("--gsv-cities-root", action="append", default=[], help="alias for a GSV-Cities directory")
    parser.add_argument("--manifest", action="append", default=[], help="existing JSONL; repeat to mix")
    parser.add_argument("--output-dir", default="data/world_training")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--block-degrees", type=float, default=0.02)
    parser.add_argument("--guard-m", type=float, default=250)
    parser.add_argument("--stop-file")
    args = parser.parse_args()
    def progress(event):
        print(json.dumps(event, ensure_ascii=False), flush=True)
    try:
        prepare_datasets(roots=args.dataset_root + args.gsv_cities_root, manifests=args.manifest,
                         output_dir=args.output_dir, seed=args.seed, block_degrees=args.block_degrees,
                         guard_m=args.guard_m, progress=progress,
                         cancelled=lambda: bool(args.stop_file and Path(args.stop_file).exists()))
    except (PreparationCancelled, KeyboardInterrupt):
        progress({"event": "dataset_preparation_stopped", "message": "数据准备已取消，可以重新执行"})
    except (ValueError, OSError) as error:
        parser.exit(1, f"数据准备失败：{error}\n")


if __name__ == "__main__":
    main()
