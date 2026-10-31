"""Resumable U.S. Mapillary image collection using the official Graph API.

SQLite commits discovery cursors before downloading, so interruptions never
lose discovered IDs. Re-running the same command resumes the same dataset.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import uuid

from PIL import Image

from streetgeo.datasets.mapillary_client import (DownloadError, MapillaryClient, MapillaryHTTPError,
                              RetryableDownloadError, ThumbnailUnavailable)
from streetgeo.datasets.us_boundaries import USBoundary, fetch_boundaries

FIELDS = ("id,computed_geometry,geometry,computed_compass_angle,compass_angle,"
          "camera_type,is_pano,captured_at,sequence,creator")
LICENSE_URL = "https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data"


class ImageTooSmallError(ValueError):
    """All available thumbnail variants are too small; do not retry next run."""

    def __init__(self, size, min_side, tried):
        self.size, self.min_side, self.tried = size, min_side, tried
        super().__init__(f"可用图片尺寸不足：{size[0]}×{size[1]}，要求短边至少 {min_side} 像素；"
                         f"已尝试分辨率 {'/'.join(map(str, tried))}")


def normalize_metadata(row, boundary, bbox=None):
    identifier = str(row.get("id", ""))
    if not re.fullmatch(r"[0-9]+", identifier):
        return None
    if row.get("is_pano") or row.get("camera_type") != "perspective":
        return None
    geometry = row.get("computed_geometry") or row.get("geometry") or {}
    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") != "Point" or len(coordinates) < 2:
        return None
    try:
        lon, lat = map(float, coordinates[:2])
    except (ValueError, TypeError):
        return None
    if not math.isfinite(lon) or not math.isfinite(lat) or not boundary.contains(lon, lat):
        return None
    if bbox is not None and not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
        return None
    heading = row.get("computed_compass_angle")
    if heading is None:
        heading = row.get("compass_angle")
    try:
        heading = float(heading) if heading is not None else None
        if heading is not None and not math.isfinite(heading):
            heading = None
    except (TypeError, ValueError):
        heading = None
    sequence = row.get("sequence")
    if isinstance(sequence, dict):
        sequence = sequence.get("id")
    return {"image": f"images/{identifier}.jpg", "latitude": lat, "longitude": lon,
            "headings": [heading], "mapillary_id": identifier, "sequence_id": str(sequence or identifier),
            "captured_at": row.get("captured_at"), "creator": row.get("creator"),
            "source": "Mapillary", "source_url": f"https://www.mapillary.com/app/?pKey={identifier}",
            "license": "CC-BY-SA", "license_url": LICENSE_URL}


class DownloadStore:
    def __init__(self, folder, settings, tiles):
        self.settings = settings
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "images").mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.folder / "downloads.sqlite3")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tiles (
                id INTEGER PRIMARY KEY, bbox TEXT NOT NULL, cursor TEXT, done INTEGER DEFAULT 0,
                selected INTEGER DEFAULT 0, root_id INTEGER, depth INTEGER NOT NULL DEFAULT 0,
                failures INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
                last_failed_run TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY, metadata TEXT NOT NULL, sequence_id TEXT NOT NULL,
                status INTEGER DEFAULT 0, bytes INTEGER DEFAULT 0, last_run TEXT, error TEXT);
            CREATE INDEX IF NOT EXISTS images_status ON images(status);
            CREATE INDEX IF NOT EXISTS images_sequence ON images(sequence_id);
        """)
        saved = self.db.execute("SELECT value FROM settings WHERE id=1").fetchone()
        encoded = json.dumps(settings, sort_keys=True)
        if saved and saved[0] != encoded:
            self.db.close()
            raise ValueError("Download settings changed; use the same settings or a new output directory")
        # Migrate existing queues in place, without changing images or cursors.
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(tiles)")}
        for name, definition in (("root_id", "INTEGER"), ("depth", "INTEGER NOT NULL DEFAULT 0"),
                                 ("failures", "INTEGER NOT NULL DEFAULT 0"),
                                 ("retry_at", "REAL NOT NULL DEFAULT 0"),
                                 ("last_failed_run", "TEXT"), ("error", "TEXT")):
            if name not in columns:
                self.db.execute(f"ALTER TABLE tiles ADD COLUMN {name} {definition}")
        self.db.execute("CREATE INDEX IF NOT EXISTS tiles_root ON tiles(root_id)")
        self.db.execute("CREATE INDEX IF NOT EXISTS tiles_ready ON tiles(done,depth DESC,id)")
        self.db.execute("CREATE INDEX IF NOT EXISTS tiles_deferred ON tiles(done,retry_at)")
        if not saved:
            with self.db:
                self.db.execute("INSERT INTO settings VALUES (1, ?)", (encoded,))
                self.db.executemany("INSERT INTO tiles(id,bbox) VALUES (?,?)",
                                    ((index, json.dumps(tile)) for index, tile in enumerate(tiles)))
        with self.db:
            self.db.execute("UPDATE tiles SET root_id=id WHERE root_id IS NULL")
        # Missing files are recoverable; never claim they are downloaded.
        missing = [(identifier,) for (identifier,) in self.db.execute("SELECT id FROM images WHERE status=1")
                   if not (self.folder / "images" / f"{identifier}.jpg").is_file()]
        with self.db:
            self.db.executemany("UPDATE images SET status=0,bytes=0 WHERE id=?", missing)

    def stats(self):
        count, size = self.db.execute("SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM images WHERE status=1").fetchone()
        failed = self.db.execute("SELECT COUNT(*) FROM images WHERE status=-1").fetchone()[0]
        skipped = self.db.execute("SELECT COUNT(*) FROM images WHERE status=2").fetchone()[0]
        deferred = self.db.execute("SELECT COUNT(*) FROM tiles WHERE done=0 AND retry_at>0").fetchone()[0]
        return {"downloaded": count, "bytes": size, "failed": failed, "skipped": skipped,
                "deferred_regions": deferred}

    def _root_selected(self, root_id):
        return self.db.execute("SELECT COALESCE(SUM(selected),0) FROM tiles WHERE root_id=?", (root_id,)).fetchone()[0]

    def next_tile(self, run_id, max_per_tile):
        while True:
            tile = self.db.execute(
                "SELECT id,bbox,cursor,selected,root_id FROM tiles WHERE done=0 AND retry_at<=? "
                "AND (last_failed_run IS NULL OR last_failed_run!=?) ORDER BY depth DESC,id LIMIT 1",
                (time.time(), run_id)).fetchone()
            if tile is None:
                return None
            if self._root_selected(tile[4]) < max_per_tile:
                return tile[:4]
            with self.db:
                self.db.execute("UPDATE tiles SET done=1 WHERE root_id=? AND done=0", (tile[4],))

    def recover_query(self, tile, error, run_id):
        """Persist subdivision or deferred retry; never mark failed work complete."""
        tile_id, encoded_bounds = tile[:2]
        root_id, depth, failures = self.db.execute(
            "SELECT root_id,depth,failures FROM tiles WHERE id=?", (tile_id,)).fetchone()
        west, south, east, north = json.loads(encoded_bounds)
        split = (isinstance(error, MapillaryHTTPError)
                 and (error.status == 500 or error.query_too_large)
                 and depth < 4 and min(east - west, north - south) / 2 >= 0.005 - 1e-9)
        with self.db:
            if split:
                lon, lat = (west + east) / 2, (south + north) / 2
                children = ((west, south, lon, lat), (lon, south, east, lat),
                            (west, lat, lon, north), (lon, lat, east, north))
                self.db.execute("UPDATE tiles SET done=2,failures=failures+1,error=? WHERE id=?",
                                (str(error), tile_id))
                self.db.executemany("INSERT INTO tiles(bbox,root_id,depth) VALUES (?,?,?)",
                                    ((json.dumps(bounds), root_id, depth + 1) for bounds in children))
                return "split"
            # Do not immediately hammer the same failed request again. Other
            # regions can continue; a later invocation retries it after cooldown.
            delay = min(300, 30 * 2 ** min(failures, 4))
            self.db.execute("UPDATE tiles SET failures=failures+1,retry_at=?,last_failed_run=?,error=? WHERE id=?",
                            (time.time() + delay, run_id, str(error), tile_id))
        return "deferred"

    def pending(self, run_id, count):
        pending = []
        for identifier, encoded, error in self.db.execute(
            "SELECT id,metadata,error FROM images WHERE status IN (0,-1) AND (last_run IS NULL OR last_run!=?) LIMIT ?",
            (run_id, count)):
            metadata = json.loads(encoded)
            # Recover old failures directly at 1024 rather than re-fetching the
            # already-rejected 256px variant. This is an in-memory hint only.
            if (error or "").startswith("Image is smaller than min-side") and self.settings["resolution"] == 256:
                metadata["_resolution_hint"] = 1024
            pending.append((identifier, metadata))
        return pending

    def discover(self, tile, rows, cursor, boundary, args):
        tile_id, bounds, old_cursor, selected = tile
        if cursor is not None and cursor == old_cursor:
            raise DownloadError("Mapillary repeated a pagination cursor; stopping with progress retained")
        root_id = self.db.execute("SELECT root_id FROM tiles WHERE id=?", (tile_id,)).fetchone()[0]
        root_selected = self._root_selected(root_id)
        with self.db:
            for row in rows:
                if root_selected >= args.max_per_tile:
                    break
                metadata = normalize_metadata(row, boundary, args.bbox)
                if metadata is None:
                    continue
                if self.db.execute("SELECT 1 FROM images WHERE id=?", (metadata["mapillary_id"],)).fetchone():
                    continue
                count = self.db.execute("SELECT COUNT(*) FROM images WHERE sequence_id=?",
                                        (metadata["sequence_id"],)).fetchone()[0]
                if count >= args.max_per_sequence:
                    continue
                self.db.execute("INSERT INTO images(id,metadata,sequence_id) VALUES (?,?,?)",
                                (metadata["mapillary_id"], json.dumps(metadata), metadata["sequence_id"]))
                selected += 1
                root_selected += 1
            done = cursor is None or root_selected >= args.max_per_tile
            self.db.execute("UPDATE tiles SET cursor=?,done=?,selected=?,retry_at=0,error=NULL WHERE id=?",
                            (cursor, int(done), selected, tile_id))
            if root_selected >= args.max_per_tile:
                self.db.execute("UPDATE tiles SET done=1 WHERE root_id=? AND done=0", (root_id,))

    def export(self):
        for filename, attribution_only in (("all.jsonl", False), ("attribution.jsonl", True)):
            path = self.folder / filename
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                for (encoded,) in self.db.execute("SELECT metadata FROM images WHERE status=1 ORDER BY id"):
                    row = json.loads(encoded)
                    if attribution_only:
                        row = {key: row[key] for key in ("mapillary_id", "creator", "source_url", "license", "license_url")}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)

    def close(self):
        self.db.close()


def download_one(client, identifier, folder, resolution, min_side):
    path = Path(folder) / "images" / f"{identifier}.jpg"
    if path.exists():
        try:
            with Image.open(path) as image:
                if min(image.size) < min_side:
                    raise ValueError("Existing thumbnail is too small")
                image.load()
            return path.stat().st_size
        except (OSError, ValueError):
            pass
    resolutions = [value for value in (256, 1024, 2048) if value >= resolution]
    size, tried = None, []
    for candidate in resolutions:
        tried.append(candidate)
        try:
            data = client.image_bytes(identifier, candidate)
        except ThumbnailUnavailable:
            continue
        with Image.open(io.BytesIO(data)) as image:
            size = image.size
            # Keep the quality threshold. Request more pixels rather than
            # upscaling an undersized thumbnail to make it pass validation.
            if min(size) < min_side:
                continue
            image.load()
        break
    else:
        if size is not None:
            raise ImageTooSmallError(size, min_side, tried)
        raise ThumbnailUnavailable(f"Image {identifier} has no available thumbnail")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return len(data)


def collect(args, client, boundary):
    settings = {name: getattr(args, name) for name in (
        "region", "states", "bbox", "tile_size", "seed", "resolution", "min_side",
        "max_per_tile", "max_per_sequence")}
    settings["boundary_fingerprint"] = boundary.fingerprint
    database = Path(args.output) / "downloads.sqlite3"
    tiles = [] if database.exists() else boundary.tiles(args.tile_size, args.seed, args.bbox)
    store = DownloadStore(args.output, settings, tiles)
    run_id, pages, logged = uuid.uuid4().hex, 0, -100
    processed, small_examples = 0, 0
    consecutive_query_errors, outcome = 0, "running"
    print(f"图像质量：优先 {args.resolution} 像素版本，短边不足 {args.min_side} 时自动尝试更大版本；"
          "所有可用版本仍不达标时记录为跳过。", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while store.stats()["downloaded"] < args.max_images:
                if getattr(args, "stop_file", None) and Path(args.stop_file).exists():
                    print("Download stopped safely; completed images and cursors are retained", flush=True)
                    outcome = "stopped"
                    break
                stats = store.stats()
                if stats["bytes"] >= args.max_gb * 1024 ** 3:
                    print("Reached image disk budget; raise --max-gb to continue", flush=True)
                    outcome = "budget_limit"
                    break
                if shutil.disk_usage(args.output).free < args.min_free_gb * 1024 ** 3:
                    print("Disk free space is below --min-free-gb; stopping safely", flush=True)
                    outcome = "budget_limit"
                    break
                pending = store.pending(run_id, min(args.workers, args.max_images - stats["downloaded"]))
                if pending:
                    futures = [(identifier, pool.submit(download_one, client, identifier, args.output,
                                                         metadata.get("_resolution_hint", args.resolution), args.min_side))
                               for identifier, metadata in pending]
                    for identifier, future in futures:
                        try:
                            size = future.result()
                            with store.db:
                                store.db.execute("UPDATE images SET status=1,bytes=?,last_run=?,error=NULL WHERE id=?",
                                                 (size, run_id, identifier))
                        except ImageTooSmallError as error:
                            with store.db:
                                store.db.execute("UPDATE images SET status=2,last_run=?,error=? WHERE id=?",
                                                 (run_id, str(error), identifier))
                            if small_examples < 3:
                                print(f"跳过图片 {identifier}：{error}", flush=True)
                                small_examples += 1
                        except (DownloadError, OSError, ValueError) as error:
                            message = str(error)
                            if "graph.mapillary.com" in message and any(code in message for code in ("HTTP 401", "HTTP 403")):
                                raise DownloadError("Mapillary token was rejected; check the client access token") from None
                            with store.db:
                                store.db.execute("UPDATE images SET status=-1,last_run=?,error=? WHERE id=?",
                                                 (run_id, message[:300], identifier))
                            print(f"Image {identifier}: {message}", flush=True)
                        processed += 1
                    if processed - logged >= 100:
                        logged = processed
                        print(json.dumps(store.stats()), flush=True)
                    continue
                if pages >= args.max_pages:
                    print("Reached --max-pages for this invocation; re-run to continue discovery", flush=True)
                    outcome = "discovery_limit"
                    break
                tile = store.next_tile(run_id, args.max_per_tile)
                if tile is None:
                    if store.stats()["deferred_regions"]:
                        print("本次可查询区域已处理，其余失败区域保留待重试；稍后继续即可。", flush=True)
                        outcome = "paused"
                    else:
                        print("Selected search tiles are exhausted", flush=True)
                        outcome = "source_exhausted"
                    break
                # Failed queries also consume the invocation's discovery budget.
                pages += 1
                try:
                    rows, cursor = client.search(json.loads(tile[1]), FIELDS, after=tile[2])
                except (MapillaryHTTPError, RetryableDownloadError) as error:
                    if isinstance(error, MapillaryHTTPError) and not (error.retryable or error.query_too_large):
                        raise
                    action = store.recover_query(tile, error, run_id)
                    if action == "split":
                        print(f"区域 {tile[0]} 查询失败（{error}），已保存为 4 个更小区域继续查询。", flush=True)
                    else:
                        print(f"区域 {tile[0]} 查询暂不可用（{error}），已保留游标并延后重试，先处理其他区域。", flush=True)
                    if not getattr(error, "query_too_large", False):
                        consecutive_query_errors += 1
                    if consecutive_query_errors >= 6:
                        print("接口连续多次返回错误，本次安全暂停；图片、坐标和查询进度已保留，请稍后继续。", flush=True)
                        outcome = "paused"
                        break
                    continue
                consecutive_query_errors = 0
                store.discover(tile, rows, cursor, boundary, args)
                if pages % 100 == 0:
                    print(json.dumps({"discovery_pages": pages, **store.stats()}), flush=True)
            if outcome == "running":
                outcome = "target_reached"
    except KeyboardInterrupt:
        outcome = "stopped"
        print("Download interrupted; IDs, cursors and completed files are retained", flush=True)
    except BaseException:
        outcome = "error"
        raise
    finally:
        try:
            store.export()
            print(json.dumps({"event": "download_finished", "status": outcome,
                              "output": str(Path(args.output).resolve()), **store.stats()}), flush=True)
        finally:
            store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/mapillary_us")
    parser.add_argument("--boundary", default="data/boundaries/us_states.geojson")
    parser.add_argument("--boundary-only", action="store_true", help="cache Census boundaries without a Mapillary token")
    parser.add_argument("--region", choices=("conus", "us"), default="conus")
    parser.add_argument("--states", type=lambda value: sorted(set(value.upper().split(","))))
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--max-images", type=int, default=10000, help="target total, including previous downloads")
    parser.add_argument("--max-pages", type=int, default=2000, help="API discovery budget per invocation")
    parser.add_argument("--max-per-tile", type=int, default=30)
    parser.add_argument("--max-per-sequence", type=int, default=50)
    parser.add_argument("--tile-size", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resolution", type=int, choices=(256, 1024, 2048), default=1024,
                        help="preferred thumbnail size; automatically try larger variants if the short side is insufficient")
    parser.add_argument("--min-side", type=int, default=224, help="minimum accepted short side in pixels")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-second", type=float, default=2)
    parser.add_argument("--max-gb", type=float, default=20)
    parser.add_argument("--min-free-gb", type=float, default=5)
    parser.add_argument("--prompt-token", action="store_true", help="read token without echoing it")
    parser.add_argument("--stop-file", help="GUI stop request file; checked between download batches")
    args = parser.parse_args()
    for name in ("max_images", "max_pages", "max_per_tile", "max_per_sequence", "workers", "min_side"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not 0 < args.tile_size <= 0.09 or args.requests_per_second <= 0 or args.max_gb <= 0 or args.min_free_gb < 0:
        parser.error("invalid tile-size, request rate or disk budget")
    if args.bbox and not (-180 <= args.bbox[0] < args.bbox[2] <= 180 and -90 <= args.bbox[1] < args.bbox[3] <= 90):
        parser.error("bbox must be WEST SOUTH EAST NORTH without crossing the antimeridian")
    token = None
    if not args.boundary_only:
        token = getpass.getpass("Mapillary client access token (not saved): ") if args.prompt_token else os.environ.get("MAPILLARY_ACCESS_TOKEN")
        if not token:
            parser.error("set MAPILLARY_ACCESS_TOKEN locally, or use --prompt-token; never put the token in source code")
    fetch_boundaries(args.boundary)
    if args.boundary_only:
        print(f"Boundary cache ready: {args.boundary}")
        return
    boundary = USBoundary(args.boundary, args.region, args.states)
    client = MapillaryClient(token, args.requests_per_second,
                             on_retry=lambda event: print(json.dumps(event), flush=True))
    try:
        collect(args, client, boundary)
    except DownloadError as error:
        print(f"下载请求失败：{error}。已保存的图片和清单仍可继续使用。", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
