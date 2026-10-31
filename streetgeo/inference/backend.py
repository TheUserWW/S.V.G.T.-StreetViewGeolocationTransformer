"""Local inference and interactive tile cache; imports torch only on prediction."""

from collections import OrderedDict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
import json
import os
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

from streetgeo.paths import ROOT


def read_picture(path):
    with Image.open(path) as source:
        source.load()
        return ImageOps.exif_transpose(source).convert("RGB")


class InferenceEngine:
    """Owned by one worker thread. Reload atomically replaced training checkpoints."""

    def __init__(self):
        self.model = None
        self.key = None

    def predict(self, checkpoint, images, *, device="auto", top_k=5, progress=lambda text: None):
        path = Path(checkpoint).expanduser()
        path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise ValueError("找不到 checkpoint，请先完成训练，或选择已有的视觉模型 .pt 文件")
        if device not in ("auto", "cuda", "cpu"):
            raise ValueError("设备必须为 auto、cuda 或 cpu")
        if not 1 <= int(top_k) <= 5:
            raise ValueError("候选数量应为 1～5")
        import torch
        from streetgeo.core.checkpoints import load_model
        from streetgeo.inference.predict import predict_images

        stamp = path.stat()
        key = (str(path), stamp.st_mtime_ns, stamp.st_size, device)
        start = time.perf_counter()
        reused = self.model is not None and key == self.key
        if not reused:
            progress("正在加载模型…")
            self.model, self.key = None, None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            self.model = load_model(path, device)
            self.key = key
        if not 1 <= len(images) <= self.model.config.max_views:
            raise ValueError(f"此模型支持同一地点的 1～{self.model.config.max_views} 张图片")
        progress("正在分析街景…")
        results = predict_images(self.model, images, top_k=min(int(top_k), self.model.config.total_cells))
        if next(self.model.parameters()).is_cuda:
            torch.cuda.synchronize()
        return {
            "source": "model", "checkpoint": str(path), "checkpoint_mtime_ns": stamp.st_mtime_ns,
            "images": [str(Path(image).resolve()) for image in images],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "device": str(next(self.model.parameters()).device), "reused_model": reused,
            "elapsed_seconds": time.perf_counter() - start,
            "candidates": [{"latitude": p.latitude, "longitude": p.longitude,
                            "coordinates": p.coordinates, "cell_probability": p.probability,
                            "cell_id": p.cell_id, "maps_url": p.maps_url} for p in results],
        }


class TileCache:
    """Fetch only requested visible OSM tiles, with persistent HTTP freshness."""

    def __init__(self, folder=None, *, opener=urlopen, clock=time.time):
        self.folder = Path(folder) if folder else ROOT / ".gui" / "map_tiles"
        self.opener, self.clock = opener, clock
        self.memory = OrderedDict()

    def get(self, zoom, x, y):
        if not all(type(value) is int for value in (zoom, x, y)) or not 0 <= zoom <= 19:
            raise ValueError("无效地图瓦片")
        if not 0 <= y < 2 ** zoom:
            raise ValueError("无效地图瓦片行")
        x %= 2 ** zoom
        key = (zoom, x, y)
        now = self.clock()
        if key in self.memory:
            picture, expires = self.memory.pop(key)
            if expires > now:
                self.memory[key] = (picture, expires)
                return picture
        path = self.folder / str(zoom) / str(x) / f"{y}.png"
        metadata = path.with_suffix(".json")
        try:
            expires = float(json.loads(metadata.read_text(encoding="utf-8"))["expires"])
            if expires > now:
                picture = read_picture(path)
                if picture.size == (256, 256):
                    self._remember(key, picture, expires)
                    return picture
        except (OSError, ValueError, KeyError, TypeError):
            pass
        request = Request(f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png",
                          headers={"User-Agent": "StreetGeoLocalMap/1.0 (desktop; interactive map)"})
        with self.opener(request, timeout=7) as response:
            content = response.read(2 * 1024 * 1024 + 1)
            headers = response.headers
        if len(content) > 2 * 1024 * 1024:
            raise ValueError("地图瓦片过大")
        with Image.open(BytesIO(content)) as source:
            if source.size != (256, 256):
                raise ValueError("地图瓦片尺寸不正确")
            picture = source.convert("RGB")
        policy = headers.get("Cache-Control", "").lower()
        match = re.search(r"max-age\s*=\s*(\d+)", policy)
        lifetime = int(match.group(1)) if match else 7 * 86400
        if not match and headers.get("Expires"):
            try:
                lifetime = max(0, parsedate_to_datetime(headers["Expires"]).timestamp() - now)
            except (TypeError, ValueError, OverflowError):
                pass
        try:
            lifetime = max(0, lifetime - int(headers.get("Age", 0)))
        except ValueError:
            pass
        expires = now + lifetime
        if "no-store" not in policy and "no-cache" not in policy:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(content)
                os.replace(temporary, path)
                metadata.write_text(json.dumps({"expires": expires}), encoding="utf-8")
            except OSError:
                pass  # A full disk must not prevent normal interactive viewing.
            self._remember(key, picture, expires)
        return picture

    def _remember(self, key, picture, expires):
        self.memory[key] = (picture, expires)
        while len(self.memory) > 96:
            self.memory.popitem(last=False)
