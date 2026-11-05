"""Optional Köppen–Geiger supervision for Mapillary geolocation training."""

from collections import defaultdict
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from streetgeo.paths import ROOT
from streetgeo.training.runtime import manifest_fingerprint


CLIMATE_CLASSES = ("A", "B", "C", "D", "E")
UNKNOWN_CLIMATE = -100
DEFAULT_CLIMATE_RASTER = ROOT / "data/climate/beck2023_1991_2020/koppen_geiger_0p01.nc"
CLIMATE_DATA_SOURCE = "https://doi.org/10.6084/m9.figshare.21937568.v1"
CLIMATE_RECIPE = "beck2023_1991_2020_5classes_v1"
# Beck et al. V2 legend in numeric raster order. Index 0 is water/NoData.
CODE_TO_CLASS = np.asarray([UNKNOWN_CLIMATE, *([0] * 3), *([1] * 4), *([2] * 9),
                            *([3] * 12), *([4] * 2)])


def climate_provenance(path, minimum_confidence):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Climate raster is missing: {path}. Use the official 1991–2020 NetCDF from {CLIMATE_DATA_SOURCE}")
    return {"recipe": CLIMATE_RECIPE, "source": CLIMATE_DATA_SOURCE, "path": str(path),
            "sha256": manifest_fingerprint(path), "minimum_confidence": minimum_confidence}


def _nearest_indices(axis, coordinates):
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
        raise ValueError("Climate raster must have finite, one-dimensional latitude and longitude axes")
    step = (axis[-1] - axis[0]) / (len(axis) - 1)
    if not step or not np.allclose(np.diff(axis), step, rtol=.005, atol=1e-7):
        raise ValueError("Climate raster axes must be regularly spaced")
    return np.floor((coordinates - axis[0]) / step + .5).astype(np.int64)


def sample_climates(path, coordinates, minimum_confidence):
    """Look up categorical pixels. Return UNKNOWN_CLIMATE for low confidence or NoData."""
    from netCDF4 import Dataset

    labels = np.full(len(coordinates), UNKNOWN_CLIMATE, dtype=np.int64)
    if not len(coordinates):
        return labels.tolist()
    points = np.asarray(coordinates, dtype=np.float64)
    with Dataset(path, mode="r") as dataset:
        climate = dataset.variables["kg_class"]
        confidence = dataset.variables["kg_confidence"]
        lat_axis, lon_axis = dataset.variables["lat"], dataset.variables["lon"]
        if climate.dimensions != (lat_axis.dimensions[0], lon_axis.dimensions[0]) or climate.dimensions != confidence.dimensions:
            raise ValueError("Expected aligned kg_class and kg_confidence grids in latitude/longitude order")
        rows = _nearest_indices(lat_axis[:], points[:, 0])
        longitude = np.remainder(points[:, 1] + 180, 360) - 180
        cols = _nearest_indices(lon_axis[:], longitude)
        chunk = climate.chunking()
        block_rows, block_cols = (256, 256) if chunk == "contiguous" else chunk
        height, width = climate.shape
        blocks = defaultdict(list)
        for index, (row, col) in enumerate(zip(rows, cols)):
            if 0 <= row < height and 0 <= col < width:
                blocks[row // block_rows, col // block_cols].append(index)
        for (block_row, block_col), indices in blocks.items():
            top, left = block_row * block_rows, block_col * block_cols
            region = np.s_[top:top + block_rows, left:left + block_cols]
            location = rows[indices] - top, cols[indices] - left
            codes = np.ma.asarray(climate[region])[location].astype(np.int16).filled(0)
            scores = np.ma.asarray(confidence[region])[location].astype(np.float32).filled(np.nan)
            valid = (codes > 0) & (codes < len(CODE_TO_CLASS)) & (scores >= minimum_confidence) & (scores <= 100)
            labels[np.asarray(indices)[valid]] = CODE_TO_CLASS[codes[valid]]
    return labels.tolist()


def assign_climates(dataset, provenance):
    dataset.climate_targets = sample_climates(provenance["path"], [sample[1] for sample in dataset.samples],
                                               provenance["minimum_confidence"])
    return {**{name: dataset.climate_targets.count(index) for index, name in enumerate(CLIMATE_CLASSES)},
            "unknown": dataset.climate_targets.count(UNKNOWN_CLIMATE)}


def climate_logits(head, embedding):
    # The model exposes unit embeddings. Scale them to the RMS of a normalized
    # 768-wide feature so the small classifier has a useful input magnitude.
    return head(embedding * math.sqrt(embedding.shape[-1]))


def climate_loss(logits, targets):
    loss = F.cross_entropy(logits.float(), targets, ignore_index=UNKNOWN_CLIMATE, reduction="sum")
    return loss / targets.ne(UNKNOWN_CLIMATE).sum().clamp_min(1)


def update_confusion(confusion, logits, targets):
    valid = targets.ne(UNKNOWN_CLIMATE)
    classes = len(CLIMATE_CLASSES)
    pairs = targets[valid] * classes + logits.argmax(dim=-1)[valid]
    confusion += torch.bincount(pairs, minlength=classes * classes).reshape(classes, classes).cpu()


def climate_metrics(confusion):
    count = int(confusion.sum())
    if not count:
        return {"labeled": 0, "accuracy": None, "macro_f1": None}
    support = confusion.sum(dim=1)
    total = support + confusion.sum(dim=0)
    f1 = 2 * confusion.diag() / total.clamp_min(1)
    return {"labeled": count, "accuracy": int(confusion.diag().sum()) / count,
            "macro_f1": float(f1[support > 0].mean()),
            "support": dict(zip(CLIMATE_CLASSES, support.tolist())), "confusion": confusion.tolist()}
