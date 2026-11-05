"""Local image preprocessing and JSONL manifests. No network access."""

import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from streetgeo.core.config import SIGLIP2_PREPROCESSING, ModelConfig
from streetgeo.datasets.policy import require_mapillary_row
from streetgeo.datasets.road_mask import RoadMask, MASK_FILL

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
IMAGE_FILL = MASK_FILL
SIGLIP2_MEAN_STD = (0.5, 0.5, 0.5)


def image_statistics(preprocessing):
    if preprocessing == SIGLIP2_PREPROCESSING:
        return SIGLIP2_MEAN_STD, SIGLIP2_MEAN_STD
    return IMAGE_MEAN, IMAGE_STD


def manifest_lines(path):
    """Stream large imported datasets instead of keeping another full text copy."""
    with path.open(encoding="utf-8-sig") as stream:
        yield from enumerate(stream, 1)


def load_image(path, image_size=224, *, road_mask=None, mask_bottom_third=False):
    """RGB and EXIF orientation, then the checkpoint's resize and normalization.

    Avoid random horizontal flips: they can reverse driving-side/sign clues.
    Training and inference always use exactly the same preprocessing.
    """
    preprocessing = None
    if isinstance(image_size, ModelConfig):
        preprocessing, image_size = image_size.preprocessing, image_size.image_size
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if road_mask is not None and mask_bottom_third:
            raise ValueError("choose one environment mask shape")
        if road_mask is not None:
            image = road_mask.apply(image)
        if mask_bottom_third:
            # Retained only to resume checkpoints created with recipe version 1.
            image.paste(IMAGE_FILL, (0, image.height * 2 // 3, image.width, image.height))
        if preprocessing == SIGLIP2_PREPROCESSING:
            image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
        else:
            image = ImageOps.pad(image, (image_size, image_size),
                                 method=Image.Resampling.BICUBIC, color=IMAGE_FILL)
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    pixels = pixels.reshape(image_size, image_size, 3).permute(2, 0, 1).float() / 255
    mean, std = image_statistics(preprocessing)
    mean = pixels.new_tensor(mean).view(3, 1, 1)
    std = pixels.new_tensor(std).view(3, 1, 1)
    return (pixels - mean) / std


class StreetViewDataset(Dataset):
    """Each JSONL row is one location with 1..max_views local image paths.

    {"images": ["front.jpg", "back.jpg"], "latitude": 34.0,
     "longitude": -118.0, "headings": [0, 180]}

    A single view may use "image": "front.jpg". Headings are optional and may
    contain null for unknown directions. Relative paths resolve beside JSONL.
    """

    def __init__(self, manifest_path, config=None, *, mapillary_only=False, environment_specialist=False,
                 road_mask=None, legacy_environment_mask=False):
        self.config = config or ModelConfig()
        self.environment_specialist = environment_specialist
        self.road_mask = road_mask if road_mask is not None else RoadMask()
        self.legacy_environment_mask = legacy_environment_mask
        manifest = Path(manifest_path).resolve()
        self.samples = []
        self.climate_targets = None
        self.source_counts = {}
        self.geographic_bounds = {"south": 90.0, "north": -90.0, "west": 180.0, "east": -180.0}
        for line_number, line in manifest_lines(manifest):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if mapillary_only:
                    require_mapillary_row(row)
                if not isinstance(row, dict):
                    raise ValueError("each row must be a JSON object")
                if "images" in row and "image" in row:
                    raise ValueError("use either image or images")
                paths = row.get("images", [row["image"]] if "image" in row else [])
                if not isinstance(paths, list) or not 1 <= len(paths) <= self.config.max_views:
                    raise ValueError(f"provide 1..{self.config.max_views} image paths")
                if not all(isinstance(path, str) and path for path in paths):
                    raise ValueError("image paths must be nonempty strings")
                latitude, longitude = float(row["latitude"]), float(row["longitude"])
                if not math.isfinite(latitude) or not -90 <= latitude <= 90:
                    raise ValueError("latitude must be finite and in [-90, 90]")
                if not math.isfinite(longitude) or not -180 <= longitude <= 180:
                    raise ValueError("longitude must be finite and in [-180, 180]")
                headings = row.get("headings", [None] * len(paths))
                if not isinstance(headings, list) or len(headings) != len(paths):
                    raise ValueError("headings must match the number of images")
                headings = [float("nan") if value is None else float(value) for value in headings]
                for original, value in zip(row.get("headings", [None] * len(paths)), headings):
                    if original is not None and not math.isfinite(value):
                        raise ValueError("headings must be finite numbers or null")
                self.samples.append((
                    [manifest.parent / path for path in paths],
                    (latitude, longitude), headings,
                ))
                source = str(row.get("source", "Mapillary" if "mapillary_id" in row else "custom"))
                self.source_counts[source] = self.source_counts.get(source, 0) + 1
                bounds = self.geographic_bounds
                bounds["south"], bounds["north"] = min(bounds["south"], latitude), max(bounds["north"], latitude)
                bounds["west"], bounds["east"] = min(bounds["west"], longitude), max(bounds["east"], longitude)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{manifest}:{line_number}: {error}") from error
        if not self.samples:
            raise ValueError("manifest contains no samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        phase = None
        if isinstance(index, tuple):
            index, phase = index
            if not self.environment_specialist or phase not in (0, 1):
                raise ValueError("invalid environment training phase")
        paths, coordinates, headings = self.samples[index]
        # Workers spawned by an already-running version-1 trainer may unpickle
        # a dataset created before these shape attributes existed.
        legacy = getattr(self, "legacy_environment_mask", True)
        sample = {
            "images": torch.stack([load_image(path, self.config,
                                  road_mask=self.road_mask if phase == 1 and not legacy else None,
                                  mask_bottom_third=phase == 1 and legacy) for path in paths]),
            "coordinates": torch.tensor(coordinates, dtype=torch.float32),
            "headings": torch.tensor(headings, dtype=torch.float32),
        }
        if phase is not None:
            sample["training_phase"] = phase
        climate_targets = getattr(self, "climate_targets", None)
        if climate_targets is not None:
            sample["climate_target"] = climate_targets[index]
        return sample


class EnvironmentViewDataset(Dataset):
    """Reuse a validated dataset for masked evaluation without copying its metadata."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index, 1]


def collate_streetviews(samples):
    """Pad view counts and explicitly mask padded views."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    views = max(sample["images"].shape[0] for sample in samples)
    batch = len(samples)
    shape = samples[0]["images"].shape[1:]
    images = samples[0]["images"].new_zeros(batch, views, *shape)
    mask = torch.zeros(batch, views, dtype=torch.bool)
    headings = torch.full((batch, views), float("nan"))
    for index, sample in enumerate(samples):
        count = sample["images"].shape[0]
        images[index, :count] = sample["images"]
        mask[index, :count] = True
        headings[index, :count] = sample["headings"]
    result = {"images": images, "view_mask": mask, "headings": headings,
              "targets": torch.stack([sample["coordinates"] for sample in samples])}
    if "climate_target" in samples[0]:
        result["climate_targets"] = torch.tensor([sample["climate_target"] for sample in samples], dtype=torch.long)
    phases = {sample.get("training_phase") for sample in samples}
    if len(phases) != 1:
        raise ValueError("a batch must contain exactly one training phase")
    if phases != {None}:
        result["training_phase"] = torch.tensor(phases.pop())
    return result
