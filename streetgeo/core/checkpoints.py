"""Versioned checkpoints carry the architecture, not just tensor weights."""

from dataclasses import asdict
from pathlib import Path
import os
import random
import shutil
import tempfile

import torch

from streetgeo.core.config import ModelConfig
from streetgeo.core.model import StreetGeoModel

ARCHITECTURE = "streetview_hierarchical_vit_v1"


def resolve_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    return device


def save_checkpoint(model, path, *, training_state=None, keep_previous=False):
    """Commit by atomic rename; a failed write leaves the old file usable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save({"architecture": ARCHITECTURE, "model_config": asdict(model.config),
                        "model_state_dict": model.state_dict(),
                        "training_state": training_state or {}}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if keep_previous and path.exists():
            previous = path.with_name(path.stem + ".previous" + path.suffix)
            backup = str(temporary) + ".backup"
            try:
                shutil.copyfile(path, backup)
                os.replace(backup, previous)
            finally:
                Path(backup).unlink(missing_ok=True)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def capture_rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; use the same devices for exact resume")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Expected a StreetGeo visual checkpoint; old text GPT weights are incompatible")
    return checkpoint


def load_model(path, device="auto"):
    device = resolve_device(device)
    checkpoint = read_checkpoint(path)
    model = StreetGeoModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()
