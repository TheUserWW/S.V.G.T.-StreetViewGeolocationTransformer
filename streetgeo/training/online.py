"""Supervised, local updates to geographic heads from map-click feedback.

The visual encoder stays frozen. Replay features only belong to that unchanged
encoder and are embedded atomically in each personal checkpoint. Ordinary OSV
training admission rules are not relaxed by this separate feedback workflow.
"""

from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import torch

from streetgeo.core.checkpoints import load_model, read_checkpoint, save_checkpoint
from streetgeo.datasets.images import load_image
from streetgeo.maps.projection import coordinates
from streetgeo.paths import ROOT

SCHEMA = 1
MEMORY_SIZE = 256


def _stamp(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _image_key(paths):
    digest = hashlib.sha256()
    for path in paths:
        image_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                image_digest.update(block)
        digest.update(image_digest.digest())
    return digest.hexdigest()


def _reject_official_test(paths):
    for path in paths:
        parts = [p.casefold() for p in path.parts]
        if "openstreetview-5m" in parts:
            tail = parts[parts.index("openstreetview-5m") + 1:]
            if tail[:1] == ["test"] or tail[:2] == ["images", "test"]:
                raise ValueError("官方 test 图片只用于评估，不能加入个人学习；请使用自己的游戏街景")


def learn_from_feedback(checkpoint, images, latitude, longitude, *, device="auto", output_root=None,
                        steps=4, learning_rate=1e-5, replay_batch=8, progress=lambda text: None):
    """Immediately train on a human label, saving a separate inference checkpoint.

    This is adaptation, not a held-out quality claim. Repeat submissions of the
    same image/view sequence replace its label in memory instead of duplicating
    it. The base file is never overwritten. A failed update is never committed.
    """
    lat, lon = coordinates(latitude, longitude)
    if type(steps) is not int or not 1 <= steps <= 16:
        raise ValueError("即时学习步数必须为 1～16")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1e-4:
        raise ValueError("即时学习率必须在 0～0.0001 之间")
    if type(replay_batch) is not int or not 0 <= replay_batch <= 32:
        raise ValueError("历史回放数量必须为 0～32")
    path = Path(checkpoint).expanduser().resolve()
    image_paths = [Path(image).expanduser().resolve() for image in images]
    if not image_paths:
        raise ValueError("请先导入同一地点的街景图片")
    _reject_official_test(image_paths)
    source_stamp = _stamp(path)
    state = read_checkpoint(path).get("training_state", {}).get("online_learning")
    if state and state.get("schema") != SCHEMA:
        raise ValueError("不支持此个人学习记录的版本")
    progress("正在准备个人模型 · 仅更新定位头…")
    model = load_model(path, device)
    if len(image_paths) > model.config.max_views:
        raise ValueError(f"此模型最多接受同一地点的 {model.config.max_views} 张图片")
    target_device = next(model.parameters()).device
    model.eval()  # Stable features; no dropout or encoder gradients during adaptation.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("encoder."))
    with torch.no_grad():
        inputs = torch.stack([load_image(p, model.config.image_size) for p in image_paths])
        feature = model.encoder(inputs.unsqueeze(0).to(target_device)).detach()
    key = _image_key(image_paths)
    memory = list(state.get("memory", [])) if state else []
    memory = [entry for entry in memory if entry["image_key"] != key]
    # Round-robin replay visits the whole retained history, rather than only
    # repeatedly using the most recent games. Randomness never touches global RNG.
    update = state["updates"] + 1 if state else 1
    if memory and replay_batch:
        start = ((update - 1) * replay_batch) % len(memory)
        replay = [memory[(start + i) % len(memory)] for i in range(min(replay_batch, len(memory)))]
    else:
        replay = []
    features = torch.cat([feature, *[entry["feature"].to(target_device) for entry in replay]])
    targets = torch.tensor([[lat, lon], *[entry["coordinates"] for entry in replay]],
                           dtype=torch.float32, device=target_device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=.01)
    with torch.no_grad():
        before = float(model.forward_embedding(features[:1], targets[:1], top_k=1).loss)
    for step in range(steps):
        progress(f"正在学习 {step + 1}/{steps} · 回放 {len(replay)} 个历史地点…")
        optimizer.zero_grad(set_to_none=True)
        # The new correction and the historical batch receive equal weight.
        fresh = model.forward_embedding(features[:1], targets[:1], top_k=1).loss
        loss = fresh
        if replay:
            old_loss = model.forward_embedding(features[1:], targets[1:], top_k=1).loss
            loss = .5 * (fresh + old_loss)
        if not torch.isfinite(loss):
            raise ValueError("学习出现非有限损失，已放弃本次更新")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, .5, error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad():
        after = float(model.forward_embedding(features[:1], targets[:1], top_k=1).loss)
    if not math.isfinite(after) or any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("学习出现无效参数，已放弃本次更新")
    if _stamp(path) != source_stamp:
        raise ValueError("学习期间模型文件已改变，本次更新未保存")
    if state:
        output = path
        original = state["base_checkpoint"]
        original_stamp = state["base_stamp"]
    else:
        folder = Path(output_root) if output_root else ROOT / ".gui" / "online_learning"
        output = folder / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]) / "personal.pt"
        original, original_stamp = str(path), source_stamp
    memory.append({"image_key": key, "feature": feature.cpu(), "coordinates": [lat, lon],
                   "images": [str(p) for p in image_paths]})
    memory = memory[-MEMORY_SIZE:]
    online = {"schema": SCHEMA, "personal_checkpoint": str(output.resolve()),
              "base_checkpoint": original, "base_stamp": original_stamp, "updates": update,
              "memory": memory, "updated_at": datetime.now(timezone.utc).isoformat(),
              "frozen_encoder": True, "steps": steps, "learning_rate": learning_rate,
              "validation_status": "unvalidated_user_feedback", "last_coordinates": [lat, lon]}
    progress("正在保存个人模型…")
    save_checkpoint(model, output, training_state={"online_learning": online}, keep_previous=bool(state))
    return {"checkpoint": str(output.resolve()), "base_checkpoint": original, "updates": update,
            "memory_size": len(memory), "replayed": len(replay), "loss_before": before, "loss_after": after,
            "latitude": lat, "longitude": lon}


def undo_feedback(checkpoint):
    """Undo the last committed update, or return to the untouched base model."""
    path = Path(checkpoint).expanduser().resolve()
    state = read_checkpoint(path).get("training_state", {}).get("online_learning")
    if not state:
        raise ValueError("当前模型没有个人学习记录")
    if state["updates"] == 1:
        original = Path(state["base_checkpoint"])
        if not original.is_file() or _stamp(original) != state["base_stamp"]:
            raise ValueError("基础模型已移动或改变，无法自动切回")
        return {"checkpoint": str(original), "updates": 0}
    previous = path.with_name(path.stem + ".previous" + path.suffix)
    if not previous.is_file():
        raise ValueError("没有可撤销的上一次更新")
    old = read_checkpoint(previous).get("training_state", {}).get("online_learning")
    if not old or old["updates"] != state["updates"] - 1 or old["base_checkpoint"] != state["base_checkpoint"]:
        raise ValueError("只能撤销最近一次更新；更早版本请手动选择备份模型")
    descriptor, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        shutil.copyfile(previous, temp)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return {"checkpoint": str(path), "updates": old["updates"]}
