"""A separate game-feedback trainer. It does not relax OSV training admission.

Training samples and validation samples are supplied as disjoint snapshots by
the runner. The base checkpoint is immutable. Interrupted jobs save trainable
parameters and optimizer state separately; only validated candidates are promoted.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

from .geography import distance_km, metrics, promotion_allowed


def _atomic_torch_save(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare_snapshot(store, config):
    train = store.samples("train", config.max_train_samples)
    validation = store.samples("validation", config.max_validation_samples)
    validation_hashes = {row["image_hash"] for row in validation}
    train = [row for row in train if row["image_hash"] not in validation_hashes and
             all(distance_km((row["latitude"], row["longitude"]), (v["latitude"], v["longitude"]))
                 > config.validation_buffer_km for v in validation)]
    if len(train) < config.min_train or len(validation) < config.min_validation:
        return None
    return {"train": train, "validation": validation}


def prepare_recent_batch(store, config):
    """Next disjoint batch of X completed rounds, with a separate holdout.

    Completed batches are acknowledged even if a candidate is rejected. A job
    interrupted during training retains its exact snapshot in pending_train.
    """
    offset=store.get('last_training_count',0)
    window=store.completed_window(offset,config.train_every)
    meta={'start':offset+1,'end':offset+len(window),'round_ids':[r['id'] for r in window]}
    if len(window)<config.train_every:
        return {'state':'waiting','detail':f'本批有效回合 {len(window)}/{config.train_every} / Batch rounds',**meta}
    validation=store.samples('validation',config.max_validation_samples)
    hashes={r['image_hash'] for r in validation}
    train=[r for r in window if r['split']=='train' and r['image_hash'] not in hashes and
           all(distance_km((r['latitude'],r['longitude']),(v['latitude'],v['longitude']))>config.validation_buffer_km
               for v in validation)]
    if not train:
        return {'state':'skipped','detail':'本批均为验证地点或过近样本 / No eligible training locations',**meta}
    if len(train)<config.min_train or len(validation)<config.min_validation:
        return {'state':'waiting','detail':f'等待独立样本：训练 {len(train)}/{config.min_train}，验证 {len(validation)}/{config.min_validation} / Waiting for holdout',**meta}
    return {'state':'ready','train':train,'validation':validation,**meta}


def _batch(model, rows, device):
    import torch
    from streetgeo.datasets.images import load_image
    views, size = max(len(row["images"]) for row in rows), model.config.image_size
    if views > model.config.max_views:
        raise ValueError("采集视角数量超过当前模型支持数量")
    images = torch.zeros(len(rows), views, 3, size, size)
    mask = torch.zeros(len(rows), views, dtype=torch.bool)
    headings = torch.full((len(rows), views), float("nan"))
    for index, row in enumerate(rows):
        for view, path in enumerate(row["images"]):
            images[index, view] = load_image(path, size)
            mask[index, view] = True
            if row["headings"][view] is not None:
                headings[index, view] = row["headings"][view]
    targets = torch.tensor([[row["latitude"], row["longitude"]] for row in rows], dtype=torch.float32, device=device)
    return images.to(device), mask.to(device), headings.to(device), targets


def train_candidate(request, cancelled=lambda: False, *, sample_validator=None, promoter=None):
    import torch
    from streetgeo.core.checkpoints import load_model, save_checkpoint
    from .config import LearningConfig

    # Jobs written before recent-batch mode retain their original replay sampler.
    cfg = LearningConfig(**{"sample_mode": "replay", **request["learning"]})
    base, output, resume = map(Path, (request["checkpoint"], request["candidate"], request["resume"]))
    train, validation = request["train"], request["validation"]
    if not train or not validation:
        raise ValueError("训练和验证集都不能为空")
    train_ids, val_ids = {r["id"] for r in train}, {r["id"] for r in validation}
    if train_ids & val_ids or {r["image_hash"] for r in train} & {r["image_hash"] for r in validation}:
        raise ValueError("训练和验证数据重叠")
    if sample_validator is not None:
        sample_validator(train + validation)
    elif any(row.get("status") != "completed" or row.get("label_source") != "openguessr_completed_round"
             for row in train + validation):
        raise ValueError("只能训练已提交并核对过真值的网页回合")
    if cancelled():
        return {"status": "paused", "step": 0}
    stamp = base.stat()
    identity = hashlib.sha256(json.dumps({"base": str(base.resolve()), "mtime": stamp.st_mtime_ns,
        "bytes": stamp.st_size, "train": [r["id"] for r in train], "validation": [r["id"] for r in validation],
        "learning": request["learning"]}, sort_keys=True).encode()).hexdigest()
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume.exists() else None
    if saved and saved.get("identity") != identity:
        raise ValueError("断点对应的数据或基础模型已改变，请重置该候选训练任务")
    model = load_model(base, request["device"]).eval()
    device = next(model.parameters()).device
    if cfg.last_visual_blocks > len(model.encoder.visual_blocks):
        raise ValueError("解冻层数超过视觉主干层数")
    prefixes = tuple(f"encoder.visual_blocks.{index}." for index in
                     range(len(model.encoder.visual_blocks) - cfg.last_visual_blocks, len(model.encoder.visual_blocks)))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("encoder.") or bool(prefixes and name.startswith(prefixes)))
    named = {name: value for name, value in model.named_parameters() if value.requires_grad}
    optimizer = torch.optim.AdamW(named.values(), lr=cfg.learning_rate, weight_decay=.01)
    features = {}
    cache_path = resume.with_suffix(".features.pt")
    if cfg.last_visual_blocks == 0 and cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache.get("identity") == identity:
            features.update(cache["features"])

    def encode(rows):
        missing = [row for row in rows if row["id"] not in features]
        for start in range(0, len(missing), cfg.batch_size):
            if cancelled():
                _atomic_torch_save({"identity": identity, "features": features}, cache_path)
                return False
            group = missing[start:start + cfg.batch_size]
            images, mask, headings, _ = _batch(model, group, device)
            with torch.no_grad():
                encoded = model.encoder(images, view_mask=mask, headings=headings).detach().cpu()
            features.update((row["id"], vector) for row, vector in zip(group, encoded))
            if start % (8 * cfg.batch_size) == 0:
                _atomic_torch_save({"identity": identity, "features": features}, cache_path)
        _atomic_torch_save({"identity": identity, "features": features}, cache_path)
        return True

    def forward(rows, training=True):
        if cfg.last_visual_blocks == 0:
            values = torch.stack([features[row["id"]] for row in rows]).to(device)
            targets = torch.tensor([[r["latitude"], r["longitude"]] for r in rows], device=device)
            return model.forward_embedding(values, targets if training else None, top_k=1)
        images, mask, headings, targets = _batch(model, rows, device)
        return model(images, targets if training else None, view_mask=mask, headings=headings, top_k=1)

    def evaluate():
        errors = []
        with torch.no_grad():
            for start in range(0, len(validation), cfg.batch_size):
                if cancelled():
                    return None
                rows = validation[start:start + cfg.batch_size]
                result = forward(rows, training=False)
                errors.extend(distance_km(p, (r["latitude"], r["longitude"]))
                              for p, r in zip(result.coordinates.cpu().tolist(), rows))
        return metrics(errors)

    # Frozen-encoder features exist only within this job. Never reuse after an encoder update.
    if cfg.last_visual_blocks == 0 and not encode(train + validation):
        return {"status": "paused", "step": int(saved["step"]) if saved else 0}
    before = saved["before"] if saved else evaluate()
    if before is None:
        return {"status": "paused", "step": 0}
    step = 0
    if saved:
        with torch.no_grad():
            for name, value in saved["parameters"].items():
                named[name].copy_(value.to(device))
        optimizer.load_state_dict(saved["optimizer"])
        step = saved["step"]

    def save_progress():
        _atomic_torch_save({"identity": identity, "step": step, "before": before,
                           "parameters": {name: value.detach().cpu() for name, value in named.items()},
                           "optimizer": optimizer.state_dict()}, resume)

    epoch_steps = (len(train) + cfg.batch_size - 1) // cfg.batch_size
    total_steps = max(cfg.steps, epoch_steps) if cfg.sample_mode == "recent" else cfg.steps
    while step < total_steps:
        if cancelled():
            save_progress()
            return {"status": "paused", "step": step}
        if cfg.sample_mode == "recent":
            # Every eligible round receives a gradient update before repeating.
            # Reconstructing the order from the step also makes resume deterministic.
            epoch, position = divmod(step, epoch_steps)
            order = list(train)
            random.Random(1337 + epoch).shuffle(order)
            rows = order[position * cfg.batch_size:(position + 1) * cfg.batch_size]
        else:
            rows = random.Random(1337 + step).sample(train, min(cfg.batch_size, len(train)))
        optimizer.zero_grad(set_to_none=True)
        result = forward(rows)
        if not torch.isfinite(result.loss):
            raise ValueError("训练损失无效，候选模型未发布")
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(named.values(), .5, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step % cfg.checkpoint_every == 0:
            save_progress()
    save_progress()
    after = evaluate()
    if after is None or cancelled():
        return {"status": "paused", "step": step}
    if base.stat().st_mtime_ns != stamp.st_mtime_ns or base.stat().st_size != stamp.st_size:
        raise ValueError("基础模型在训练期间改变，候选模型未发布")
    accepted = (promoter or promotion_allowed)(before, after, cfg.minimum_improvement_km)
    report = {"status": "accepted" if accepted else "rejected", "step": step,
              "checkpoint": str(base.resolve()), "candidate": str(output.resolve()),
              "before": before, "after": after, "train_count": len(train), "validation_count": len(validation),
              "frozen_encoder": cfg.last_visual_blocks == 0,
              "validation_scope": "held_out_game_locations_only"}
    if accepted:
        save_checkpoint(model, output, training_state={"autoplay": report})
    resume.unlink(missing_ok=True)
    cache_path.unlink(missing_ok=True)
    return report
