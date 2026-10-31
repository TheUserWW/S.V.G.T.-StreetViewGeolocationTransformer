"""Train on OpenStreetView-5M with atomic checkpoints and batch-level resume."""

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from streetgeo.core.checkpoints import (capture_rng_state, load_model, read_checkpoint, resolve_device,
                         restore_rng_state, save_checkpoint)
from streetgeo.core.config import ModelConfig, TrainConfig
from streetgeo.datasets.images import StreetViewDataset, collate_streetviews
from streetgeo.core.geography import haversine_km
from streetgeo.core.model import StreetGeoModel
from streetgeo.training.runtime import EpochBatchSampler, StopRequest, ignore_worker_interrupts, manifest_fingerprint


@torch.no_grad()
def evaluate(model, loader, device, stop_request=None):
    was_training = model.training
    model.eval()
    errors, total_loss, count = [], 0.0, 0
    try:
        for batch in loader:
            if stop_request is not None and stop_request.requested:
                return None
            batch = {name: value.to(device) for name, value in batch.items()}
            output = model(**batch, top_k=1)
            size = batch["images"].shape[0]
            count += size
            total_loss += output.loss.item() * size
            errors.append(haversine_km(output.coordinates, batch["targets"]).cpu())
        errors = torch.cat(errors)
        metrics = {"loss": total_loss / count, "mean_km": errors.mean().item(),
                   "median_km": torch.quantile(errors, 0.5).item()}
        for percentile in (90, 95, 99):
            metrics[f"p{percentile}_km"] = torch.quantile(errors, percentile / 100).item()
        for radius in (1, 25, 100, 200, 750, 2500):
            metrics[f"within_{radius}km"] = (errors <= radius).double().mean().item()
        return metrics
    finally:
        model.train(was_training)


def smoke_test(device):
    torch.manual_seed(1337)
    cfg = ModelConfig(image_size=32, patch_size=8, n_embed=48, n_head=4,
                      n_layer=2, fusion_layers=1, view_queries=2, cell_embed_dim=16,
                      coarse_lat_bins=3, coarse_lon_bins=6, fine_bins=2)
    model = StreetGeoModel(cfg).to(device)
    images = torch.randn(2, 2, 3, 32, 32, device=device)
    targets = torch.tensor([[34.0522, -118.2437], [-33.8688, 151.2093]], device=device)
    mask = torch.tensor([[True, False], [True, True]], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    output = model(images, targets, view_mask=mask)
    output.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not torch.isfinite(output.loss) or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("nonfinite loss or gradient in smoke test")
    optimizer.step()
    predictions = model.predict(images, view_mask=mask, top_k=2)
    print(json.dumps({"smoke_test": "passed", "device": str(device),
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "loss": output.loss.item(),
                      "random_untrained_output": [p[0].coordinates for p in predictions]}, indent=2))


def _run_config(args, saved=None):
    defaults = TrainConfig()
    fields = ("epochs", "batch_size", "learning_rate", "weight_decay", "grad_clip",
              "warmup_steps", "seed")
    result = {}
    for name in fields:
        supplied = getattr(args, name)
        result[name] = supplied if supplied is not None else (saved[name] if saved else getattr(defaults, name))
        if saved and supplied is not None and supplied != saved[name]:
            raise ValueError(f"--resume requires unchanged {name}; use --init-from for a new training stage")
    result["amp"] = args.amp if args.amp is not None else (saved["amp"] if saved else "auto")
    if saved and result["amp"] != saved["amp"]:
        raise ValueError("--resume requires the same AMP mode")
    for name in ("epochs", "batch_size"):
        if result[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if result["learning_rate"] <= 0 or result["grad_clip"] <= 0:
        raise ValueError("learning-rate and grad-clip must be positive")
    if result["weight_decay"] < 0 or result["warmup_steps"] < 0:
        raise ValueError("weight-decay and warmup-steps cannot be negative")
    if not all(math.isfinite(result[name]) for name in ("learning_rate", "weight_decay", "grad_clip")):
        raise ValueError("training hyperparameters must be finite")
    return result


def train(args):
    device = resolve_device(args.device)
    if args.smoke_test:
        smoke_test(device)
        return
    dataset_roots = list(getattr(args, "dataset_root", []) or [])
    if dataset_roots and (args.resume or args.train_manifest or args.val_manifest):
        raise ValueError("原始数据目录与 --train-manifest/--val-manifest/--resume 分开使用；续训只需指定 --resume")
    if args.resume and args.init_from:
        raise ValueError("use either --resume or --init-from")
    if args.save_every_steps <= 0 or args.save_every_seconds < 0 or args.num_workers < 0 or not math.isfinite(args.save_every_seconds):
        raise ValueError("save-every-steps must be positive; seconds/workers cannot be negative")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        raise ValueError("stop-after-steps must be positive")
    checkpoint = read_checkpoint(args.resume) if args.resume else None
    state = checkpoint["training_state"] if checkpoint else None
    if state is not None and state.get("resume_version") != 1:
        raise ValueError("This is an inference-only checkpoint. Use --init-from, or resume a *.last.pt file")
    run = _run_config(args, state["run_config"] if state else None)
    output_path = Path(args.output or (state["output_path"] if state else TrainConfig().model_path)).resolve()
    last_path = output_path.with_name(output_path.stem + ".last" + output_path.suffix)
    if not state and (last_path.exists() or output_path.exists()):
        raise FileExistsError(f"Training output already exists. Use --resume {last_path}, or choose a new --output")
    prepared = None
    if dataset_roots:
        from streetgeo.datasets.sources import PreparationCancelled
        from streetgeo.datasets.prepare import prepare_datasets
        prepared = Path(getattr(args, "data_output", None) or "data/world_osv_514m").resolve()
        try:
            prepare_datasets(roots=dataset_roots, output_dir=prepared, seed=run["seed"],
                             progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True),
                             cancelled=lambda: bool(getattr(args, "stop_file", None) and Path(args.stop_file).exists()))
        except (PreparationCancelled, KeyboardInterrupt):
            print("数据准备已停止，可以重新运行；原始图片未修改。", flush=True)
            return
    random.seed(run["seed"])
    torch.manual_seed(run["seed"])
    train_manifest = str(prepared / "train.jsonl") if prepared else (args.train_manifest or (state["train_manifest"] if state else None))
    val_manifest = str(prepared / "val.jsonl") if prepared else (args.val_manifest or (state["val_manifest"] if state else None))
    if not train_manifest:
        raise ValueError("provide --dataset-root, --train-manifest, --resume, or --smoke-test")
    fingerprint = manifest_fingerprint(train_manifest)
    val_fingerprint = manifest_fingerprint(val_manifest) if val_manifest else None
    if state and (fingerprint != state["train_fingerprint"] or val_fingerprint != state["val_fingerprint"]):
        raise ValueError("Manifests changed since checkpoint; resume would change the data order. Use --init-from")
    if checkpoint:
        cfg = ModelConfig(**checkpoint["model_config"])
    elif args.init_from:
        cfg = ModelConfig(**read_checkpoint(args.init_from)["model_config"])
    else:
        cfg = ModelConfig()
    if args.model_config:
        supplied_config = ModelConfig(**json.loads(Path(args.model_config).read_text(encoding="utf-8")))
        if (checkpoint or args.init_from) and asdict(supplied_config) != asdict(cfg):
            raise ValueError("model-config is incompatible with checkpoint")
        cfg = supplied_config
    train_data = StreetViewDataset(train_manifest, cfg, osv5m_only=True)
    loader_options = {"num_workers": args.num_workers, "collate_fn": collate_streetviews,
                      "pin_memory": device.type == "cuda", "worker_init_fn": ignore_worker_interrupts}
    val_loader = None
    if val_manifest:
        val_loader = DataLoader(StreetViewDataset(val_manifest, cfg, osv5m_only=True), batch_size=run["batch_size"],
                               generator=torch.Generator().manual_seed(run["seed"] + 9999), **loader_options)
    print(json.dumps({"event": "training_dataset", "scope": "world", "samples": len(train_data),
                      "sources": train_data.source_counts, "bounds": train_data.geographic_bounds,
                      "validation_samples": len(val_loader.dataset) if val_loader else 0}, ensure_ascii=False), flush=True)
    model = StreetGeoModel(cfg).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif args.init_from:
        model = load_model(args.init_from, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=run["learning_rate"], weight_decay=run["weight_decay"])
    amp_mode = run["amp"]
    if amp_mode == "auto":
        amp_mode = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "off"
    if device.type != "cuda" and amp_mode != "off":
        raise ValueError("fp16/bf16 training requires CUDA in this project; choose --amp off for CPU")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_mode == "fp16")
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    batches_per_epoch = math.ceil(len(train_data) / run["batch_size"])
    total_steps = run["epochs"] * batches_per_epoch
    warmup = min(run["warmup_steps"], max(0, total_steps - 1))
    epoch, next_batch, step = 0, 0, 0
    epoch_loss, epoch_examples, best_metric = 0.0, 0, float("inf")
    if state:
        if state["amp_mode"] != amp_mode:
            raise ValueError("Resolved AMP mode changed; use the same mode/device for --resume")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        epoch, next_batch, step = state["epoch"], state["next_batch"], state["global_step"]
        epoch_loss, epoch_examples, best_metric = state["epoch_loss"], state["epoch_examples"], state["best_metric"]
        restore_rng_state(state["rng"])
    last_saved_at, started_step = time.monotonic(), step

    def save_progress(reason):
        nonlocal last_saved_at
        save_checkpoint(model, last_path, keep_previous=True, training_state={
            "resume_version": 1, "dataset_source": "OpenStreetView-5M", "run_config": run, "amp_mode": amp_mode,
            "epoch": epoch, "next_batch": next_batch, "global_step": step,
            "epoch_loss": epoch_loss, "epoch_examples": epoch_examples, "best_metric": best_metric,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": capture_rng_state(),
            "lr_schedule": {"total_steps": total_steps, "warmup_steps": warmup, "step": step},
            "train_manifest": str(Path(train_manifest).resolve()), "train_fingerprint": fingerprint,
            "val_manifest": str(Path(val_manifest).resolve()) if val_manifest else None,
            "val_fingerprint": val_fingerprint, "output_path": str(output_path), "reason": reason,
        })
        last_saved_at = time.monotonic()
        print(json.dumps({"checkpoint": str(last_path), "step": step, "reason": reason}), flush=True)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}; device: {device}; AMP: {amp_mode}", flush=True)
    print(json.dumps({"event": "training_plan", "epochs": run["epochs"],
                      "batches_per_epoch": batches_per_epoch, "total_steps": total_steps,
                      "step": step, "epoch": epoch + 1, "output": str(output_path),
                      "resume_checkpoint": str(last_path)}), flush=True)
    if state:
        print(f"Resumed epoch {epoch + 1}, next batch {next_batch}, global step {step}", flush=True)
    else:
        save_progress("initial")
    try:
        with StopRequest(getattr(args, "stop_file", None)) as stop:
            if stop.requested:
                save_progress("graceful_stop")
                return
            while epoch < run["epochs"]:
                model.train()
                sampler = EpochBatchSampler(len(train_data), run["batch_size"], run["seed"], epoch, next_batch)
                # DataLoader's worker base seed must not consume the model's RNG.
                loader = DataLoader(train_data, batch_sampler=sampler, **loader_options,
                                    generator=torch.Generator().manual_seed(run["seed"] + epoch + 10000))
                for batch in loader:
                    if step < warmup:
                        factor = (step + 1) / warmup
                    else:
                        progress = (step - warmup) / max(1, total_steps - warmup - 1)
                        factor = 0.1 + 0.45 * (1 + math.cos(math.pi * progress))
                    for group in optimizer.param_groups:
                        group["lr"] = run["learning_rate"] * factor
                    batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_mode != "off"):
                        result = model(**batch, top_k=1)
                    if not torch.isfinite(result.loss):
                        raise RuntimeError(f"nonfinite loss at step {step}")
                    scaler.scale(result.loss).backward()
                    scaler.unscale_(optimizer)
                    # FP16 overflow is handled by GradScaler's skipped step and
                    # scale reduction. FP32/BF16 nonfinite gradients are errors.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), run["grad_clip"],
                                                   error_if_nonfinite=amp_mode != "fp16")
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += result.loss.item() * batch["images"].shape[0]
                    epoch_examples += batch["images"].shape[0]
                    step += 1
                    next_batch += 1
                    if step % args.log_every_steps == 0:
                        print(json.dumps({"epoch": epoch + 1, "step": step, "loss": result.loss.item(),
                                          "total_steps": total_steps, "batch": next_batch,
                                          "learning_rate": optimizer.param_groups[0]["lr"]}), flush=True)
                    if stop.requested or (args.stop_after_steps and step - started_step >= args.stop_after_steps):
                        save_progress("graceful_stop")
                        return
                    elapsed = args.save_every_seconds and time.monotonic() - last_saved_at >= args.save_every_seconds
                    if step % args.save_every_steps == 0 or elapsed:
                        save_progress("periodic")
                metrics = {"epoch": epoch + 1, "step": step, "train_loss": epoch_loss / epoch_examples}
                if val_loader is not None:
                    validation = evaluate(model, val_loader, device, stop_request=stop)
                    if validation is None:
                        save_progress("graceful_stop_during_validation")
                        return
                    metrics["validation"] = validation
                print(json.dumps(metrics), flush=True)
                criterion = metrics["validation"]["median_km"] if val_loader is not None else metrics["train_loss"]
                if criterion < best_metric:
                    best_metric = criterion
                    save_checkpoint(model, output_path, training_state={"metrics": metrics, "global_step": step})
                epoch += 1
                next_batch, epoch_loss, epoch_examples = 0, 0.0, 0
                save_progress("epoch_end")
                if stop.requested:
                    return
    except KeyboardInterrupt:
        print(f"Forced stop. Resume the last committed checkpoint: {last_path}", flush=True)
        raise
    except Exception:
        print(f"Training failed. Last committed progress remains available at {last_path}", flush=True)
        raise
    print(f"Training complete. Best weights: {output_path}; resume checkpoint: {last_path}")


def main():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", action="append", default=[], help="OpenStreetView-5M directory; partial image downloads supported")
    parser.add_argument("--data-output", help="frozen generated manifests directory; default data/world_osv_514m")
    parser.add_argument("--train-manifest", help="prepared OpenStreetView-5M training JSONL")
    parser.add_argument("--val-manifest", help="prepared OpenStreetView-5M validation JSONL")
    parser.add_argument("--model-config")
    parser.add_argument("--resume", help="full *.last.pt checkpoint; continue the same run")
    parser.add_argument("--init-from", help="model weights for a NEW training stage")
    for name in ("epochs", "batch_size", "warmup_steps", "seed"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("learning_rate", "weight_decay", "grad_clip"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--amp", choices=("auto", "fp16", "bf16", "off"))
    parser.add_argument("--output")
    parser.add_argument("--save-every-steps", type=int, default=defaults.save_every_steps)
    parser.add_argument("--save-every-seconds", type=float, default=defaults.save_every_seconds)
    parser.add_argument("--log-every-steps", type=int, default=defaults.log_every_steps)
    parser.add_argument("--stop-after-steps", type=int, help="graceful stop after N additional batches")
    parser.add_argument("--stop-file", help="GUI stop request file; checked at safe batch boundaries")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.log_every_steps <= 0:
        parser.error("log-every-steps must be positive")
    train(args)


if __name__ == "__main__":
    main()
