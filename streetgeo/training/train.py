"""Train on Mapillary with atomic checkpoints and batch-level resume."""

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from streetgeo.core.checkpoints import (capture_rng_state, read_checkpoint, resolve_device,
                         restore_rng_state, save_checkpoint)
from streetgeo.core.config import ModelConfig, TrainConfig
from streetgeo.datasets.images import StreetViewDataset, EnvironmentViewDataset, collate_streetviews
from streetgeo.core.geography import haversine_km
from streetgeo.core.model import StreetGeoModel
from streetgeo.paths import validate_checkpoint_path
from streetgeo.training.runtime import AlternatingBatchSampler, EpochBatchSampler, StopRequest, ignore_worker_interrupts, manifest_fingerprint
from streetgeo.training.environment import parameter_groups, train_mode, resolve_recipe


@torch.no_grad()
def evaluate(model, loader, device, stop_request=None, progress=None):
    was_training = model.training
    model.eval()
    errors, total_loss, count = [], 0.0, 0
    total_batches = len(loader)
    started = time.monotonic()
    def report(batch):
        if progress:
            progress({"batch": batch, "total_batches": total_batches, "samples": count,
                      "total_samples": len(loader.dataset), "seconds": time.monotonic() - started})
    try:
        report(0)
        for batch_number, batch in enumerate(loader, 1):
            if stop_request is not None and stop_request.requested:
                return None
            batch = {name: value.to(device) for name, value in batch.items() if name != "training_phase"}
            output = model(**batch, top_k=1)
            size = batch["images"].shape[0]
            count += size
            total_loss += output.loss.item() * size
            errors.append(haversine_km(output.coordinates, batch["targets"]).cpu())
            if batch_number == 1 or batch_number % 10 == 0 or batch_number == total_batches:
                report(batch_number)
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
    result = {}
    for name, default in (("environment_specialist", False), ("unfreeze_last", 2), ("visual_learning_rate", 5e-6),
                          ("road_mask_top_width", .20), ("road_mask_bottom_width", 1.), ("road_mask_height", .45)):
        supplied = getattr(args, name, None)
        result[name] = supplied if supplied is not None else (saved.get(name, default) if saved else default)
        if saved and supplied is not None and supplied != saved.get(name, default):
            raise ValueError(f"--resume requires unchanged {name}; use --init-from for a new training stage")
    fields = ("epochs", "batch_size", "learning_rate", "weight_decay", "grad_clip",
              "warmup_steps", "seed")
    stage_defaults = {"epochs": 2, "learning_rate": 2e-5, "warmup_steps": 100} if result["environment_specialist"] else {}
    for name in fields:
        supplied = getattr(args, name)
        default = stage_defaults.get(name, getattr(defaults, name))
        result[name] = supplied if supplied is not None else (saved[name] if saved else default)
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
    if result["unfreeze_last"] <= 0 or not math.isfinite(result["visual_learning_rate"]) or result["visual_learning_rate"] <= 0:
        raise ValueError("unfreeze-last and visual-learning-rate must be positive and finite")
    return result


def train(args):
    device = resolve_device(args.device)
    if args.smoke_test:
        smoke_test(device)
        return
    dataset_roots = list(getattr(args, "dataset_root", []) or [])
    if getattr(args, "download", False) and (dataset_roots or args.resume or args.train_manifest or args.val_manifest):
        raise ValueError("--download 与数据目录、清单和续训分开使用；先选择区域下载，再自动准备新训练")
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
    if state and state.get("dataset_source") not in (None, "Mapillary"):
        raise ValueError("预训练续训仅接受 Mapillary checkpoint；请使用 Mapillary 数据新建训练")
    run = _run_config(args, state["run_config"] if state else None)
    if run["environment_specialist"] and not (args.init_from or state):
        raise ValueError("环境专项微调需要 --init-from 指定已有模型，或使用 --resume 继续专项进度")
    recipe, road_mask, legacy_mask = resolve_recipe(run, state if run["environment_specialist"] else None,
                                                   explicit_geometry=any(getattr(args, name, None) is not None for name in
                                                   ("road_mask_top_width", "road_mask_bottom_width", "road_mask_height")))
    output_path = validate_checkpoint_path(args.output or (state["output_path"] if state else TrainConfig().model_path)).resolve()
    last_path = output_path.with_name(output_path.stem + ".last" + output_path.suffix)
    if not state and (last_path.exists() or output_path.exists()):
        raise FileExistsError(f"Training output already exists. Use --resume {last_path}, or choose a new --output")
    if getattr(args, "download", False):
        from streetgeo.datasets.download_mapillary import run_collection
        collection_args = argparse.Namespace(**vars(args))
        collection_args.output, collection_args.seed = args.download_output, run["seed"]
        outcome = run_collection(collection_args)
        if not outcome or outcome in ("stopped", "paused"):
            print("下载已暂停；重新运行相同命令可继续。尚未启动训练。", flush=True)
            return
        dataset_roots = [args.download_output]
    prepared = None
    if dataset_roots:
        from streetgeo.datasets.sources import PreparationCancelled
        from streetgeo.datasets.prepare import prepare_datasets
        prepared = Path(getattr(args, "data_output", None) or "data/world_mapillary").resolve()
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
    if run["environment_specialist"] and not val_manifest:
        raise ValueError("环境专项微调需要独立的 Mapillary 验证清单，以分别评估全图和专项")
    fingerprint = manifest_fingerprint(train_manifest)
    val_fingerprint = manifest_fingerprint(val_manifest) if val_manifest else None
    if state and (fingerprint != state["train_fingerprint"] or val_fingerprint != state["val_fingerprint"]):
        raise ValueError("Manifests changed since checkpoint; resume would change the data order. Use --init-from")
    initial_checkpoint = read_checkpoint(args.init_from) if args.init_from else None
    if checkpoint:
        cfg = ModelConfig(**checkpoint["model_config"])
    elif initial_checkpoint:
        cfg = ModelConfig(**initial_checkpoint["model_config"])
    else:
        cfg = ModelConfig()
    if args.model_config:
        supplied_config = ModelConfig(**json.loads(Path(args.model_config).read_text(encoding="utf-8")))
        if (checkpoint or args.init_from) and asdict(supplied_config) != asdict(cfg):
            raise ValueError("model-config is incompatible with checkpoint")
        cfg = supplied_config
    if run["environment_specialist"] and run["unfreeze_last"] > cfg.n_layer:
        raise ValueError("unfreeze-last exceeds the checkpoint's number of visual layers")
    train_data = StreetViewDataset(train_manifest, cfg, mapillary_only=True,
                                  environment_specialist=run["environment_specialist"], road_mask=road_mask,
                                  legacy_environment_mask=legacy_mask)
    loader_options = {"num_workers": args.num_workers, "collate_fn": collate_streetviews,
                      "pin_memory": device.type == "cuda", "worker_init_fn": ignore_worker_interrupts}
    val_loader = environment_loader = None
    if val_manifest:
        val_data = StreetViewDataset(val_manifest, cfg, mapillary_only=True,
                                    environment_specialist=run["environment_specialist"], road_mask=road_mask,
                                    legacy_environment_mask=legacy_mask)
        if run["environment_specialist"]:
            train_ids = {paths[0].stem for paths, _, _ in train_data.samples}
            if any(paths[0].stem in train_ids for paths, _, _ in val_data.samples):
                raise ValueError("环境专项微调的训练集与验证集包含相同 Mapillary 图片；请重新生成独立划分")
            del train_ids
        val_loader = DataLoader(val_data, batch_size=run["batch_size"],
                               generator=torch.Generator().manual_seed(run["seed"] + 9999), **loader_options)
        if run["environment_specialist"]:
            environment_loader = DataLoader(EnvironmentViewDataset(val_data), batch_size=run["batch_size"],
                                           generator=torch.Generator().manual_seed(run["seed"] + 9998), **loader_options)
    print(json.dumps({"event": "training_dataset", "scope": "world", "samples": len(train_data),
                      "sources": train_data.source_counts, "bounds": train_data.geographic_bounds,
                      "validation_samples": len(val_loader.dataset) if val_loader else 0}, ensure_ascii=False), flush=True)
    model = StreetGeoModel(cfg).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif initial_checkpoint:
        model.load_state_dict(initial_checkpoint["model_state_dict"])
    del initial_checkpoint
    groups = model.parameters()
    if run["environment_specialist"]:
        groups = parameter_groups(model, unfreeze_last=run["unfreeze_last"], learning_rate=run["learning_rate"],
                                  visual_learning_rate=run["visual_learning_rate"], weight_decay=run["weight_decay"])
    optimizer = torch.optim.AdamW(groups, lr=run["learning_rate"], weight_decay=run["weight_decay"])
    amp_mode = run["amp"]
    if amp_mode == "auto":
        amp_mode = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "off"
    if device.type != "cuda" and amp_mode != "off":
        raise ValueError("fp16/bf16 training requires CUDA in this project; choose --amp off for CPU")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_mode == "fp16")
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    source_batches_per_epoch = math.ceil(len(train_data) / run["batch_size"])
    batches_per_epoch = source_batches_per_epoch * (2 if run["environment_specialist"] else 1)
    total_steps = run["epochs"] * batches_per_epoch
    warmup = min(run["warmup_steps"], max(0, total_steps - 1))
    epoch, next_batch, step = 0, 0, 0
    epoch_loss, epoch_examples, best_metric = 0.0, 0, float("inf")
    baseline_metrics = state.get("baseline_metrics") if state else None
    if state:
        if state["amp_mode"] != amp_mode:
            raise ValueError("Resolved AMP mode changed; use the same mode/device for --resume")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        epoch, next_batch, step = state["epoch"], state["next_batch"], state["global_step"]
        expected_phase = "environment" if run["environment_specialist"] and next_batch % 2 else "full"
        if state.get("next_phase", expected_phase) != expected_phase:
            raise ValueError("checkpoint phase is inconsistent with its batch position")
        epoch_loss, epoch_examples, best_metric = state["epoch_loss"], state["epoch_examples"], state["best_metric"]
        restore_rng_state(state["rng"])
    last_saved_at, started_step = time.monotonic(), step

    def save_progress(reason):
        nonlocal last_saved_at
        print(json.dumps({"event": "training_phase", "phase": "saving", "reason": reason}), flush=True)
        save_checkpoint(model, last_path, keep_previous=True, training_state={
            "resume_version": 1, "dataset_source": "Mapillary", "run_config": run, "amp_mode": amp_mode,
            "epoch": epoch, "next_batch": next_batch, "global_step": step,
            "next_phase": "environment" if run["environment_specialist"] and next_batch % 2 else "full",
            "baseline_metrics": baseline_metrics,
            "environment_recipe": recipe if run["environment_specialist"] else None,
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
                      "source_batches_per_epoch": source_batches_per_epoch,
                      "environment_specialist": run["environment_specialist"],
                      "environment_recipe": recipe if run["environment_specialist"] else None,
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
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
            def validation_pass(loader, view, *, baseline=False):
                return evaluate(model, loader, device, stop_request=stop,
                                progress=lambda details: print(json.dumps({"event": "validation_progress",
                                    "epoch": epoch + 1, "validation_view": view, "baseline": baseline,
                                    **details}), flush=True))

            if run["environment_specialist"] and baseline_metrics is None and step == 0:
                baseline_full = validation_pass(val_loader, "full", baseline=True)
                baseline_environment = validation_pass(environment_loader, "environment", baseline=True) if baseline_full else None
                if baseline_environment is None:
                    save_progress("graceful_stop_during_baseline")
                    return
                baseline_metrics = {"full": baseline_full, "environment": baseline_environment}
                print(json.dumps({"event": "baseline_evaluation", **baseline_metrics}), flush=True)
                best_metric = baseline_full["median_km"]
                save_checkpoint(model, output_path, training_state={"global_step": step, "dataset_source": "Mapillary",
                                "run_config": run, "baseline_metrics": baseline_metrics, "environment_recipe": recipe,
                                "metrics": {"validation": baseline_full, "environment_validation": baseline_environment}})
                save_progress("baseline_complete")
            while epoch < run["epochs"]:
                if stop.requested:
                    save_progress("graceful_stop")
                    return
                if run["environment_specialist"]:
                    train_mode(model, run["unfreeze_last"])
                else:
                    model.train()
                if next_batch < batches_per_epoch:
                    print(json.dumps({"event": "training_phase", "phase": "training", "epoch": epoch + 1}), flush=True)
                sampler_type = AlternatingBatchSampler if run["environment_specialist"] else EpochBatchSampler
                sampler = sampler_type(len(train_data), run["batch_size"], run["seed"], epoch, next_batch)
                # DataLoader's worker base seed must not consume the model's RNG.
                # An epoch interrupted during validation has no training batches left.
                # Avoid spawning Windows workers and copying the entire training dataset.
                loader = () if next_batch == batches_per_epoch else DataLoader(
                    train_data, batch_sampler=sampler, **loader_options,
                    generator=torch.Generator().manual_seed(run["seed"] + epoch + 10000))
                for batch in loader:
                    phase = "environment" if int(batch.pop("training_phase", 0)) else "full"
                    if step < warmup:
                        factor = (step + 1) / warmup
                    else:
                        progress = (step - warmup) / max(1, total_steps - warmup - 1)
                        factor = 0.1 + 0.45 * (1 + math.cos(math.pi * progress))
                    for group in optimizer.param_groups:
                        group["lr"] = group.get("base_learning_rate", run["learning_rate"]) * factor
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
                                          "input_view": phase,
                                          "total_steps": total_steps, "batch": next_batch,
                                          "learning_rate": optimizer.param_groups[-1]["lr"],
                                          "learning_rates": {group.get("name", "all"): group["lr"] for group in optimizer.param_groups}}), flush=True)
                    if stop.requested or (args.stop_after_steps and step - started_step >= args.stop_after_steps):
                        save_progress("graceful_stop")
                        return
                    elapsed = args.save_every_seconds and time.monotonic() - last_saved_at >= args.save_every_seconds
                    if step % args.save_every_steps == 0 or elapsed:
                        save_progress("periodic")
                metrics = {"epoch": epoch + 1, "step": step, "train_loss": epoch_loss / epoch_examples}
                if val_loader is not None:
                    validation = validation_pass(val_loader, "full")
                    if validation is None:
                        save_progress("graceful_stop_during_validation")
                        return
                    metrics["validation"] = validation
                    if environment_loader is not None:
                        masked_validation = validation_pass(environment_loader, "environment")
                        if masked_validation is None:
                            save_progress("graceful_stop_during_environment_validation")
                            return
                        metrics["environment_validation"] = masked_validation
                print(json.dumps(metrics), flush=True)
                criterion = metrics["validation"]["median_km"] if val_loader is not None else metrics["train_loss"]
                if criterion < best_metric:
                    best_metric = criterion
                    print(json.dumps({"event": "training_phase", "phase": "saving", "reason": "best_model"}), flush=True)
                    save_checkpoint(model, output_path, training_state={"metrics": metrics, "global_step": step,
                                    "dataset_source": "Mapillary", "run_config": run, "baseline_metrics": baseline_metrics,
                                    "environment_recipe": recipe if run["environment_specialist"] else None})
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
    parser.add_argument("--dataset-root", action="append", default=[], help="Mapillary API download directory containing all.jsonl")
    parser.add_argument("--data-output", help="frozen generated manifests directory; default data/world_mapillary")
    parser.add_argument("--train-manifest", help="prepared Mapillary training JSONL")
    parser.add_argument("--val-manifest", help="prepared Mapillary validation JSONL")
    parser.add_argument("--model-config")
    parser.add_argument("--resume", help="full *.last.pt checkpoint; continue the same run")
    parser.add_argument("--init-from", help="model weights for a NEW training stage")
    parser.add_argument("--environment-specialist", action=argparse.BooleanOptionalAction, default=None,
                        help="fine-tune existing weights with full/masked alternating updates; requires validation")
    parser.add_argument("--unfreeze-last", type=int, help="last visual layers to unfreeze for environment fine-tuning; default 2")
    parser.add_argument("--visual-learning-rate", type=float, help="terminal visual-layer learning rate; default 5e-6")
    parser.add_argument("--road-mask-top-width", type=float, help="trapezoid top / source width in (0,1]; default 0.20")
    parser.add_argument("--road-mask-bottom-width", type=float, help="trapezoid bottom / source width in (0,1]; default 1.0")
    parser.add_argument("--road-mask-height", type=float, help="trapezoid height / source height in (0,1]; default 0.45")
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
    parser.add_argument("--download", action="store_true", help="download selected Mapillary imagery, prepare, then train")
    from streetgeo.datasets.download_mapillary import add_collection_arguments
    add_collection_arguments(parser, for_training=True)
    args = parser.parse_args()
    if args.log_every_steps <= 0:
        parser.error("log-every-steps must be positive")
    train(args)


if __name__ == "__main__":
    main()
