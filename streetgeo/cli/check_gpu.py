"""Check CUDA training capacity using synthetic data, without saving weights."""

import argparse
import json
from pathlib import Path
import time

import torch

from streetgeo.core.config import ModelConfig, TrainConfig
from streetgeo.core.model import StreetGeoModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=TrainConfig().batch_size)
    parser.add_argument("--model-config", type=Path, help="optional model configuration JSON")
    parser.add_argument("--amp", choices=("auto", "bf16", "fp16", "off"), default="bf16")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA available; install a compatible CUDA PyTorch build")
    amp = args.amp
    if amp == "auto":
        amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16 training")
    config = ModelConfig(**json.loads(args.model_config.read_text(encoding="utf-8"))) if args.model_config else ModelConfig()
    model = StreetGeoModel(config).cuda().train()
    images = torch.randn(args.batch_size, 3, config.image_size, config.image_size, device="cuda")
    targets = torch.tensor([[34.05, -118.25]], device="cuda").expand(args.batch_size, -1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    completed_steps, attempted_steps = 0, 0
    while completed_steps < 3 and attempted_steps < 12:
        attempted_steps += 1
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=amp != "off"):
            output = model(images, targets, top_k=1)
        if not torch.isfinite(output.loss):
            raise RuntimeError("Nonfinite loss during GPU capacity check")
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=amp != "fp16")
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            completed_steps += 1
    if completed_steps != 3:
        raise RuntimeError("FP16 gradient overflows prevented three optimizer updates; try BF16")
    torch.cuda.synchronize()
    print(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                      "batch_size": args.batch_size, "amp": amp, "synthetic_steps": completed_steps,
                      "attempted_steps": attempted_steps,
                      "seconds": time.monotonic() - started,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "fp32_weights_mib": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2,
                      "loss": output.loss.item(), "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
