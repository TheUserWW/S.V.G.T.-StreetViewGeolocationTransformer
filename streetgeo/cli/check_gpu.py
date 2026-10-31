"""Check real CUDA/BF16 training capacity using synthetic data, without saving weights."""

import argparse
import json
from pathlib import Path
import time

import torch

from streetgeo.core.config import ModelConfig
from streetgeo.core.model import StreetGeoModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-config", type=Path, help="optional model configuration JSON")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA available; install a compatible CUDA PyTorch build")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16 training")
    config = ModelConfig(**json.loads(args.model_config.read_text(encoding="utf-8"))) if args.model_config else ModelConfig()
    model = StreetGeoModel(config).cuda().train()
    images = torch.randn(args.batch_size, 3, config.image_size, config.image_size, device="cuda")
    targets = torch.tensor([[34.05, -118.25]], device="cuda").expand(args.batch_size, -1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(images, targets, top_k=1)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    print(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                      "batch_size": args.batch_size, "amp": "bf16", "synthetic_steps": 3,
                      "seconds": time.monotonic() - started,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "fp32_weights_mib": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2,
                      "loss": output.loss.item(), "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
