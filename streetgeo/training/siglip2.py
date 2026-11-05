"""Import the pinned Google SigLIP2 FixRes vision tower into native S.V.G.T."""

import hashlib
from pathlib import Path

import torch

from streetgeo.core.config import SIGLIP2_REPO, SIGLIP2_WEIGHTS_PATH
from streetgeo.paths import ROOT

SIGLIP2_SHA256 = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"


def load_pretrained_vision(model, weights=None):
    """Load official Safetensors, keeping the geolocation pooling randomly initialized."""
    if not model.config.is_siglip2:
        raise ValueError("SigLIP2 weights require the NMPZ model configuration")
    from safetensors import safe_open

    if weights is None:
        cached = ROOT / SIGLIP2_WEIGHTS_PATH
        weights = cached if cached.is_file() else None
    if weights is None:
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download(SIGLIP2_REPO, "model.safetensors",
                                  revision=model.config.backbone_revision, token=False)
    weights = Path(weights)
    if not weights.is_file() or weights.suffix.lower() != ".safetensors":
        raise ValueError("Provide the official SigLIP2 model.safetensors file")
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SIGLIP2_SHA256:
        raise ValueError("SigLIP2 weights do not match the pinned Google release")

    with safe_open(weights, framework="pt", device="cpu") as source:
        def read(name):
            return source.get_tensor("vision_model." + name)

        vision = {"position_embedding": read("embeddings.position_embedding.weight").unsqueeze(0)}
        for parameter in ("weight", "bias"):
            vision[f"patch_embedding.{parameter}"] = read(f"embeddings.patch_embedding.{parameter}")
            vision[f"visual_norm.{parameter}"] = read(f"post_layernorm.{parameter}")
        layer_names = {
            "ln1": "layer_norm1", "ln2": "layer_norm2", "attention.proj": "self_attn.out_proj",
            "ffn.0": "mlp.fc1", "ffn.2": "mlp.fc2",
        }
        for index in range(model.config.n_layer):
            prefix = f"encoder.layers.{index}."
            destination = f"visual_blocks.{index}."
            for parameter in ("weight", "bias"):
                for local, upstream in layer_names.items():
                    vision[f"{destination}{local}.{parameter}"] = read(f"{prefix}{upstream}.{parameter}")
                vision[f"{destination}attention.qkv.{parameter}"] = torch.cat([
                    read(f"{prefix}self_attn.{projection}_proj.{parameter}") for projection in ("q", "k", "v")
                ])
        expected = model.encoder.state_dict()
        for name, tensor in vision.items():
            if tensor.shape != expected[name].shape:
                raise ValueError(f"Incompatible SigLIP2 tensor {name}: {tensor.shape}, expected {expected[name].shape}")
        model.encoder.load_state_dict({**expected, **vision}, strict=True)
