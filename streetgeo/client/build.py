"""Build a self-contained prediction folder, preserving an existing best.pt."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from streetgeo.paths import ROOT

PACKAGE_FILES = (
    "__init__.py", "paths.py", "core/__init__.py", "core/config.py", "core/model.py",
    "core/geography.py", "core/checkpoints.py", "datasets/__init__.py", "datasets/images.py",
    "inference/__init__.py", "inference/backend.py", "inference/predict.py", "maps/__init__.py",
    "maps/basemap.py", "maps/projection.py", "maps/widgets.py", "ui/__init__.py", "ui/common.py", "ui/prediction.py",
)
MAP_FILES = ("un_countries.geojson", "un_boundaries.geojson", "un_map_sources.json", "us_states.geojson")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_client(output, checkpoint=None):
    output = Path(output).expanduser().resolve()
    if output == ROOT or output.is_relative_to(ROOT / "streetgeo"):
        raise ValueError("Client output must be a separate distribution directory")
    target_checkpoint = output / "best.pt"
    if checkpoint is None:
        if target_checkpoint.is_file():
            checkpoint = target_checkpoint
        else:
            candidates = [p for p in (ROOT / "checkpoints").rglob("*.pt")
                          if ".last" not in p.name and ".previous" not in p.name]
            if not candidates:
                raise ValueError("请先把已训练权重放到客户端 best.pt，或传入 --checkpoint")
            checkpoint = max(candidates, key=lambda p: p.stat().st_mtime)
    checkpoint = Path(checkpoint).expanduser().resolve()
    import torch
    from streetgeo.core.checkpoints import ARCHITECTURE
    from streetgeo.core.config import ModelConfig
    state = torch.load(checkpoint, map_location="meta", weights_only=True)
    if state.get("architecture") != ARCHITECTURE:
        raise ValueError("不是 StreetGeo 视觉模型 checkpoint")
    ModelConfig(**state["model_config"])
    model_info = {"checkpoint": "best.pt", "architecture": state["architecture"],
                  "model_config": state["model_config"], "parameters": sum(t.numel() for t in state["model_state_dict"].values()),
                  "bytes": checkpoint.stat().st_size, "sha256": sha256(checkpoint)}
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint != target_checkpoint.resolve():
        shutil.copy2(checkpoint, target_checkpoint)
    generated = []

    def copy(source, relative):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        generated.append(str(relative).replace("\\", "/"))

    for relative in PACKAGE_FILES:
        copy(ROOT / "streetgeo" / relative, Path("streetgeo") / relative)
    for filename in MAP_FILES:
        copy(ROOT / "assets" / "maps" / filename, Path("assets/maps") / filename)
    templates = Path(__file__).resolve().parent
    for filename in ("bootstrap.py", "app.py", "requirements.txt", "requirements-torch-cpu.txt"):
        copy(templates / filename, filename)
    copy(templates / "tk_runtime.py", "tkinter_runtime.py")
    for source in sorted((templates / "vendor").rglob("*")):
        if source.is_file():
            copy(source, source.relative_to(templates))
    copy(templates / "launcher.bat", "启动客户端.bat")
    copy(templates / "README.md", "README.md")
    (output / "MODEL_INFO.json").write_text(json.dumps(model_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / ".gitignore").write_text(".runtime/\n__pycache__/\n*.pyc\n", encoding="utf-8")
    manifest = {"application": "StreetGeo prediction client", "created_at": datetime.now(timezone.utc).isoformat(),
                "entry": "启动客户端.bat", "python": "Windows x64 Python 3.10-3.15", "default_device": "cpu",
                "files": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
                          for name in sorted(generated)}}
    (output / "CLIENT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "generated_files": len(generated), "model_parameters": model_info["parameters"],
            "model_mib": model_info["bytes"] / 1024 ** 2, "model_preserved": checkpoint == target_checkpoint.resolve()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="clinet")
    parser.add_argument("--checkpoint", help="optional model to copy as best.pt")
    args = parser.parse_args()
    print(json.dumps(build_client(args.output, args.checkpoint), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
