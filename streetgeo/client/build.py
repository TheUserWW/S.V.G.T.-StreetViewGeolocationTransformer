"""Build a bilingual prediction client and a model-free distribution archive."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
import shutil

from streetgeo import PROJECT_SHORT_NAME, PROJECT_NAME
from streetgeo.paths import MAP_ASSETS, ROOT

PACKAGE_FILES = (
    "__init__.py", "paths.py", "core/__init__.py", "core/config.py", "core/model.py",
    "core/geography.py", "core/checkpoints.py", "datasets/__init__.py", "datasets/images.py", "datasets/policy.py", "datasets/road_mask.py",
    "inference/__init__.py", "inference/backend.py", "inference/predict.py", "maps/__init__.py",
    "inference/coverage.py",
    "inference/explain.py", "inference/explanation_text.py", "ui/explanation.py",
    "maps/basemap.py", "maps/projection.py", "maps/widgets.py", "ui/__init__.py", "ui/common.py",
    "ui/desktop_input.py", "inference/sessions.py", "training/__init__.py", "training/online.py",
)
MAP_FILES = ("un_countries.geojson", "un_boundaries.geojson", "un_map_sources.json", "us_states.geojson",
             "google_streetview_regions.json")


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
    # Optional compatibility argument validates a model but never copies it.
    if checkpoint is not None:
        import torch
        from streetgeo.core.checkpoints import ARCHITECTURE
        from streetgeo.core.config import ModelConfig
        state = torch.load(Path(checkpoint).expanduser().resolve(), map_location="meta", weights_only=True)
        if not isinstance(state, dict) or state.get("architecture") != ARCHITECTURE:
            raise ValueError("Expected an S.V.G.T. visual checkpoint")
        ModelConfig(**state["model_config"])
    output.mkdir(parents=True, exist_ok=True)
    # Only retire previously generated modules whose content is still unmodified.
    previous_manifest = output / "CLIENT_MANIFEST.json"
    try:
        previous_files = json.loads(previous_manifest.read_text(encoding="utf-8")).get("files", {})
    except (OSError, ValueError, AttributeError):
        previous_files = {}
    for obsolete in ("streetgeo/ui/prediction.py", "streetgeo/inference/osv_test.py"):
        path = output / obsolete
        if path.is_file() and previous_files.get(obsolete, {}).get("sha256") == sha256(path):
            path.unlink()
    generated = []

    def copy(source, relative):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        generated.append(str(relative).replace("\\", "/"))

    for relative in PACKAGE_FILES:
        copy(ROOT / "streetgeo" / relative, Path("streetgeo") / relative)
    for filename in MAP_FILES:
        copy(MAP_ASSETS / filename, Path("assets/maps") / filename)
    templates = Path(__file__).resolve().parent
    for filename in ("bootstrap.py", "app.py", "requirements.txt", "requirements-torch-cpu.txt"):
        copy(templates / filename, filename)
    copy(templates / "tk_runtime.py", "tkinter_runtime.py")
    copy(templates / "prediction.py", "client_ui.py")
    copy(templates / "workbench.py", "daily_ui.py")
    copy(templates / "i18n.py", "client_i18n.py")
    for source in sorted((templates / "vendor").rglob("*")):
        if source.is_file():
            copy(source, source.relative_to(templates))
    copy(templates / "launcher.bat", "启动客户端.bat")
    copy(templates / "launcher.bat", "Launch Client.bat")
    copy(templates / "README.md", "README.md")
    (output / ".gitignore").write_text(".runtime/\n__pycache__/\n*.pyc\n", encoding="utf-8")
    manifest = {"application": f"{PROJECT_SHORT_NAME} prediction client", "project_name": PROJECT_NAME,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "entry": "启动客户端.bat", "python": "Windows x64 Python 3.10-3.15", "default_device": "cpu",
                "languages": ["zh", "en"], "features": ["model_file_selection", "street_view_images", "map_predictions",
                                                          "attention_heatmaps", "manual_occlusion", "explanation_png_export",
                                                          "google_streetview_region_filter", "clipboard_images", "native_file_drop",
                                                          "manual_map_pin", "feedback_learning", "round_history", "base_personal_profiles",
                                                          "resizable_sidebar"], "model_included": False,
                "files": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
                          for name in sorted(generated)}}
    (output / "CLIENT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = build_archive(output)
    return {"output": str(output), "generated_files": len(generated), "model_included": False,
            "archive": str(archive), "archive_mib": archive.stat().st_size / 1024 ** 2}


MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin", ".h5", ".hdf5", ".pb", ".pkl", ".pickle", ".npz"}


def build_archive(output):
    """Package only generated application files; never weights or personal state."""
    output = Path(output).resolve()
    manifest = json.loads((output / "CLIENT_MANIFEST.json").read_text(encoding="utf-8"))
    names = sorted(set(manifest["files"]) | {"CLIENT_MANIFEST.json", ".gitignore"})
    for name in names:
        path = output / name
        if (not path.resolve().is_relative_to(output) or not path.is_file()
                or path.suffix.lower() in MODEL_SUFFIXES
                or any(part.lower() in (".runtime", ".gui", ".venv", "venv", ".cache", "__pycache__", "online_learning", "captures", "workspace")
                       for part in path.relative_to(output).parts)
                or path.name.lower() in ("settings.json", "history.json", "prediction.log")):
            raise ValueError(f"Refusing to include model weights, personal state, or invalid path: {name}")
    archive = output / (output.name + ".zip")
    descriptor, temporary = tempfile.mkstemp(prefix=archive.name + ".", suffix=".tmp", dir=output)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as bundle:
            for name in names:
                bundle.write(output / name, arcname=output.name + "/" + name)
        os.replace(temporary, archive)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return archive



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=ROOT / "dist" / "prediction_client",
                        help="distribution directory (default: dist/prediction_client in the workspace)")
    parser.add_argument("--checkpoint", help="optional external model to validate; never copied into the client or archive")
    args = parser.parse_args()
    print(json.dumps(build_client(args.output, args.checkpoint), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
