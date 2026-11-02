"""Small background queries that would otherwise block the desktop UI."""

import argparse
import json
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("hardware", "checkpoint"))
    parser.add_argument("--path")
    args = parser.parse_args()
    if args.operation == "hardware":
        import torch
        cuda = torch.cuda.is_available()
        info = {"event": "hardware", "cuda": cuda, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if cuda else "CPU",
                "vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 if cuda else 0,
                "disk_free_gib": shutil.disk_usage(__file__).free / 1024 ** 3}
    else:
        from streetgeo.core.checkpoints import read_checkpoint
        checkpoint = read_checkpoint(args.path)
        state = checkpoint["training_state"]
        if state.get("resume_version") == 1:
            from streetgeo.datasets.policy import check_osv_manifest_header
            check_osv_manifest_header(state["train_manifest"])
            if state.get("val_manifest"):
                check_osv_manifest_header(state["val_manifest"])
        info = {"event": "checkpoint_info", "resumable": state.get("resume_version") == 1,
                "model_config": checkpoint["model_config"],
                "epoch": state.get("epoch", 0), "step": state.get("global_step", 0),
                "next_batch": state.get("next_batch", 0), "run_config": state.get("run_config", {}),
                "output": state.get("output_path"), "train_manifest": state.get("train_manifest"),
                "val_manifest": state.get("val_manifest")}
    print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
