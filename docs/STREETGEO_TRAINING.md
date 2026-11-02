# S.V.G.T. — Street-View Geolocation Transformer training

The runtime package is `streetgeo`. From the workspace root, use `.venv\Scripts\python.exe -m streetgeo` or either batch launcher. All application code is in `streetgeo/`; the package rename does not move existing data or checkpoints.

## Accepted training data

Training and preparation accept **OpenStreetView-5M only**. The workbench has an OSV-5M import page and a training page; GSV-Cities, standalone Mapillary, custom JSONL import, and mixed-source controls have been removed. The CLI also rejects unsupported raw directories and checks every row in supplied training/validation manifests. Resuming an old OSV-5M run remains supported when its manifests are unchanged. Resuming other datasets is rejected; their weights remain usable for inference or initialization of a new OSV-5M stage with the same architecture. Do not relabel another dataset as OSV-5M.

## Default model and a new training run

The model has width 1024, 16 attention heads, 28 visual Transformer layers, and 2 view-fusion layers. A single view defaults to 224×224. It has 397,937,418 parameters and approximately 1.48 GiB (1,518 MiB) of FP32 weights. Configuration: `configs/streetgeo_400m.json`. The previous 134,686,730-parameter model remains available through `configs/streetgeo_514m.json`; its filename refers to its approximately 514 MiB weight size.

The GUI is configured for a new OSV-5M run: batch size 32, BF16, 4 data workers, 4 epochs, learning rate 0.0003, 1,000 warmup steps, and checkpoint saves every 500 batches or 300 seconds. Localization performance must be established through training and validation. Leave initial weights empty and use the separate output directory `checkpoints/streetgeo_400m_osv/`. The 134,686,730-parameter weights have incompatible dimensions and cannot initialize this larger architecture. Old models still load for prediction using their saved configuration; resuming an old run also retains its saved architecture. Existing prepared manifests in `data/world_osv_514m/` can be reused with the new model because image size and geographic targets are unchanged.

## Choosing the model size

For a new run, the GUI's **模型规模** selector offers both supported architectures:

| Preset | Parameters | Width / heads / visual layers | Suggested single-view batch | Configuration |
| --- | ---: | --- | ---: | --- |
| Original | 134,686,730 | 768 / 12 / 15 | 64 | `configs/streetgeo_514m.json` |
| Larger (default) | 397,937,418 | 1024 / 16 / 28 | 32 | `configs/streetgeo_400m.json` |

Both have two fusion layers. Selecting a preset fills its configuration, BF16 precision (off for CPU), suggested batch size, and a separate available output filename. It clears initial weights so tensors from another architecture cannot be reused accidentally. Picking a custom JSON updates the selector to custom configuration. In resume mode, the selector is disabled and saved checkpoint settings remain authoritative. GPU checks use the selected configuration, batch size, and precision.

To train the original model from the CLI, use:

```powershell
.venv\Scripts\python.exe -m streetgeo train --train-manifest data/world_osv_514m/train.jsonl --val-manifest data/world_osv_514m/val.jsonl --model-config configs/streetgeo_514m.json --output checkpoints/streetgeo_514m_osv/new_run.pt --batch-size 64 --device cuda --amp bf16
```

Model output names must follow Windows filename rules. Use names such as `SVGT_4B.pt`; a colon in `SVGT:4B.pt` is invalid. The GUI, CLI, and checkpoint writer reject invalid output names before creating model files.

## Partially downloaded OSV-5M

Supported local layout:

```text
data/OpenStreetView-5M/
├─ train/
│  ├─ train.csv
│  ├─ 00/<id>.jpg
│  ├─ 01/<id>.jpg
│  └─ …
└─ test/test.csv              Not imported for training
```

Layouts with `train.csv` at the root and images in `images/train/` or the root are also supported. The importer recursively indexes local training images, then streams the CSV. Complete string IDs are matched to filenames without floating-point conversion or assumptions about row/download order.

- Undownloaded images are skipped and counted in `missing_images`.
- Invalid coordinates for an existing local image cause an error to prevent incorrect labels.
- Multiple local files with one ID, or conflicting coordinates/sequences for an ID in the CSV, cause an error. Identical CSV duplicates are deduplicated.
- Local images without CSV labels are excluded and counted in the report.
- The official test directory is excluded. Generated validation/test splits come from the downloaded official training partition.
- Capture sequences, places, and geographic blocks are kept together. Holdouts are separated from training with a 250-meter proximity buffer.

The previously prepared local snapshot contained 600,000 images. Its CSV has 4,894,684 records, of which 4,294,684 were skipped because images were not downloaded. Output `data/world_osv_514m/` contains 498,257 training, 50,251 validation, and 50,239 test images, with 1,253 excluded by proximity filtering. Original images and CSV files are not copied, deleted, or modified.

```powershell
.venv\Scripts\python.exe -m streetgeo prepare --dataset-root data/OpenStreetView-5M --output-dir data/world_osv_514m
```

Identical data and parameters reuse verified manifests. After downloading more shards, use a new output directory such as `data/world_osv_514m_v2`. Do not alter manifests for an active run. Repeated `--dataset-root` arguments may combine OSV-5M roots only. General JSONL import and mixed-source training are no longer supported.

## Starting, saving, and resuming

Completing all training batches does not complete a run immediately. Each epoch also evaluates the validation set, selects the best weights, and saves the full resume state. The workbench shows validation batch/image progress and a saving phase, then marks the task complete when the worker exits. Stopping during final validation preserves the trained weights in the resume checkpoint; resuming finishes validation and saving without repeating completed training batches.

In the GUI, select the OSV-5M root, import it on page 01, and transfer the resulting manifests to page 02 before starting training. Ensure paths match your local files. Alternatively, use:

```powershell
.venv\Scripts\python.exe -m streetgeo train --train-manifest data/world_osv_514m/train.jsonl --val-manifest data/world_osv_514m/val.jsonl --model-config configs/streetgeo_400m.json --output checkpoints/streetgeo_400m_osv/best.pt --epochs 4 --batch-size 32 --num-workers 4 --device cuda --amp bf16
.venv\Scripts\python.exe -m streetgeo train --resume checkpoints/streetgeo_400m_osv/best.last.pt --device cuda
```

Weights occupy approximately 1.48 GiB; a full AdamW resume checkpoint occupies approximately 4.45 GiB. Retaining the preceding state uses another approximately 4.45 GiB, and atomic writes require additional temporary space for both the new state and the backup copy. Allow 24 GiB for one run's weights, current/previous resume states, and temporary files. After stopping and saving, resume the same experiment. Changing architecture or batch size starts a new stage.

Check a real CUDA/BF16 forward pass, backward pass, and AdamW update with the default large model:

```powershell
.venv\Scripts\python.exe -m streetgeo check-gpu --model-config configs/streetgeo_400m.json --batch-size 32
```

This uses random images, saves no weights, and does not evaluate localization. A three-step test of this architecture on the local 16 GB RTX 5080 Laptop GPU passed with single-view batch 32, BF16, and approximately 11.90 GiB of peak PyTorch allocation. This is a short capacity check, not a sustained training benchmark. The GUI GPU check uses the selected batch size and model configuration. Recheck capacity for multiple views or higher resolutions.
