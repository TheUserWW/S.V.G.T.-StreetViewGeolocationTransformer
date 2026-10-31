# StreetGeo large model and partial OSV-5M training

The runtime package is `streetgeo`. From the workspace root, use `.venv\Scripts\python.exe -m streetgeo` or either batch launcher. All application code is in `streetgeo/`; the package rename does not move existing data or checkpoints.

## Default model and a new training run

The model has width 768, 12 attention heads, 15 visual Transformer layers, and 2 view-fusion layers. A single view defaults to 224×224. It has 134,686,730 parameters and approximately 513.8 MiB of FP32 weights. Configuration: `configs/streetgeo_514m.json`.

The GUI is configured for a new OSV-5M run: batch size 64, BF16, 4 data workers, 4 epochs, learning rate 0.0003, 1,000 warmup steps, and checkpoint saves every 500 batches or 300 seconds. Localization performance must be established through training and validation. The new model does not inherit the abilities of the old approximately 80 MiB model automatically. Leave initial weights empty and use the separate output directory `checkpoints/streetgeo_514m_osv/`. Do not initialize this larger architecture with old small-model weights. Old models still load for prediction using their saved configuration; resuming an old run also retains its saved architecture.

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

The prepared local snapshot contains 600,000 images. Its CSV has 4,894,684 records, of which 4,294,684 were skipped because images were not downloaded. Output `data/world_osv_514m/` contains 498,257 training, 50,251 validation, and 50,239 test images, with 1,253 excluded by proximity filtering. Original images and CSV files are not copied, deleted, or modified.

```powershell
.venv\Scripts\python.exe -m streetgeo prepare --dataset-root data/OpenStreetView-5M --output-dir data/world_osv_514m
```

Identical data and parameters reuse verified manifests. After downloading more shards, use a new output directory such as `data/world_osv_514m_v2`. Do not alter manifests for an active run. Repeat `--dataset-root` or `--manifest` to combine sources. Leave unused source paths empty in GUI mixed mode.

## Starting, saving, and resuming

The GUI already contains the training manifest, validation manifest, and model configuration. Start training there, or use:

```powershell
.venv\Scripts\python.exe -m streetgeo train --train-manifest data/world_osv_514m/train.jsonl --val-manifest data/world_osv_514m/val.jsonl --model-config configs/streetgeo_514m.json --output checkpoints/streetgeo_514m_osv/best.pt --epochs 4 --batch-size 64 --num-workers 4 --device cuda --amp bf16
.venv\Scripts\python.exe -m streetgeo train --resume checkpoints/streetgeo_514m_osv/best.last.pt --device cuda
```

Weights occupy approximately 514 MiB; a full resume checkpoint occupies approximately 1.5 GiB. Retaining the preceding state uses another approximately 1.5 GiB, and temporary writes require additional space. After stopping and saving, resume the same experiment. Changing architecture or batch size starts a new stage.

Check a real CUDA/BF16 forward pass, backward pass, and AdamW update with the default large model:

```powershell
.venv\Scripts\python.exe -m streetgeo check-gpu --batch-size 64
```

This uses random images, saves no weights, and does not evaluate localization. Recheck capacity for multiple views or higher resolutions.
