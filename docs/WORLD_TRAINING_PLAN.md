# Global street-view training: GSV-Cities and Mapillary

The current default is the approximately 514 MiB model, with partial OSV-5M support and the `python -m streetgeo` entry point. See the [current training guide](STREETGEO_TRAINING.md) for the large model, prepared OSV-5M data, GUI settings, saving, and recovery. The GSV-Cities and Mapillary workflows below remain available. Their batch-size and epoch examples describe the original pilot configuration rather than the latest GUI defaults.

The geographic grid and coordinate output already cover the world. Global data import does not require changing the architecture or invalidating existing checkpoints. Preparation is entirely offline and generates manifests while retaining original images. The local GSV-Cities archive contains 529,506 images of 64,394 places across 23 cities. All of these cities are imported without US state filtering. A dataset of cities around the world does not provide comprehensive coverage of all regions, rural areas, or roads.

## Using the GUI

Reopen the training-workbench batch launcher and select page 02, the import and split page.

- **GSV-Cities (global):** select `data/archive`, containing `Images` and `Dataframes`, and use `data/world_gsv` as the output directory.
- **Mapillary / JSONL:** select an existing manifest such as `data/mapillary_us/all.jsonl`; use an output directory such as `data/prepared_mapillary`. The Mapillary download page currently collects US data, while the JSONL interface accepts valid coordinates from other regions.
- **Mixed data (global):** select both the GSV-Cities root and Mapillary manifest; use `data/world_mixed`. Sources are combined before joint grouping and proximity filtering, rather than split separately and concatenated. Sampling follows actual sample counts by default; a small Mapillary subset is not oversampled to half the dataset.

Run import and split generation, then transfer the result to the training configuration. GSV/mixed mode selects a new output such as `checkpoints/world_gsv/best.pt` or `checkpoints/world_mixed/best.pt`. If training output already exists, a numbered directory avoids overwriting progress. Existing US models remain in place and can resume their original experiments from their `.last.pt` files.

The original pilot used one 224×224 view, batch size 16, BF16, and four data workers on the RTX 5080 Laptop. Three epochs provide an initial trend; use independent validation to plan subsequent experiments. No epoch count guarantees a particular game score. Validation reports P90/P95/P99 distance errors in kilometers and the fraction within 100 km to reveal long-distance mistakes.

Preparation can be canceled safely. Unpublished output is not treated as a complete dataset. Restarting begins import again; this differs from batch-level training checkpoint recovery. Successfully prepared, unchanged data can be reused without regenerating manifests.

## Command line: prepare, then train

GSV-Cities only:

```powershell
.venv\Scripts\python.exe -m streetgeo prepare --dataset-root data/archive --output-dir data/world_gsv
.venv\Scripts\python.exe -m streetgeo train --train-manifest data/world_gsv/train.jsonl --val-manifest data/world_gsv/val.jsonl --output checkpoints/world_gsv/best.pt --epochs 3 --batch-size 16 --num-workers 4 --device cuda --amp bf16
```

Joint training with both sources:

```powershell
.venv\Scripts\python.exe -m streetgeo prepare --dataset-root data/archive --manifest data/mapillary_us/all.jsonl --output-dir data/world_mixed
.venv\Scripts\python.exe -m streetgeo train --train-manifest data/world_mixed/train.jsonl --val-manifest data/world_mixed/val.jsonl --output checkpoints/world_mixed/best.pt --epochs 3 --batch-size 16 --num-workers 4 --device cuda --amp bf16
```

Mapillary only:

```powershell
.venv\Scripts\python.exe -m streetgeo prepare --dataset-root data/mapillary_us --output-dir data/prepared_mapillary
```

Repeat `--manifest` to merge several JSONL files, or `--dataset-root` to merge extracted dataset directories. GSV-Cities directories are detected by `Images` and `Dataframes`; Mapillary directories are detected by `all.jsonl`.

## Command line: use raw dataset directories directly

Training can prepare the data automatically before starting:

```powershell
.venv\Scripts\python.exe -m streetgeo train --dataset-root data/archive --dataset-root data/mapillary_us --data-output data/world_mixed --output checkpoints/world_mixed/best.pt --epochs 3 --batch-size 16 --num-workers 4 --device cuda --amp bf16
```

`--gsv-cities-root` is an alias for a GSV directory argument. Choose either raw dataset inputs or explicit `--train-manifest/--val-manifest` inputs. To resume, omit raw directories and use:

```powershell
.venv\Scripts\python.exe -m streetgeo train --resume checkpoints/world_mixed/best.last.pt --num-workers 4 --device cuda
```

Existing JSONL training, `--init-from` initialization for a new stage, periodic atomic checkpoints, stop files, SIGINT handling, and recovery after power loss remain supported. Switching datasets is a new experiment; `--resume` cannot replace the training manifest.

## Import rules and evaluation separation

- CSV fields `place_id/year/month/northdeg/city_id/lat/lon/panoid` are matched to actual files. Floating-point formatting, `.JPG/.jpg`, and underscores within panorama IDs do not cause incorrect path construction. CSV coordinates are checked against filename coordinates; missing images identify the specific record.
- `northdeg` is normalized modulo 360°, retaining the original value for traceability. GSV images from different dates are independent single-view samples, not an artificial four-direction panorama captured at one time. Original images are neither re-encoded nor moved.
- GSV group identity combines city and place IDs. Historical images of one place, images from one panorama, duplicate files, and each Mapillary capture sequence remain in one split. A fixed representative coordinate assigns each GSV place to a block, preventing small historical GPS shifts from merging an entire city through block crossings. Global preparation defaults to 0.02° blocks with city/source stratification; Mapillary GUI mode retains 0.25°. Sources are grouped jointly with a default 250-meter exclusion buffer between splits.
- Outputs are `all.jsonl`, `train.jsonl`, `val.jsonl`, `test.jsonl`, and `split_report.json`. Reports include source/city counts, groups, proximity exclusions, input fingerprints, and manifest checksums. Image paths are normally relative to the output directory; cross-drive Windows paths remain absolute.
- Repeating identical inputs and parameters verifies and reuses frozen manifests. Use a new output directory after adding images or changing parameters/manifests. Fingerprints check metadata and paths, not the contents of every original image. Do not replace images during training.
- Reserve the test set for final evaluation. Do not randomly split historical photographs of the same place across evaluation partitions, or interpret training accuracy as global localization ability.

## Official dataset documentation

- [GSV-Cities repository](https://github.com/amaralibey/gsv-cities)
- [Official directory layout and CSV fields](https://github.com/amaralibey/gsv-cities#database-organisation)
- [US training plan](US_TRAINING_PLAN.md) for Mapillary downloads and checkpoint recovery
