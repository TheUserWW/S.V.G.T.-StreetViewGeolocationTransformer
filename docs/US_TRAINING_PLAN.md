# US Mapillary street-view training plan

Hardware baseline: RTX 5080 Laptop GPU with approximately 16 GB of measured VRAM, Ultra 9 CPU, and 32 GB RAM. The initial plan starts with **the contiguous 48 states and DC, one view, 224×224 images, and batch size 16**. Use `--region us` for all 50 states, including Alaska and Hawaii but excluding overseas territories. For the current larger model and GUI defaults, see the [StreetGeo training guide](STREETGEO_TRAINING.md).

Download 10,000 images to verify the pipeline, then scale to 100,000 for a larger experiment. These are collection targets, not accuracy guarantees. If API coverage, page limits, or disk budgets prevent reaching the target, the downloader reports the actual total. Measure step time before estimating training hours.

## 1. Environment and Mapillary access token

```powershell
uv --cache-dir .uv-cache pip install --python .venv\Scripts\python.exe -r requirements.txt
```

GPU training requires a CUDA build of PyTorch. The official CUDA 12.8 wheels provide a build suitable for RTX 50-series GPUs. Install it in the project environment:

```powershell
uv --cache-dir .uv-cache pip install --python .venv\Scripts\python.exe "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Use the CUDA check result to confirm availability. `--device cuda` fails explicitly when CUDA is unavailable; `--device auto` falls back to CPU. Specify CUDA for production GPU training to avoid an unintended CPU run.

The local setup installed `torch 2.11.0+cu128` and verified a full-model BF16 forward pass, backward pass, and AdamW update on the 5080. This random-image capacity check establishes execution and memory feasibility, not localization performance or real-data loading speed. Repeat it with:

```powershell
.venv\Scripts\python.exe -m streetgeo check-gpu --batch-size 16
```

Register an application in the [Mapillary developer dashboard](https://www.mapillary.com/dashboard/developers) and obtain your own Client Access Token. `--prompt-token` accepts hidden terminal input without echoing or saving it. Automated jobs can use the `MAPILLARY_ACCESS_TOKEN` environment variable. Do not put tokens in source code, datasets, or chats.

## 2. Stage one: download and inspect 10,000 images

```powershell
.venv\Scripts\python.exe -m streetgeo download --output data/mapillary_us --max-images 10000 --prompt-token
```

Default behavior:

- Uses the official Graph API and prefers 1024-pixel thumbnails. Only non-panoramic `perspective` images are accepted. If the actual short edge is below 224 pixels, larger 1024/2048 versions are attempted in order; small images are not simply upscaled to pass filtering. Each moving capture frame remains a separate single-view sample rather than a false multi-view sample of one location.
- Caches Census state boundaries on first use and requires coordinates to lie within selected states. Location prefers `computed_geometry`, falling back to `geometry`; computed headings are preferred when available.
- Queries shuffled 0.08° tiles, capped at 30 images per tile and 50 per capture sequence to reduce dense-route dominance. This does not guarantee equal state coverage or collection of every available image.
- Uses four image-download threads with a combined default limit of two API/CDN requests per second. HTTP 429 and transient server failures use backoff and respect `Retry-After`. Expired thumbnail URLs are refreshed.
- Repeated HTTP 500 responses subdivide a query region into four prioritized children, up to four levels and a minimum side length of 0.005°. Children share the original region's sampling cap. Confirmed oversized-query errors trigger immediate subdivision instead of repeatedly retrying the same large region.
- Regions that cannot be subdivided further, rate limits, and temporary network failures retain pagination cursors and are deferred for 30–300 seconds while other regions proceed. Six consecutive failures not explicitly identified as oversized-query errors safely pause the run. Restarting reads retained state without losing downloaded images.
- Each invocation queries up to 2,000 pages. Repeat the command to continue toward the target, or raise `--max-pages`. Each successful API response commits image IDs and the pagination cursor before proceeding.
- The default image budget is 20 GiB with at least 5 GiB free disk space. Checks occur between download batches, so one in-flight batch can exceed the budget. Adjust `--max-gb` and `--min-free-gb` as needed.

After interruption, run the same download command again. `--max-images` is the total target including existing downloads. Completed images are skipped, failed images are retried, and missing local files are downloaded again. An interrupted individual image is downloaded afresh; byte-range resume is not implemented.

A preference of `256` refers to the thumbnail version, not a guarantee that its short edge reaches 224 pixels: a 256×144 landscape image fails that requirement. Automatic upgrades can therefore use more bandwidth than 256-pixel downloads alone. Images too small in every available version are reported with their dimensions and recorded as skipped instead of repeatedly retried as network failures. `skipped` counts these undersized images. Old `Image is smaller than min-side` failures retry larger versions when downloading resumes; do not clear the directory or database.

HTTP 500 can reflect image density in a query region or temporary service problems rather than an invalid image. Logs record subdivision, backoff, and safe pauses. `deferred_regions` counts retained regions awaiting retry. The GUI distinguishes reaching the target, exhausting the query budget, and temporary service unavailability; incomplete collection is not reported as full success. Older SQLite queues are migrated with recovery fields while preserving image records and pagination positions.

To focus on California, Nevada, and Arizona:

```powershell
.venv\Scripts\python.exe -m streetgeo download --output data/mapillary_west --states CA,NV,AZ --max-images 10000 --prompt-token
```

`--bbox WEST SOUTH EAST NORTH` adds a bounding-box restriction while retaining state filtering. Region, resolution, random seed, and sampling settings are fixed for each output directory; use a new directory to change them. Targets, page limits, thread counts, and disk budgets may change. Run only one downloader per data directory at a time.

| File | Contents |
|---|---|
| `images/*.jpg` | Downloaded and validated original thumbnails |
| `downloads.sqlite3` | Image queue, status, page positions, and failure reasons |
| `all.jsonl` | Directly usable image paths, coordinates, headings, and source metadata |
| `attribution.jsonl` | Author information, image pages, and license references returned by the API |

Do not back up a running downloader by copying a single SQLite file. Exit cleanly and copy the entire data directory. Inspect example images, locations, and attribution records before training. Mapillary coordinates and headings may contain errors; retained metadata supports later cleaning.

## 3. Freeze training, validation, and test splits

```powershell
.venv\Scripts\python.exe -m streetgeo split --manifest data/mapillary_us/all.jsonl
```

The default target is approximately 80%/10%/10% by image count. Whole-group assignments and proximity filtering can change the ratios:

1. Each capture sequence stays in one group.
2. Images in the same 0.25° block stay together. Sequences crossing blocks merge the affected groups to prevent sequence leakage.
3. Validation images within 250 meters of training images are removed. Test images within 250 meters of training or validation images are then removed.

This reduces adjacent-frame leakage but cannot guarantee removal of all distant visual duplicates. Too few independent geographic groups or an empty filtered split causes an error; collect more independent locations.

Outputs are `train.jsonl`, `val.jsonl`, `test.jsonl`, and `split_report.json`. Freeze these files once training starts. New downloads do not automatically alter manifests. Do not expand a manifest during checkpoint recovery.

## 4. Pilot run and automatic checkpoints

Train on the 10,000-image dataset for three epochs to check loss, validation distance in kilometers, and resource use:

```powershell
.venv\Scripts\python.exe -m streetgeo train --train-manifest data/mapillary_us/train.jsonl --val-manifest data/mapillary_us/val.jsonl --epochs 3 --batch-size 16 --num-workers 4 --device cuda --amp bf16 --output checkpoints/us_pilot/best.pt --save-every-steps 100 --save-every-seconds 300
```

The original pilot used AdamW, learning rate `3e-4`, weight decay `0.1`, 100 warmup steps, gradient clipping at 1.0, and cosine decay. BF16 requires no loss scaler by default; FP16 checkpoints include scaler state. Start with one view. If VRAM is insufficient, create a new stage with a smaller batch size; exact recovery of an existing run cannot change batch size.

| File | Purpose |
|---|---|
| `best.pt` | Weights with the best median validation distance, for inference or new-stage initialization |
| `best.last.pt` | Latest full training state for recovery |
| `best.last.previous.pt` | Previous full state if the latest file is damaged |

An initial state is saved before training. Saves occur at batch boundaries every 100 batches or 300 seconds, whichever comes first, and at each epoch end. Full state includes model weights, AdamW moments, AMP state, Python/PyTorch/CUDA random generators, learning-rate progress, completed batches, and statistics for an unfinished epoch.

Saving writes a temporary file in the same directory, flushes and fsyncs it, then atomically replaces the destination. This protects the committed checkpoint from an interrupted write. The original small-model full checkpoint was approximately 250 MB, with at least 2 GiB recommended for two full states, best weights, and temporary writes. For the current approximately 514 MiB model, each full state is approximately 1.5 GiB; budget additional space accordingly. See the [current training guide](STREETGEO_TRAINING.md).

## 5. Stopping, power loss, and recovery

Press **Ctrl+C once** to finish the current batch, save, and exit cleanly. A second Ctrl+C forces interruption, leaving the previously committed checkpoint as the recovery point. Process termination, power loss, or a system crash cannot trigger a final save; periodic saves limit lost progress.

```powershell
.venv\Scripts\python.exe -m streetgeo train --resume checkpoints/us_pilot/best.last.pt --num-workers 4 --device cuda
```

If the latest state is damaged:

```powershell
.venv\Scripts\python.exe -m streetgeo train --resume checkpoints/us_pilot/best.last.previous.pt --num-workers 4 --device cuda
```

Recovery restores data paths, model configuration, total epochs, and hyperparameters, then starts at the next batch without loading images for skipped batches. Keep images and training/validation manifests. Changed manifest fingerprints, batch size, or total training length cause an error rather than silently altering the experiment. Fingerprints do not hash every image's content; do not replace training images either.

CPU tests with nonzero dropout verified tensor-for-tensor agreement between interrupted/resumed and continuous training for weights, optimizer state, and learning-rate progress. CUDA parallel operations, software versions, or hardware changes may produce numerical differences; GPU bitwise equality is not guaranteed.

`--stop-after-steps 10` saves and stops after ten additional batches to test recovery. `best.pt` is an inference file without full resume state. Use `--init-from` to begin a new stage from it, resetting the optimizer and training progress.

If output already exists, starting without `--resume` fails rather than overwriting old progress.

## 6. Scale up the experiment

After validating the pilot, raise the target in the same download directory to 100,000 and increase budgets:

```powershell
.venv\Scripts\python.exe -m streetgeo download --output data/mapillary_us --max-images 100000 --max-pages 20000 --max-gb 80 --prompt-token
.venv\Scripts\python.exe -m streetgeo split --manifest data/mapillary_us/all.jsonl --output-dir data/mapillary_us/splits_100k
.venv\Scripts\python.exe -m streetgeo train --train-manifest data/mapillary_us/splits_100k/train.jsonl --val-manifest data/mapillary_us/splits_100k/val.jsonl --epochs 20 --batch-size 16 --num-workers 4 --device cuda --amp bf16 --output checkpoints/us_100k/best.pt
```

Use a separate new run. This example trains from scratch so pilot weights cannot contaminate newly assigned validation/test places. Twenty epochs are an initial budget, not a convergence guarantee. Plan subsequent experiments using validation distance. Keep the test set sealed and select models using validation only. Preserve geographic separation when scaling further.

The coordinate head remains global while US labels teach the US distribution; outputs are not hard-coded to stay within the country. Evaluate Mapillary in-domain performance separately from Google Street View cross-source performance because cameras, color, blurring, and capture coverage differ.

## Official references

- [Mapillary authentication and tokens](https://mapillary.github.io/mapillary-python-sdk/docs/mapillary.models/mapillary.models.client/)
- [Mapillary image metadata fields](https://mapillary.github.io/mapillary-python-sdk/docs/mapillary.config.api/mapillary.config.api.entities/)
- [Mapillary licensing and attribution](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data)
- [Census US state boundary layer](https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/0)
- [PyTorch training-state checkpoints](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Official PyTorch CUDA 12.8 wheel index](https://download.pytorch.org/whl/cu128/torch/)
