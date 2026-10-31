# StreetGeo: Street-View Geolocation Transformer

StreetGeo converts local street-view images into geographic locations. It outputs decimal coordinates accepted by Google Maps: `latitude, longitude`, for example `34.052200, -118.243700`. This example illustrates the format only.

The default model has **134,686,730 parameters and approximately 513.8 MiB of FP32 weights**: width 768, 12 attention heads, 15 visual layers, and 2 fusion layers. It supports partially downloaded OpenStreetView-5M, GSV-Cities, Mapillary JSONL, and mixed datasets. The larger architecture requires a new training run. Existing approximately 80 MiB models remain usable with their saved architecture; they do not automatically become larger models. See the [training guide](docs/STREETGEO_TRAINING.md). Model size and short capacity checks do not establish localization accuracy.

All runtime code is in `streetgeo/`, launched with `python -m streetgeo <command>`. See [project structure](docs/PROJECT_STRUCTURE.md) for directories and commands. The two root `.bat` launchers open the training workbench and prediction map. Map resources are in `assets/maps/`; model configurations are in `configs/`.

```powershell
.venv\Scripts\python.exe -m streetgeo --help
.venv\Scripts\python.exe -m streetgeo gui
.venv\Scripts\python.exe -m streetgeo map
```

## Distributing the prediction client

`clinet/` is a standalone Windows prediction client containing a trained `best.pt` and offline maps. With 64-bit Python 3.10–3.15 installed, testers can double-click its `.bat` launcher. The first launch downloads CPU dependencies; later launches reuse the local environment. Training data and an NVIDIA GPU are unnecessary. See the [client README](clinet/README.md). Before redistributing a previously used client, remove its generated `.runtime/` directory to exclude dependency caches and personal settings.

The client is generated from project code. Existing weights in the destination are preserved by default. After changing prediction code, regenerate it with:

```powershell
.venv\Scripts\python.exe -m streetgeo build-client --output clinet
```

## Architecture

```mermaid
flowchart TD
    A[1–4 RGB street views of one location] --> B[16×16 patches and spatial position encoding]
    B --> C[Shared 15-layer bidirectional visual Transformer]
    C --> D[4 learned queries per view aggregate visual cues]
    D --> E[Optional camera heading encoding]
    E --> F[2-layer multi-view Transformer with GEO token]
    F --> G[Classification over 648 coarse geographic cells]
    F --> H[Conditional classification over 16 subcells per coarse cell]
    G --> I[Joint probabilities and Top K across 10368 fine cells]
    H --> I
    I --> J[Visual features and candidate embeddings predict within-cell offsets]
    J --> K[Spherical coordinate decoding]
    K --> L[Latitude, longitude, and Google Maps link]
```

- LayerNorm, residual connections, attention, and feed-forward networks are retained from the original Transformer. Text vocabulary, BPE, and causal masking are removed, allowing image regions to attend to each other.
- Each view defaults to `224×224`, with width 768 and 12 attention heads of 64 dimensions each. Views are encoded independently before fusion.
- Inputs may be `[B, 3, H, W]` or `[B, V, 3, H, W]`, with a `view_mask` for variable view counts. All views must depict the same location. View indices are not treated as compass directions without absolute headings. Supplied headings use sine, cosine, and a known-heading indicator.
- Equal-area geographic cells divide `sin(latitude)` and longitude uniformly. They are not countries or administrative regions. The default coarse grid is `18×36`, with `4×4` subcells in each cell, and requires no external boundary files.
- Joint probability is `P(coarse cell | image) × P(subcell | coarse cell, image)`. All fine cells are ranked without early pruning. Offsets are predicted separately for each candidate; distant coordinates are not averaged together.
- `tanh` constrains continuous offsets to their candidate cells. Decoding yields latitude in `[-90,90]` and longitude in `[-180,180)`. Spherical operations handle the date line; geometry uses FP64 to preserve polar precision.
- Training combines coarse-cell cross entropy, subcell cross entropy conditioned on the true coarse cell, within-cell Smooth L1, and unit-sphere chord distance. Offset training uses the true cell so early classification errors do not block learning. Validation uses predicted cells and reports spherical distance errors.
- `embedding` exposes normalized visual features for future place retrieval. A retrieval database, dedicated OCR branch, road matching, and pretrained-model loading are not currently implemented.

Cells provide discrete location hypotheses; within-cell regression produces continuous coordinates. Grid resolution alone does not imply street-level accuracy. Raw 360° equirectangular panoramas are not automatically projected into perspective views. Supply perspective street views in several directions instead. Increasing `image_size` may help high-resolution sign tasks but increases attention computation.

## Environment and verification

Python 3.12 is recommended. In Windows PowerShell:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m streetgeo train --smoke-test --device cpu
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`--smoke-test` checks the forward pass, loss, backward pass, and coordinate output with a small model and random images. It does not save weights or evaluate localization. Install a PyTorch build appropriate for your CUDA environment. `--device auto` uses CUDA when available and otherwise falls back to CPU.

## Local training data

Workbench page 02 supports **OpenStreetView-5M, GSV-Cities, Mapillary, and mixed data**. The OSV-5M importer matches CSV string IDs to local filenames, streams coordinate records, skips undownloaded images, and reports their count. Only the official training partition is imported. Missing GSV-Cities or custom JSONL images remain errors to protect existing experiments. Sources can be grouped jointly by capture sequence and geographic block, with nearby samples excluded across splits. After adding images, use a new output directory rather than changing data for an active run.

```powershell
.venv\Scripts\python.exe -m streetgeo prepare --dataset-root data/OpenStreetView-5M --output-dir data/world_osv_514m
.venv\Scripts\python.exe -m streetgeo train --train-manifest data/world_osv_514m/train.jsonl --val-manifest data/world_osv_514m/val.jsonl --model-config configs/streetgeo_514m.json --output checkpoints/streetgeo_514m_osv/best.pt --epochs 4 --batch-size 64 --num-workers 4 --device cuda --amp bf16
```

Each JSONL line describes one location. Image paths are relative to the manifest directory. `python -m streetgeo download` can generate this format:

```jsonl
{"image":"images/front.jpg","latitude":34.0522,"longitude":-118.2437}
{"images":["images/north.jpg","images/east.jpg"],"latitude":-33.8688,"longitude":151.2093,"headings":[0,90]}
```

`headings` is optional, and individual entries may be `null` for unknown headings. Angles run clockwise from north. Images receive EXIF orientation correction, RGB conversion, aspect-preserving resizing with padding, and normalization. Horizontal flipping is disabled to preserve driving-side and text cues.

```powershell
.venv\Scripts\python.exe -m streetgeo train --train-manifest data/train.jsonl --val-manifest data/val.jsonl --epochs 20 --batch-size 8
```

Split validation data by location or panorama to prevent views of the same place appearing in both training and validation. `python -m streetgeo split` groups capture sequences and geographic blocks and excludes nearby samples across splits. With validation data, the best model is selected by median validation distance; otherwise it is selected by training loss. Checkpoints include their model configuration. Weights from different architectures cannot be mixed. See `streetgeo/core/config.py` for model settings; `--model-config custom.json` overrides configuration fields.

Training periodically writes checkpoints atomically and retains the preceding full state. For example, an output named `checkpoints/streetgeo.pt` uses `checkpoints/streetgeo.last.pt` and `streetgeo.last.previous.pt`. Press Ctrl+C once to finish the current batch, save, and exit. After power loss or forced termination, resume the latest committed state:

```powershell
.venv\Scripts\python.exe -m streetgeo train --resume checkpoints/streetgeo.last.pt
```

Resume state includes the optimizer, AMP scaler, random generators, learning-rate progress, epoch, and next batch. CPU interruption tests are tensor-for-tensor identical to continuous training with the same data, settings, and execution environment. GPU operations may be nondeterministic; cross-device or cross-version bitwise equality is not guaranteed. See the training plans for recovery constraints.

## Inference and Python API

### Local prediction map

Open the root prediction-map `.bat` launcher, select page 04 in the training workbench, or run:

```powershell
.venv\Scripts\python.exe -m streetgeo map
```

1. Import 1–4 perspective views of the same location, select a trained visual checkpoint such as `checkpoints/us_pilot/best.pt`, and start prediction. Both best-weight files and full resume checkpoints support inference. Weights load on the first prediction and are reused until the file changes.
2. Select among the top five candidates. The map displays numbered markers and supports zooming, panning, copying coordinates, and exporting JSON. Grid probabilities are not calibrated correctness estimates.
3. The offline world map uses **UN Geospatial / OCHA country polygons and boundary lines**, including source geometry and labels for China. Source line styles distinguish solid and dashed boundaries; Census US state boundaries are also included. The source does not imply UN endorsement of boundaries or territorial status. Optional online roads use independent OpenStreetMap tiles and their boundary depiction. Tiles follow HTTP cache lifetimes and are not prefetched in bulk. Only tile-location requests leave the machine; street-view images and inference remain local. See [map resources](assets/README.md).
4. Before training, use manual coordinates in `latitude, longitude` format to test markers and screenshot projection. Manual results are labeled as such; random predictions are not substituted for trained results.

**Screenshot calibration and manual placement:** Import a saved game-map screenshot, for example one captured with Windows `Win+Shift+S`. Add corresponding points by selecting a recognizable location on the local map, then the same location in the screenshot. Use at least three pairs; four or more well-spread, non-collinear pairs are recommended. Use coastline corners or intersections rather than displaced text labels. The application fits an affine transformation in Web Mercator space and draws a crosshair for the selected candidate. Export a PNG at the screenshot's original size and place the marker in the game manually. The application does not control the game or submit guesses.

Both maps must be planar and use the same projection. Keep them north-up without tilt and calibrate within the same local area. Corresponding longitudes must span less than half the globe; local views near the date line are supported. Recalibrate after map movement, zooming, or window changes. Replacing the screenshot clears calibration. Off-screen suggestions are reported explicitly. Calibration error is measured in original screenshot pixels. Three points fit exactly and cannot validate accuracy; additional points check consistency but do not establish localization accuracy. Display scaling does not alter exported pixel coordinates. Calibration JSON can be saved and loaded; the original screenshot SHA256 is checked to prevent reuse with a different screenshot.

Inference runs in a background thread so the map remains responsive. Settings are stored in `.gui/prediction_settings.json`. Imported images, screenshots, and predictions are not saved by default. Closing the window during inference waits for that inference to finish.

After training `checkpoints/streetgeo.pt`:

```powershell
.venv\Scripts\python.exe -m streetgeo predict street.jpg
.venv\Scripts\python.exe -m streetgeo predict north.jpg east.jpg --headings 0 90 --top-k 5 --json
```

Default output is one `latitude, longitude` line. `--json` includes candidates, joint grid probabilities, and Maps links. Probabilities are raw model grid probabilities, are not calibrated correctness estimates, and are not renormalized within Top K.

```python
from streetgeo.core.checkpoints import load_model
from streetgeo.inference.predict import predict_images

model = load_model("checkpoints/streetgeo.pt", device="auto")
best = predict_images(model, ["street.jpg"], top_k=1)[0]
print(best.coordinates)
print(best.maps_url)
```

For tensor-based training, use `output = model(images, targets, view_mask=mask, headings=headings)`, where `targets` has shape `[B,2]` and contains `(latitude, longitude)`, then call `output.loss.backward()`. Images must use the normalization implemented by `load_image`.

The original text GPT vocabulary, tokenizer, text data, and text weights are not used for street-view prediction.

## Design references

Patch encoding follows the [Vision Transformer paper](https://arxiv.org/abs/2010.11929). Geographic classification and localization refinement are informed by the [PIGEON paper](https://arxiv.org/abs/2307.05845). This implementation uses its own fixed equal-area grid and continuous offset head; it is not a PIGEON reproduction. Output links follow the [Google Maps URLs documentation](https://developers.google.com/maps/documentation/urls/get-started).
