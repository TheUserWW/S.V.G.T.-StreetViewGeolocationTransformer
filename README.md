# Street-View Geolocation Transformer (S.V.G.T.)

S.V.G.T. learns to estimate latitude and longitude from perspective street-view images. This repository provides the PyTorch model, checkpoint format, Mapillary data preparation, and command-line training and resume tools. Model weights and datasets are distributed separately.

## Setup

Use Python 3.10 or later in a virtual environment. Install the PyTorch build appropriate for your system.

```sh
python -m pip install -r requirements.txt
python -m streetgeo train --smoke-test --device cpu
python -m streetgeo --help
```

## Train with Mapillary

Get a [Mapillary Client Access Token](https://www.mapillary.com/dashboard/developers). `--prompt-token` reads it without echoing it. You can also set `MAPILLARY_ACCESS_TOKEN` locally.

```sh
python -m streetgeo download --list-regions
python -m streetgeo download --prompt-token --countries JPN --output data/mapillary_japan --max-images 10000
python -m streetgeo prepare --dataset-root data/mapillary_japan --output-dir data/japan_split
python -m streetgeo train --train-manifest data/japan_split/train.jsonl --val-manifest data/japan_split/val.jsonl --model-config configs/streetgeo_400m.json --output checkpoints/japan.pt --device cuda --amp bf16
python -m streetgeo train --resume checkpoints/japan.last.pt --device cuda
```

The data tools retain Mapillary image and sequence identifiers and separate geographic groups across splits. See the [training guide](docs/training/mapillary.md) for geographic filtering, environment fine-tuning, checkpoint behavior, and CPU or CUDA settings.

## Source layout

- `streetgeo/core/` contains the model, geographic grid, configuration, and checkpoints.
- `streetgeo/datasets/` contains data collection, validation, geographic splitting, and preprocessing.
- `streetgeo/training/` contains the training loop, validation, and resume logic.
- `configs/` contains the default 400M configuration and the legacy 135M configuration (`streetgeo_514m.json`).
- `assets/` contains [geographic reference data](assets/README.md) for dataset collection.

Run the headless tests with `python -m unittest discover -s tests -v`.
