# Project structure

```text
GeoGuesser/
├─ streetgeo/                 Runtime code; importing does not start training or networking
│  ├─ __main__.py              Command dispatcher: python -m streetgeo
│  ├─ core/                    Model, configuration, geographic grid, checkpoints
│  ├─ datasets/                Image loading, OSV-5M import, splitting, legacy collection utilities
│  ├─ training/                Training loop, sampling, recovery, stop requests
│  ├─ inference/               Image prediction, model reuse, tile cache
│  ├─ maps/                    Map data, projections, canvas, offline map refresh
│  ├─ ui/                      Training/prediction windows and background tasks
│  ├─ cli/                     Hardware checks
│  ├─ client/                  Standalone client builder and launcher templates
│  └─ paths.py                 Workspace and map resource paths
├─ assets/maps/                Offline maps and provenance records
├─ configs/                    Model configurations, including streetgeo_514m.json
├─ clinet/                     Distributable client with best.pt and maps
├─ docs/                       Training plans and project documentation
├─ tests/                      Automated tests
├─ data/                       Existing street views and JSONL manifests
├─ checkpoints/                Existing models and resume states
├─ .gui/                       Settings, logs, stop signals, tile cache
└─ *.bat                       Training-workbench and prediction-map launchers
```

Training accepts OpenStreetView-5M only. Prepared manifests retain source and image-ID metadata, which is validated for training, validation, and resume. Legacy collection/source utilities are not accepted training inputs. The workbench exposes only OSV-5M import, training, and prediction.

## Development and execution

All application, command-line, and maintenance code is in `streetgeo/`. Old root Python forwarding files and the `chat` alias have been removed. Run these commands from the workspace root. Both batch launchers use the same package entry point.

| Command | Purpose | Implementation |
|---|---|---|
| `python -m streetgeo gui` | Training workbench | `streetgeo.ui.training` |
| `python -m streetgeo map` | Local prediction map | `streetgeo.ui.prediction` |
| `python -m streetgeo train` | OSV-5M training and checkpoint recovery | `streetgeo.training.train` |
| `python -m streetgeo predict` | Image geolocation | `streetgeo.inference.predict` |
| `python -m streetgeo download` | Legacy Mapillary collection (not accepted for training) | `streetgeo.datasets.download_mapillary` |
| `python -m streetgeo prepare` | OSV-5M import and geographic splitting | `streetgeo.datasets.prepare` |
| `python -m streetgeo split` | JSONL splitting | `streetgeo.datasets.split` |
| `python -m streetgeo check-gpu` | GPU capacity check | `streetgeo.cli.check_gpu` |
| `python -m streetgeo refresh-map` | Refresh offline UN-source maps | `streetgeo.maps.refresh` |
| `python -m streetgeo worker` | GUI hardware/checkpoint helper tasks | `streetgeo.ui.worker` |
| `python -m streetgeo build-client` | Generate standalone prediction client | `streetgeo.client.build` |

Every command supports `--help`. Full module paths such as `python -m streetgeo.training.train` also work. Workbench subprocesses use `-m` to run package modules, and tests import package code directly.

Recommended Python interfaces:

```python
from streetgeo.core.config import ModelConfig
from streetgeo.core.model import StreetGeoModel
from streetgeo.core.checkpoints import load_model
from streetgeo.datasets.images import StreetViewDataset
from streetgeo.inference.predict import predict_images
```

`streetgeo.paths.ROOT` identifies the workspace root. Moving implementation files does not change GUI settings, manifests, relative image paths, outputs, or subprocess working directories. Checkpoint architecture identifiers and state-dictionary keys retain their existing format.

Training plans are in this directory. Existing street views, manifests, checkpoints, and settings retain their paths; obsolete test weights and refactoring backups have been removed. The OSV-5M adapter, `streetgeo/datasets/osv5m.py`, supports `train.csv` at the dataset root or inside `train/`, shard directories, and partial downloads. See the [training guide](STREETGEO_TRAINING.md).

Run tests with:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

See [map resources](../assets/README.md) for provenance, line styles, and refresh instructions, and the [client README](../clinet/README.md) for standalone distribution.
