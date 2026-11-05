# S.V.G.T. project structure

```text
S.V.G.T.-StreetViewGeolocationTransformer/
├─ streetgeo/                 Runtime code; importing does not start training or networking
│  ├─ __main__.py              Command dispatcher: python -m streetgeo
│  ├─ core/                    Model, configuration, geographic grid, checkpoints
│  ├─ datasets/                Image loading, Mapillary API collection, global region selection and splitting
│  ├─ training/                Training loop, environment fine-tuning, alternating sampling, recovery, stop requests
│  ├─ continual/               Shared continual-learning engine, budgets, batches and checkpoint promotion
│  ├─ inference/               Prediction, shared explanation, model reuse, developer evaluation/charts, tile cache
│  ├─ maps/                    Map data, projections, canvas, offline map refresh
│  ├─ ui/                      Training/prediction windows, test reports, attention/occlusion window, background tasks
│  ├─ cli/                     Hardware checks
│  ├─ client/                  Standalone client builder and launcher templates
│  └─ paths.py                 Workspace and map resource paths
├─ assets/maps/                Offline maps and provenance records
├─ configs/                    Model configurations: streetgeo_400m.json (default), streetgeo_514m.json (legacy)
├─ docs/
│  ├─ training/                Mapillary guide and regional training plans
│  ├─ prediction/              Prediction coverage and feedback learning
│  └─ structure.md             Directory and command reference
├─ scripts/windows/           Training, daily prediction, and developer prediction launchers
├─ tests/                      Automated tests
├─ dist/prediction_client/     Generated bilingual client and ZIP (local)
├─ data/                       Street views and JSONL manifests (local)
├─ checkpoints/                Models and resume states (local)
└─ .gui/                       Settings, logs, tile cache, and evaluation reports (local)
```

The pretraining workbench accepts Mapillary imagery only. Prepared manifests retain source and image-ID metadata, which is validated for training, validation, and resume. The workbench exposes Mapillary API download, geographic import/splitting, training and prediction.

## Development and execution

Run these commands from the workspace root. The root batch files forward to the corresponding launchers in `scripts/windows/` for compatibility with existing shortcuts.

| Command | Purpose | Implementation |
|---|---|---|
| `python -m streetgeo gui` | Training workbench | `streetgeo.ui.training` |
| `python -m streetgeo workspace` | Daily prediction and corrections | `streetgeo.client.workbench` |
| `python -m streetgeo map` | Developer prediction map and evaluation | `streetgeo.ui.prediction` |
| `python -m streetgeo train` | Mapillary training and checkpoint recovery | `streetgeo.training.train` |
| `python -m streetgeo predict` | Image geolocation | `streetgeo.inference.predict` |
| `python -m streetgeo download` | Mapillary collection by continent, country or region | `streetgeo.datasets.download_mapillary` |
| `python -m streetgeo prepare` | Mapillary import and geographic splitting | `streetgeo.datasets.prepare` |
| `python -m streetgeo split` | Mapillary manifest splitting through the shared frozen preparation pipeline | `streetgeo.datasets.split` |
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

The Mapillary adapter imports `all.jsonl` and `images/` from the API downloader. `streetgeo/datasets/regions.py` selects continents, ISO alpha-3 countries, built-in US states and custom GeoJSON/bbox regions from bundled geography. See the [training guide](training/mapillary.md).

Run tests with:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

See [map resources](../assets/README.md) for provenance, line styles, and refresh instructions, and the [client README](../streetgeo/client/README.md) for standalone distribution.
