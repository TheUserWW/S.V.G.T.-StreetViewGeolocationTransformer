# Global training with OpenStreetView-5M

The training program now accepts **OpenStreetView-5M only**, including partially downloaded training images. The former GSV-Cities, Mapillary, and mixed-source import workflows are retired.

Use the [current OSV-5M training guide](STREETGEO_TRAINING.md) for the supported directory layouts, preparation commands, model configuration, checkpoint storage, and resume instructions. The model continues to predict worldwide coordinates. Actual learned coverage depends on the images present locally and used for training.

Preparation imports the official OSV-5M training partition and creates geographically separated training, validation, and test manifests. The official test directory is excluded from training. Missing undownloaded images are counted and skipped. After adding shards, use a new preparation directory to keep existing runs reproducible.

Every training and validation record is checked for its OSV-5M source and image ID, including resumed runs. Other datasets and mixed manifests are rejected. Old models remain usable for prediction or initialization of a new OSV-5M stage with a matching architecture. A new stage starts a fresh optimizer and schedule; exact resume continues the saved OSV-5M experiment.

Official resources: [OSV-5M project](https://osv5m.github.io/) and [dataset repository](https://huggingface.co/datasets/osv5m/osv5m).
