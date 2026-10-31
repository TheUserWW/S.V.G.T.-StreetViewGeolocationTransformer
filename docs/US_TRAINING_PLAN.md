# Retired US Mapillary training plan

The training program now accepts **OpenStreetView-5M only**. The former Mapillary training workflow is retired, and its GUI collection and import controls have been removed. Existing Mapillary, GSV-Cities, mixed-source, and custom JSONL manifests are rejected for training, validation, and exact resume.

See the [current OSV-5M training guide](STREETGEO_TRAINING.md) for preparation, partial downloads, training, and checkpoint recovery. Source images and existing checkpoints are not deleted by this change. Existing trained weights can still be used for prediction or as `--init-from` weights for a new OSV-5M stage with a matching architecture.

The standalone Mapillary collection utilities remain in the source tree, but their output is not an accepted training input.
