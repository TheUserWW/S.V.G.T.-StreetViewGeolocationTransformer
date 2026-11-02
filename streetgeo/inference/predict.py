"""Predict one location from one or more local street-view images."""

import argparse
import json
import math

import torch

from streetgeo.core.checkpoints import load_model
from streetgeo.core.config import TrainConfig
from streetgeo.datasets.images import load_image


def predict_images(model, image_paths, *, headings=None, top_k=5, google_coverage_only=False):
    if not 1 <= len(image_paths) <= model.config.max_views:
        raise ValueError(f"provide 1..{model.config.max_views} images of the SAME location")
    device = next(model.parameters()).device
    images = torch.stack([load_image(path, model.config.image_size) for path in image_paths])
    direction_tensor = None
    if headings is not None:
        if len(headings) != len(image_paths):
            raise ValueError("provide one heading per image")
        if not all(value is None or math.isfinite(float(value)) for value in headings):
            raise ValueError("headings must be finite numbers or None")
        direction_tensor = torch.tensor([[float("nan") if value is None else value
                                          for value in headings]], device=device)
    options = {}
    if google_coverage_only:
        from streetgeo.inference.coverage import google_coverage_mask
        options["coordinate_filter"] = google_coverage_mask
    return model.predict(images.unsqueeze(0).to(device), headings=direction_tensor, top_k=top_k, **options)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="1..max_views images of the same location")
    parser.add_argument("--checkpoint", default=TrainConfig().model_path)
    parser.add_argument("--headings", nargs="+", type=float, help="degrees clockwise from north")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--google-coverage-only", action="store_true",
                        help="restrict coordinates to the offline Google Street View country/region allowlist (not road-level coverage)")
    parser.add_argument("--json", action="store_true", help="include candidate scores and Maps URLs")
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    predictions = predict_images(model, args.images, headings=args.headings, top_k=args.top_k,
                                 google_coverage_only=args.google_coverage_only)
    if args.json:
        print(json.dumps([{"coordinates": p.coordinates, "latitude": p.latitude,
                           "longitude": p.longitude, "cell_probability": p.probability,
                           "cell_id": p.cell_id, "maps_url": p.maps_url}
                          for p in predictions], ensure_ascii=False, indent=2))
    else:
        for prediction in predictions:
            print(prediction.coordinates)


if __name__ == "__main__":
    main()
