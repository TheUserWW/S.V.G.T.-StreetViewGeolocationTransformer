import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import torch

from streetgeo.core.checkpoints import save_checkpoint
from streetgeo.core import EmptyPredictionDomainError
from streetgeo.core.model import StreetGeoModel
from streetgeo.inference.backend import InferenceEngine
from streetgeo.inference.coverage import coverage_details, google_coverage_mask
from streetgeo.inference.explain import analyze_attention, occlude_region
from streetgeo.inference.evaluate import evaluate_test_set
from test_evaluation import fixture
from test_model import tiny_config


class CoverageTests(unittest.TestCase):
    def test_real_region_boundaries_exclude_noncovered_countries_and_ocean(self):
        coordinates = [[35.6762, 139.6503], [51.5072, -.1276], [-33.8688, 151.2093],
                       [22.42, 114.12], [22.15, 113.56],  # Hong Kong and Macao, separate from mainland.
                       [39.9042, 116.4074], [35.69, 51.39], [0, 0], [91, 0], [float("nan"), 0]]
        self.assertEqual(google_coverage_mask(coordinates).tolist(), [True] * 5 + [False] * 5)
        self.assertFalse(coverage_details()["road_level_verified"])

    def test_polygon_holes_boundary_and_coordinate_order(self):
        polygon = Polygon([(100, 10), (110, 10), (110, 20), (100, 20)],
                          holes=[[(103, 13), (107, 13), (107, 17), (103, 17)]])
        with patch("streetgeo.inference.coverage._coverage_geometry", return_value=polygon):
            self.assertEqual(google_coverage_mask([[12, 102], [15, 105], [10, 100], [12, -102]]).tolist(),
                             [True, False, True, False])

    def test_ranking_scans_past_initial_top_five_and_chunk_boundary(self):
        model = StreetGeoModel(tiny_config(coarse_lat_bins=18, coarse_lon_bins=36))
        count = model.config.total_cells
        embedding = torch.zeros(1, model.config.n_embed)
        log_probs = torch.log_softmax(-torch.arange(count).float()[None] / 100, dim=-1)
        def zero_offsets(embedding, ids):
            return torch.zeros(*ids.shape, 2, device=ids.device)
        with patch.object(model, "_offsets", side_effect=zero_offsets):
            ids = torch.arange(count)[None]
            coordinates = model.grid.decode(ids, zero_offsets(embedding, ids))
            allowed = coordinates[0, :, 0] > 30
            expected = ids[0, allowed][:5]
            self.assertGreater(int(expected[0]), 256)
            values, actual, coords = model._rank_candidates(embedding, log_probs, 5,
                                                            lambda coords: coords[:, 0] > 30)
            torch.testing.assert_close(actual[0], expected)
            torch.testing.assert_close(values[0], log_probs[0, expected])
            torch.testing.assert_close(coords[0], coordinates[0, expected])
            # A small domain returns fewer results, without padding invalid points.
            one_point = coordinates[0, -1].numpy()
            _, actual, _ = model._rank_candidates(embedding, log_probs, 5,
                                                   lambda coords: np.all(coords == one_point, axis=-1))
            self.assertEqual(actual.tolist(), [[count - 1]])

    def test_empty_domain_fails_without_global_fallback_and_restores_training_state(self):
        model = StreetGeoModel(tiny_config()).train()
        images = torch.zeros(1, 1, 3, 32, 32)
        with self.assertRaises(EmptyPredictionDomainError):
            model.predict(images, coordinate_filter=lambda coords: np.zeros(len(coords), dtype=bool))
        self.assertTrue(model.training)
        self.assertEqual(len(model.predict(images)[0]), 5)

    def test_backend_switches_domain_without_reloading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            image, checkpoint = folder / "street.png", folder / "tiny.pt"
            Image.new("RGB", (64, 40), "orange").save(image)
            torch.manual_seed(12)
            save_checkpoint(StreetGeoModel(tiny_config(coarse_lat_bins=18, coarse_lon_bins=36)), checkpoint)
            engine = InferenceEngine()
            world = engine.predict(checkpoint, [image], device="cpu")
            model = engine.model
            restricted = engine.predict(checkpoint, [image], device="cpu", google_coverage_only=True)
            self.assertIs(engine.model, model)
            self.assertTrue(restricted["reused_model"])
            self.assertEqual(restricted["prediction_scope"], "google_streetview_regions")
            self.assertFalse(restricted["coverage"]["road_level_verified"])
            self.assertTrue(google_coverage_mask([[p["latitude"], p["longitude"]]
                                                for p in restricted["candidates"]]).all())
            json.dumps(restricted, allow_nan=False)
            self.assertEqual(engine.predict(checkpoint, [image], device="cpu")["candidates"], world["candidates"])
            attention = analyze_attention(model, [image], google_coverage_only=True)
            occlusion = occlude_region(model, attention, 0, (.1, .1, .8, .8))
            for result in (attention, occlusion):
                self.assertTrue(google_coverage_mask([[p["latitude"], p["longitude"]]
                                                    for p in result["candidates"]]).all())

    def test_evaluation_filters_predictions_and_keeps_all_test_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, directory = fixture(folder)
            torch.manual_seed(12)
            save_checkpoint(StreetGeoModel(tiny_config(coarse_lat_bins=18, coarse_lon_bins=36)), checkpoint)
            with patch("streetgeo.inference.evaluate.save_evaluation_charts", return_value={}):
                report = evaluate_test_set(checkpoint, directory, device="cpu", batch_size=2,
                                           output_root=folder, google_coverage_only=True)
            self.assertEqual(report["status"], "completed", report.get("error"))
            self.assertEqual(report["successful"], 5)
            self.assertEqual(report["prediction_scope"], "google_streetview_regions")
            self.assertFalse(report["prediction_domain"]["road_level_verified"])
            import csv
            with (Path(report["output_directory"]) / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    self.assertTrue(google_coverage_mask([[float(row["predicted_latitude"]),
                                                          float(row["predicted_longitude"])]]).all())


if __name__ == "__main__":
    unittest.main()
