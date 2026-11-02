import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from PIL import Image

from streetgeo.core.checkpoints import load_model, save_checkpoint
from streetgeo.core.config import ModelConfig
from streetgeo.datasets.images import StreetViewDataset, collate_streetviews, load_image
from streetgeo.core.geography import HierarchicalGeoGrid, LocationPrediction, haversine_km
from streetgeo.core.model import StreetGeoModel, MultiHeadAttention
from streetgeo.inference.predict import predict_images

torch.set_num_threads(2)


def tiny_config(**overrides):
    fields = dict(image_size=32, patch_size=8, n_embed=32, n_head=4, n_layer=2,
                  fusion_layers=1, view_queries=2, cell_embed_dim=16, dropout=0.0,
                  coarse_lat_bins=3, coarse_lon_bins=6, fine_bins=2)
    fields.update(overrides)
    return ModelConfig(**fields)


class GeographyTests(unittest.TestCase):
    def test_round_trip_world_boundaries_and_near_poles(self):
        grid = HierarchicalGeoGrid(ModelConfig())
        coordinates = torch.tensor([
            [0, 0], [90, 180], [-90, -180], [89.9999, 179.9999],
            [-89.9999, -179.9999], [34.0522, -118.2437], [-33.8688, 151.2093],
        ], dtype=torch.float64)
        parent, child, offsets = grid.encode(coordinates)
        ids = parent * grid.children + child
        reconstructed = grid.decode(ids, offsets)
        self.assertLess(haversine_km(coordinates, reconstructed).max().item(), 0.00001)
        self.assertTrue(((ids >= 0) & (ids < grid.total_cells)).all())
        self.assertTrue((offsets.abs() <= 1).all())

    def test_antimeridian_distance(self):
        error = haversine_km(torch.tensor([[0., 179.9]]), torch.tensor([[0., -179.9]]))
        self.assertAlmostEqual(error.item(), 22.239, delta=0.01)

    def test_invalid_coordinates_rejected(self):
        grid = HierarchicalGeoGrid(tiny_config())
        for values in ([[91., 0.]], [[0., 181.]], [[float("nan"), 0.]]):
            with self.assertRaises(ValueError):
                grid.encode(torch.tensor(values))

    def test_polar_loss_gradients_finite(self):
        grid = HierarchicalGeoGrid(tiny_config())
        ids = torch.tensor([0, grid.total_cells - 1])
        offsets = torch.tensor([[-1., -1.], [1., 1.]], requires_grad=True)
        grid.unit_vectors(ids, offsets).sum().backward()
        self.assertTrue(torch.isfinite(offsets.grad).all())

    def test_google_maps_format(self):
        prediction = LocationPrediction(34.0522, -118.2437, 0.2, 1)
        self.assertEqual(prediction.coordinates, "34.052200, -118.243700")
        self.assertEqual(prediction.maps_url,
                         "https://www.google.com/maps/search/?api=1&query=34.052200%2C-118.243700")


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = StreetGeoModel(tiny_config()).eval()
        self.images = torch.randn(2, 3, 3, 32, 32)

    def test_joint_distribution_and_topk(self):
        output = self.model(self.images, top_k=self.model.config.total_cells)
        torch.testing.assert_close(output.candidate_probabilities.sum(-1), torch.ones(2))
        self.assertTrue((output.candidate_probabilities[:, :-1] >=
                         output.candidate_probabilities[:, 1:]).all())
        self.assertTrue((output.candidate_coordinates[..., 0].abs() <= 90).all())
        self.assertTrue((output.candidate_coordinates[..., 1].abs() <= 180).all())
        torch.testing.assert_close(output.coordinates, output.candidate_coordinates[:, 0])

    def test_backprop_reaches_visual_and_location_heads(self):
        self.model.train()
        targets = torch.tensor([[34., -118.], [-33., 151.]])
        output = self.model(self.images, targets)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for parameter in (self.model.encoder.patch_embedding.weight,
                          self.model.encoder.visual_blocks[0].attention.qkv.weight,
                          self.model.encoder.fusion_blocks[0].attention.qkv.weight,
                          self.model.coarse_head.weight, self.model.fine_head.weight,
                          self.model.cell_embedding.weight, self.model.offset_head[0].weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_padding_is_ignored_even_if_nan(self):
        mask = torch.tensor([[True, False, True], [True, True, False]])
        expected = self.model(self.images, view_mask=mask).coarse_logits
        padded = self.images.clone()
        padded[~mask] = float("nan")
        actual = self.model(padded, view_mask=mask).coarse_logits
        torch.testing.assert_close(actual, expected)
        for i in range(2):
            unpadded = self.model(self.images[i:i+1, mask[i]]).coarse_logits
            torch.testing.assert_close(expected[i:i+1], unpadded, atol=1e-6, rtol=1e-5)

    def test_unordered_views_with_headings(self):
        headings = torch.tensor([[0., 90., float("nan")], [180., 270., 0.]])
        permutation = [2, 0, 1]
        original = self.model(self.images, headings=headings).coarse_logits
        permuted = self.model(self.images[:, permutation], headings=headings[:, permutation]).coarse_logits
        torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-5)

    def test_attention_is_bidirectional(self):
        attention = MultiHeadAttention(tiny_config()).eval()
        tokens = torch.randn(1, 4, 32)
        modified = tokens.clone()
        modified[:, -1] += 10
        self.assertFalse(torch.allclose(attention(tokens)[:, 0], attention(modified)[:, 0]))

    def test_single_view_and_predict_restores_mode(self):
        self.model.train()
        result = self.model.predict(self.images[:, 0], top_k=2)
        self.assertTrue(self.model.training)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        self.assertIsInstance(result[0][0].coordinates, str)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.model(self.images, view_mask=torch.zeros(2, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            self.model(self.images, top_k=0)
        with self.assertRaises(ValueError):
            self.model(torch.zeros(2, 3, 30, 30))
        with self.assertRaises(ValueError):
            self.model(self.images, targets=torch.tensor([[91., 0.], [0., 0.]]))

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_checkpoint(self.model, path)
            restored = load_model(path, "cpu")
            torch.testing.assert_close(self.model(self.images).coordinates,
                                       restored(self.images).coordinates)
            torch.save({"model_state_dict": {}}, path)
            with self.assertRaisesRegex(ValueError, "old text GPT"):
                load_model(path, "cpu")

    def test_default_architecture_four_views(self):
        model = StreetGeoModel().eval()
        weight_mib = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
        self.assertTrue(1500 < weight_mib < 1530)
        with torch.no_grad():
            output = model(torch.randn(1, 4, 3, 224, 224), top_k=5)
        self.assertEqual(output.fine_logits.shape, (1, 648, 16))
        self.assertEqual(output.candidate_coordinates.shape, (1, 5, 2))


class PipelineTests(unittest.TestCase):
    def test_local_images_training_checkpoint_and_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            Image.new("RGB", (80, 40), (80, 130, 200)).save(folder / "front.jpg")
            Image.new("RGB", (40, 80), (140, 200, 80)).save(folder / "back.jpg")
            rows = [
                {"image": "front.jpg", "latitude": 34.0522, "longitude": -118.2437},
                {"images": ["front.jpg", "back.jpg"], "latitude": -33.8688,
                 "longitude": 151.2093, "headings": [0, None]},
            ]
            manifest = folder / "samples.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            dataset = StreetViewDataset(manifest, tiny_config())
            batch = collate_streetviews([dataset[0], dataset[1]])
            self.assertEqual(batch["images"].shape, (2, 2, 3, 32, 32))
            self.assertEqual(batch["view_mask"].tolist(), [[True, False], [True, True]])
            self.assertTrue(torch.isfinite(load_image(folder / "front.jpg", 32)).all())
            from dataclasses import asdict
            config_path = folder / "config.json"
            config_path.write_text(json.dumps(asdict(tiny_config())), encoding="utf-8")
            train_manifest = folder / "osv_train.jsonl"
            train_rows = [{"image": name + ".jpg", "osv5m_id": name, "source": "OpenStreetView-5M",
                           "latitude": lat, "longitude": lon}
                          for name, lat, lon in (("front", 34.0522, -118.2437), ("back", -33.8688, 151.2093))]
            train_manifest.write_text("\n".join(json.dumps(row) for row in train_rows), encoding="utf-8")
            checkpoint = folder / "trained.pt"
            root = Path(__file__).resolve().parents[1]
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            trained = subprocess.run([
                sys.executable, "-m", "streetgeo", "train", "--train-manifest", str(train_manifest),
                "--val-manifest", str(train_manifest), "--model-config", str(config_path),
                "--epochs", "1", "--batch-size", "2", "--device", "cpu",
                "--output", str(checkpoint),
            ], capture_output=True, text=True, timeout=90, env=env)
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
            model = load_model(checkpoint, "cpu")
            self.assertEqual(len(predict_images(model, [folder / "front.jpg"], top_k=2)), 2)
            predicted = subprocess.run([
                sys.executable, "-m", "streetgeo", "predict", str(folder / "front.jpg"),
                "--checkpoint", str(checkpoint), "--device", "cpu", "--json",
            ], capture_output=True, text=True, timeout=60, env=env)
            self.assertEqual(predicted.returncode, 0, predicted.stdout + predicted.stderr)
            self.assertIn("maps_url", json.loads(predicted.stdout)[0])


if __name__ == "__main__":
    unittest.main()
