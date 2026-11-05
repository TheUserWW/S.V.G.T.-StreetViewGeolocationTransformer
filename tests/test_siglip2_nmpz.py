from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from streetgeo.core.config import ModelConfig, SIGLIP2_CONFIG_PATH, SIGLIP2_REVISION
from streetgeo.core.model import StreetGeoModel
from streetgeo.datasets.images import load_image
from streetgeo.paths import ROOT
from streetgeo.training.environment import parameter_groups, train_mode


class Siglip2NmpzTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ModelConfig(**json.loads((ROOT / SIGLIP2_CONFIG_PATH).read_text(encoding="utf-8")))
        cls.model = StreetGeoModel(replace(cls.config, geo_smoothing=.1))

    def test_preset_and_single_image_validation(self):
        self.assertEqual(self.config.backbone_revision, SIGLIP2_REVISION)
        with self.assertRaises(ValueError):
            replace(self.config, max_views=2)
        self.assertFalse(hasattr(self.model.encoder, "fusion_blocks"))

    def test_frozen_training_and_terminal_fine_tuning(self):
        def groups(layers):
            return parameter_groups(self.model, unfreeze_last=layers, learning_rate=3e-4,
                                    visual_learning_rate=5e-6, weight_decay=.1)

        self.assertEqual(len(groups(0)), 1)
        train_mode(self.model, 0)
        self.assertFalse(self.model.encoder.patch_embedding.weight.requires_grad)
        self.assertFalse(any(block.training for block in self.model.encoder.visual_blocks))
        self.assertTrue(self.model.encoder.pool_projection.weight.requires_grad)
        inputs = torch.zeros(1, 3, 224, 224)
        output = self.model(inputs, torch.tensor([[36.0, -120.0]]), top_k=1)
        self.assertIn("geo_smoothing", output.loss_components)
        output.loss.backward()
        self.assertIsNotNone(self.model.encoder.pool_projection.weight.grad)
        self.assertIsNone(self.model.encoder.visual_blocks[-1].ln1.weight.grad)
        self.model.zero_grad(set_to_none=True)
        self.assertEqual([group["lr"] for group in groups(2)], [5e-6, 3e-4])
        train_mode(self.model, 2)
        self.assertTrue(self.model.encoder.visual_blocks[-2].ln1.weight.requires_grad)
        self.assertFalse(self.model.encoder.visual_blocks[-3].ln1.weight.requires_grad)
        self.assertFalse(self.model.encoder.visual_blocks[-3].training)

    def test_preprocessing_preserves_the_full_image(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "wide.png"
            Image.new("RGB", (500, 100), (255, 0, 128)).save(image)
            tensor = load_image(image, self.config)
            self.assertEqual(tensor.shape, (3, 224, 224))
            self.assertTrue(torch.all(tensor[0] == 1))
            self.assertTrue(torch.all(tensor[1] == -1))



class GeographicSmoothingTests(unittest.TestCase):
    def test_soft_targets_cross_the_antimeridian_and_exclude_far_cells(self):
        cfg = ModelConfig(image_size=16, patch_size=8, n_embed=16, n_head=2, n_layer=1,
                          fusion_layers=1, coarse_lat_bins=2, coarse_lon_bins=8,
                          fine_bins=2, geo_smoothing=.1, geo_smoothing_km=1500)
        model = StreetGeoModel(cfg)
        target = torch.tensor([[14.5, 179.99]])
        parent, child, _ = model.grid.encode(target)
        true_id = parent * cfg.children_per_cell + child
        other_parent, other_child, _ = model.grid.encode(torch.tensor([[14.5, -179.99]]))
        neighbor = (other_parent * cfg.children_per_cell + other_child).item()
        self.assertNotEqual(parent.item(), other_parent.item())
        log_probs = torch.zeros((1, cfg.total_cells), requires_grad=True)
        loss = model._smooth_cell_loss(log_probs, target, true_id)
        loss.backward()
        weights = -log_probs.grad[0]
        self.assertGreater(weights[neighbor].item(), 0)
        self.assertGreater(weights[true_id.item()].item(), 0)
        self.assertTrue((weights == 0).any())
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)
