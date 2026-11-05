from pathlib import Path
import tempfile
import unittest

from netCDF4 import Dataset
import torch

from streetgeo.training.climate import (UNKNOWN_CLIMATE, climate_logits, climate_loss, climate_metrics,
                                        sample_climates, update_confusion)


class ClimateTrainingTests(unittest.TestCase):
    def test_raster_lookup_masks_ocean_low_confidence_and_nodata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "climate.nc"
            with Dataset(path, "w") as raster:
                raster.createDimension("lat", 2)
                raster.createDimension("lon", 4)
                raster.createVariable("lat", "f4", ("lat",))[:] = [30, -30]
                raster.createVariable("lon", "f4", ("lon",))[:] = [-135, -45, 45, 135]
                classes = raster.createVariable("kg_class", "u1", ("lat", "lon"),
                                                fill_value=255, chunksizes=(1, 2))
                scores = raster.createVariable("kg_confidence", "u1", ("lat", "lon"),
                                               fill_value=255, chunksizes=(1, 2))
                classes[:] = [[1, 7, 15, 25], [29, 0, 9, 255]]
                scores[:] = [[100, 80, 79, 100], [100, 99, 255, 100]]
            labels = sample_climates(path, [(30, -135), (30, -45), (30, 45), (30, 135), (-30, -135),
                                             (-30, -45), (-30, 45), (-30, 135), (30, 180)], 80)
        self.assertEqual(labels, [0, 1, UNKNOWN_CLIMATE, 3, 4, UNKNOWN_CLIMATE, UNKNOWN_CLIMATE, UNKNOWN_CLIMATE, 0])

    def test_unlabeled_images_receive_no_climate_gradient(self):
        features = torch.randn(3, 768, requires_grad=True)
        head = torch.nn.Linear(768, 5)
        logits = climate_logits(head, torch.nn.functional.normalize(features, dim=-1))
        loss = climate_loss(logits, torch.tensor([0, UNKNOWN_CLIMATE, 3]))
        loss.backward()
        self.assertEqual(int(features.grad[1].count_nonzero()), 0)
        self.assertGreater(int(features.grad[0].count_nonzero()), 0)
        self.assertEqual(float(climate_loss(logits.detach(), torch.full((3,), UNKNOWN_CLIMATE))), 0)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 3845)

    def test_climate_metrics_exclude_missing_and_report_class_support(self):
        confusion = torch.zeros(5, 5, dtype=torch.long)
        targets = torch.tensor([0, 0, 1, UNKNOWN_CLIMATE])
        logits = torch.nn.functional.one_hot(torch.tensor([0, 1, 1, 4]), num_classes=5)
        update_confusion(confusion, logits, targets)
        report = climate_metrics(confusion)
        self.assertEqual(report["support"], {"A": 2, "B": 1, "C": 0, "D": 0, "E": 0})
        self.assertEqual(report["labeled"], 3)
        self.assertAlmostEqual(report["accuracy"], 2 / 3)
        self.assertAlmostEqual(report["macro_f1"], 2 / 3)
