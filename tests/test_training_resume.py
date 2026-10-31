from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from streetgeo.core.checkpoints import capture_rng_state, read_checkpoint, restore_rng_state, save_checkpoint
from streetgeo.core.model import StreetGeoModel
from test_model import tiny_config
from streetgeo.training.runtime import EpochBatchSampler, StopRequest


class ResumeTests(unittest.TestCase):
    def test_sampler_resumes_without_replaying_completed_indices(self):
        full = list(EpochBatchSampler(11, 3, 1337, 2))
        resumed = list(EpochBatchSampler(11, 3, 1337, 2, start_batch=2))
        self.assertEqual(full[2:], resumed)
        self.assertEqual(sorted(item for batch in full for item in batch), list(range(11)))

    def test_failed_atomic_write_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            model = StreetGeoModel(tiny_config())
            save_checkpoint(model, path, training_state={"step": 1})
            original = path.read_bytes()
            with patch("streetgeo.core.checkpoints.torch.save", side_effect=OSError("simulated disk failure")):
                with self.assertRaises(OSError):
                    save_checkpoint(model, path, training_state={"step": 2}, keep_previous=True)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])
            save_checkpoint(model, path, training_state={"step": 2}, keep_previous=True)
            self.assertEqual(read_checkpoint(Path(folder) / "last.previous.pt")["training_state"]["step"], 1)

    def test_first_signal_requests_graceful_stop(self):
        stop = StopRequest()
        stop._handle(2, None)
        self.assertTrue(stop.requested)
        with self.assertRaises(KeyboardInterrupt):
            stop._handle(2, None)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_bf16_checkpoint_restores_rng_and_adam_state(self):
        torch.manual_seed(42)
        model = StreetGeoModel(tiny_config(dropout=0.2)).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        images = torch.randn(2, 3, 32, 32, device="cuda")
        targets = torch.tensor([[34., -118.], [40., -74.]], device="cuda")

        def step(current_model, current_optimizer):
            current_optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = current_model(images, targets, top_k=1)
            output.loss.backward()
            current_optimizer.step()

        step(model, optimizer)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cuda.pt"
            save_checkpoint(model, path, training_state={"optimizer": optimizer.state_dict(), "rng": capture_rng_state()})
            step(model, optimizer)
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
            saved = read_checkpoint(path)
            restored = StreetGeoModel(tiny_config(dropout=0.2)).cuda().train()
            restored.load_state_dict(saved["model_state_dict"])
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
            restored_optimizer.load_state_dict(saved["training_state"]["optimizer"])
            restore_rng_state(saved["training_state"]["rng"])
            step(restored, restored_optimizer)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[name], atol=2e-5, rtol=2e-4)

    def test_interrupted_run_matches_uninterrupted_weights_and_optimizer(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = []
            for index in range(6):
                Image.new("RGB", (40, 32), (index * 35, 120, 200 - index * 20)).save(folder / f"{index}.png")
                rows.append({"image": f"{index}.png", "latitude": 25 + index * 5,
                             "longitude": -120 + index * 7, "source": "OpenStreetView-5M", "osv5m_id": str(index)})
            manifest = folder / "train.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            cfg = folder / "model.json"
            # Nonzero dropout proves RNG and data-order restoration, not just weights.
            cfg.write_text(json.dumps(asdict(tiny_config(dropout=0.2))), encoding="utf-8")
            command = [sys.executable, "-m", "streetgeo", "train"]
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

            def run(arguments, success=True):
                result = subprocess.run([*command, *arguments], env=env,
                                        capture_output=True, text=True, timeout=90)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result

            common = ["--train-manifest", str(manifest), "--val-manifest", str(manifest),
                      "--model-config", str(cfg), "--epochs", "2", "--batch-size", "2",
                      "--device", "cpu", "--num-workers", "0", "--save-every-steps", "1"]
            full = folder / "full" / "best.pt"
            partial = folder / "partial" / "best.pt"
            run([*common, "--output", str(full)])
            # Windows spawn workers must not change shuffle/dropout RNG behavior.
            run([*common, "--output", str(partial), "--num-workers", "2", "--stop-after-steps", "2"])
            partial_last = partial.with_name("best.last.pt")
            interrupted = read_checkpoint(partial_last)
            self.assertEqual(interrupted["training_state"]["next_batch"], 2)
            self.assertEqual(interrupted["training_state"]["global_step"], 2)
            failed = run([*common, "--output", str(partial)], False)
            self.assertIn("Training output already exists", failed.stderr)
            run(["--resume", str(partial_last), "--device", "cpu"])
            expected = read_checkpoint(full.with_name("best.last.pt"))
            actual = read_checkpoint(partial_last)
            for name, tensor in expected["model_state_dict"].items():
                torch.testing.assert_close(tensor, actual["model_state_dict"][name], atol=0, rtol=0)
            expected_state, actual_state = expected["training_state"], actual["training_state"]
            self.assertEqual(actual_state["global_step"], 6)
            self.assertEqual(expected_state["lr_schedule"], actual_state["lr_schedule"])
            torch.testing.assert_close(expected_state["rng"]["torch"], actual_state["rng"]["torch"])
            for identifier, values in expected_state["optimizer"]["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, actual_state["optimizer"]["state"][identifier][name], atol=0, rtol=0)
            failed = run(["--resume", str(partial_last), "--batch-size", "3", "--device", "cpu"], False)
            self.assertIn("unchanged batch_size", failed.stderr)
            manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            failed = run(["--resume", str(partial_last), "--device", "cpu"], False)
            self.assertIn("Manifests changed", failed.stderr)


if __name__ == "__main__":
    unittest.main()
