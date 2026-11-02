import hashlib
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from streetgeo.core.checkpoints import load_model, read_checkpoint, save_checkpoint
from streetgeo.core.model import StreetGeoModel
from streetgeo.training.online import learn_from_feedback, undo_feedback
from test_model import tiny_config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OnlineLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.base = self.folder / "base.pt"
        self.images = [self.folder / "street.png", self.folder / "second.png"]
        for image, color in zip(self.images, ("green", "orange")):
            Image.new("RGB", (48, 32), color).save(image)
        torch.manual_seed(17)
        save_checkpoint(StreetGeoModel(tiny_config()).eval(), self.base)

    def tearDown(self):
        self.temp.cleanup()

    def learn(self, model=None, image=0, coords=(38.04, -122.98), **kwargs):
        return learn_from_feedback(model or self.base, [self.images[image]], *coords,
                                   output_root=self.folder / "personal", device="cpu", **kwargs)

    def test_feedback_changes_heads_preserves_encoder_and_base_and_loads_for_inference(self):
        original_hash = digest(self.base)
        result = self.learn()
        self.assertNotEqual(result["checkpoint"], str(self.base))
        self.assertEqual(digest(self.base), original_hash)
        self.assertLess(result["loss_after"], result["loss_before"])
        a = read_checkpoint(self.base)["model_state_dict"]
        learned = read_checkpoint(result["checkpoint"])
        b = learned["model_state_dict"]
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a if k.startswith("encoder.")))
        self.assertTrue(any(not torch.equal(a[k], b[k]) for k in a if not k.startswith("encoder.")))
        state = learned["training_state"]["online_learning"]
        self.assertEqual(state["memory"][0]["coordinates"], [38.04, -122.98])
        self.assertEqual(state["validation_status"], "unvalidated_user_feedback")
        model = load_model(result["checkpoint"], "cpu")
        self.assertEqual(len(model.predict(torch.zeros(1, 1, 3, 32, 32), top_k=1)[0]), 1)

    def test_replay_survives_reload_duplicate_label_replaces_and_undo_restores(self):
        first = self.learn()
        personal = first["checkpoint"]
        one_hash = digest(personal)
        second = self.learn(personal, image=1, coords=(35.6, 139.7))
        self.assertEqual(second["updates"], 2)
        self.assertEqual(second["replayed"], 1)
        self.assertEqual(second["memory_size"], 2)
        undone = undo_feedback(personal)
        self.assertEqual(undone["updates"], 1)
        self.assertEqual(digest(personal), one_hash)
        revised = self.learn(personal, coords=(38.1, -123.0))
        self.assertEqual(revised["memory_size"], 1)
        state = read_checkpoint(personal)["training_state"]["online_learning"]
        self.assertEqual(state["memory"][0]["coordinates"], [38.1, -123.0])
        undo_feedback(personal)
        back = undo_feedback(personal)
        self.assertEqual(back["checkpoint"], str(self.base))

    def test_invalid_label_and_official_test_do_not_create_a_personal_model(self):
        with self.assertRaises(ValueError):
            self.learn(coords=(91, 0))
        image = self.folder / "OpenStreetView-5M" / "test" / "00" / "123.jpg"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(image)
        with self.assertRaisesRegex(ValueError, "test"):
            learn_from_feedback(self.base, [image], 0, 0, output_root=self.folder / "personal")
        self.assertFalse((self.folder / "personal").exists())

    def test_failed_save_leaves_committed_personal_weights_unchanged(self):
        first = self.learn()
        before = digest(first["checkpoint"])
        with patch("streetgeo.training.online.save_checkpoint", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.learn(first["checkpoint"], image=1)
        self.assertEqual(digest(first["checkpoint"]), before)

    def test_feature_head_route_matches_original_forward(self):
        model = load_model(self.base, "cpu")
        x = torch.randn(2, 1, 3, 32, 32)
        y = torch.tensor([[38., -123.], [35., 139.]])
        with torch.no_grad():
            normal = model(x, y)
            cached = model.forward_embedding(model.encoder(x), y)
        self.assertTrue(torch.equal(normal.coarse_logits, cached.coarse_logits))
        self.assertTrue(torch.equal(normal.fine_logits, cached.fine_logits))
        self.assertTrue(torch.equal(normal.coordinates, cached.coordinates))
        self.assertTrue(torch.equal(normal.loss, cached.loss))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_feedback_saves_a_cpu_loadable_personal_model(self):
        result = learn_from_feedback(self.base, [self.images[0]], 38.04, -122.98,
                                     device="cuda", output_root=self.folder / "cuda_personal")
        self.assertLess(result["loss_after"], result["loss_before"])
        loaded = load_model(result["checkpoint"], "cpu")
        self.assertEqual(next(loaded.parameters()).device.type, "cpu")


class OnlineLearningGuiTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from streetgeo.ui.prediction import PredictionApp
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = PredictionApp(self.root, settings_path=self.folder / "settings.json")
        self.errors = []
        self.app.error = self.errors.append
        self.base, self.image = self.folder / "base.pt", self.folder / "street.png"
        save_checkpoint(StreetGeoModel(tiny_config()).eval(), self.base)
        Image.new("RGB", (48, 32), "blue").save(self.image)
        self.app.images = [str(self.image)]
        self.app.checkpoint.set(str(self.base))
        self.app.device.set("cpu")
        self.app.show_result(self.app.engine.predict(self.base, [self.image], device="cpu"))
        self.root.update_idletasks()

    def tearDown(self):
        self.app.close()
        self.temp.cleanup()

    def wait(self):
        deadline = time.monotonic() + 15
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertFalse(self.app.busy)
        self.assertFalse(self.errors)

    def test_map_click_updates_personal_selection_persists_and_undoes(self):
        before = digest(self.base)
        self.app.begin_learning()
        self.assertTrue(self.app.learning_point)
        self.app.map_click(38.04, -122.98)
        self.assertTrue(self.app.busy)
        self.wait()
        self.assertNotEqual(self.app.checkpoint.get(), str(self.base))
        self.assertTrue(Path(self.app.checkpoint.get()).is_file())
        self.assertEqual(digest(self.base), before)
        self.assertEqual(self.app.map.ground_truth["source"], "user_feedback")
        self.assertIn("更新前", self.app.status.get())
        self.app.undo_learning()
        self.wait()
        self.assertEqual(self.app.checkpoint.get(), str(self.base))
        self.assertIsNone(self.app.result)

    def test_stale_prediction_and_official_test_cannot_arm_learning(self):
        self.app.test_sample = {"source": "official_test"}
        self.app.begin_learning()
        self.assertFalse(self.app.learning_point)
        self.assertIn("test", str(self.errors.pop()))
        self.app.test_sample = None
        self.app.checkpoint.set(str(self.folder / "other.pt"))
        self.app.begin_learning()
        self.assertFalse(self.app.learning_point)
        self.assertIn("重新预测", str(self.errors.pop()))


if __name__ == "__main__":
    unittest.main()
