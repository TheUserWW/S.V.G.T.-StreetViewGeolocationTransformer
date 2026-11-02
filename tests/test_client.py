import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from streetgeo.client.build import build_archive, build_client, sha256, MODEL_SUFFIXES
from streetgeo.client import bootstrap
from streetgeo.client import tk_runtime
from streetgeo.core.checkpoints import save_checkpoint
from streetgeo.core.model import StreetGeoModel
from test_model import tiny_config


class ClientTests(unittest.TestCase):
    def test_launcher_setup_does_not_require_a_bundled_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "app.py").touch()
            with patch.object(bootstrap, "ROOT", root), patch.object(bootstrap, "RUNTIME", root / ".runtime"), \
                    patch.object(bootstrap, "check_python"), patch.object(bootstrap, "ensure_environment", return_value="python"), \
                    patch("sys.argv", ["bootstrap.py", "--setup-only"]), redirect_stdout(io.StringIO()):
                self.assertEqual(bootstrap.main(), 0)
            self.assertFalse((root / "best.pt").exists())

    def test_tcl_fallback_initializes_window_when_base_script_paths_are_missing(self):
        version = tk_runtime.runtime_version()
        if version != '8.6.15':
            self.skipTest('Bundled fallback regression uses the local 8.6.15 DLL')
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(sys, 'base_prefix', folder), patch.object(sys, '_base_executable', str(Path(folder)/'python.exe')):
                with patch.dict(os.environ, {'TCL_LIBRARY':'missing-tcl','TK_LIBRARY':'missing-tk'}):
                    paths = tk_runtime.configure_tk()
                    self.assertIn('vendor', paths[0])
                    tk_runtime.check_tk()

    def test_bundle_is_closed_and_preserves_existing_weight_and_user_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "model.pt"
            save_checkpoint(StreetGeoModel(tiny_config()), checkpoint)
            output = root / "client"
            build_client(output, checkpoint)
            self.assertFalse((output / "best.pt").exists())
            # An old locally retained model must survive rebuilds but never enter a ZIP.
            (output / "best.pt").write_bytes(checkpoint.read_bytes())
            before = sha256(output / "best.pt")
            (output / "notes.txt").write_text("tester notes", encoding="utf-8")
            # Upgrade an older full-feature client without leaving its developer UI.
            manifest_path = output / "CLIENT_MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for obsolete in ("streetgeo/ui/prediction.py", "streetgeo/inference/osv_test.py"):
                path = output / obsolete
                path.write_text("# previous generated UI\n", encoding="utf-8")
                manifest["files"][obsolete] = {"sha256": sha256(path)}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            second = build_client(output)
            self.assertFalse(second["model_included"])
            self.assertEqual(sha256(output / "best.pt"), before)
            self.assertEqual((output / "notes.txt").read_text(), "tester notes")
            self.assertTrue((output / "streetgeo/training/online.py").is_file())
            self.assertFalse((output / "streetgeo/training/train.py").exists())
            self.assertFalse((output / "streetgeo/ui/training.py").exists())
            self.assertFalse((output / "streetgeo/ui/prediction.py").exists())
            self.assertFalse((output / "streetgeo/inference/osv_test.py").exists())
            self.assertFalse((output / "streetgeo/inference/evaluate.py").exists())
            self.assertFalse((output / "streetgeo/inference/evaluation_charts.py").exists())
            self.assertFalse((output / "streetgeo/ui/evaluation.py").exists())
            self.assertTrue((output / "streetgeo/inference/explain.py").is_file())
            self.assertTrue((output / "streetgeo/inference/explanation_text.py").is_file())
            self.assertTrue((output / "streetgeo/ui/explanation.py").is_file())
            self.assertTrue((output / "client_ui.py").is_file())
            self.assertTrue((output / "client_i18n.py").is_file())
            self.assertTrue((output / "streetgeo/inference/coverage.py").is_file())
            self.assertTrue((output / "assets/maps/google_streetview_regions.json").is_file())
            self.assertNotIn("evaluate_test_set", (output / "client_ui.py").read_text(encoding="utf-8"))
            with zipfile.ZipFile(second["archive"]) as archive:
                self.assertIsNone(archive.testzip())
                self.assertFalse(any(Path(name).suffix.lower() in MODEL_SUFFIXES for name in archive.namelist()))
                self.assertNotIn("client/notes.txt", archive.namelist())
            for source in (output / "streetgeo").rglob("*.py"):
                for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("streetgeo"):
                        module = output / node.module.replace(".", "/")
                        self.assertTrue(module.with_suffix(".py").is_file() or (module / "__init__.py").is_file(),
                                        f"Missing client dependency {node.module}")

    def test_client_runs_from_unrelated_unicode_path(self):
        with tempfile.TemporaryDirectory(prefix="StreetGeo 客户端 ") as folder:
            root = Path(folder)
            checkpoint = root / "model.pt"
            save_checkpoint(StreetGeoModel(tiny_config()), checkpoint)
            output = root / "portable folder"
            build_client(output, checkpoint)
            environment = dict(os.environ, PYTHONPATH="", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            self.assertFalse((output / "best.pt").exists())
            ui_only = subprocess.run([sys.executable, "app.py", "--self-test"], cwd=output, env=environment,
                                    capture_output=True, text=True, encoding="utf-8", timeout=90)
            self.assertEqual(ui_only.returncode, 0, ui_only.stdout + ui_only.stderr)
            self.assertEqual(json.loads(ui_only.stdout.strip().splitlines()[-1])["prediction"], "not_run")
            result = subprocess.run([sys.executable, "app.py", "--self-test", "--checkpoint", str(checkpoint)], cwd=output, env=environment,
                                    capture_output=True, text=True, encoding="utf-8", timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(report["self_test"], "passed")
            self.assertEqual(report["explanation"], "passed")
            self.assertEqual(report["device"], "cpu")
            self.assertEqual(Path(report["runtime_root"]), output)
            self.assertEqual(report["candidates"], 5)
            self.assertEqual(Path(report["checkpoint"]), checkpoint)

    def test_archive_rejects_weights_even_if_added_to_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "client"
            result = build_client(output)
            original = sha256(result["archive"])
            manifest_path = output / "CLIENT_MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for suffix in (".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin"):
                model = "weights" + suffix
                (output / model).touch()
                manifest["files"][model] = {"sha256": sha256(output / model)}
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "model weights"):
                    build_archive(output)
                del manifest["files"][model]
            self.assertEqual(sha256(result["archive"]), original)

    def test_environment_paths_are_local_and_change_after_relocation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(bootstrap, "ROOT", root):
                with patch.object(bootstrap, "RUNTIME", root / ".runtime"):
                    first = bootstrap.environment_path()
                    self.assertTrue(first.is_relative_to(root / ".runtime"))
                    with self.assertRaises(RuntimeError):
                        bootstrap.safe_runtime_path(root.parent / "outside-client")
            relocated = root / "relocated"
            with patch.object(bootstrap, "ROOT", relocated), patch.object(bootstrap, "RUNTIME", relocated / ".runtime"):
                second = bootstrap.environment_path()
                self.assertNotEqual(first.name, second.name)

    def test_startup_lock_blocks_concurrent_install_and_releases(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(bootstrap, "ROOT", root), patch.object(bootstrap, "RUNTIME", root / ".runtime"):
                with bootstrap.startup_lock():
                    with self.assertRaisesRegex(RuntimeError, "已经启动|already running"):
                        with bootstrap.startup_lock():
                            self.fail("The same client cannot initialize concurrently")
                with bootstrap.startup_lock():
                    pass

    def test_dependency_subprocess_does_not_inherit_project_import_paths(self):
        with patch.dict(os.environ, {"PYTHONPATH": "parent-project", "PYTHONHOME": "other-python"}):
            environment = bootstrap.child_environment()
            self.assertNotIn("PYTHONPATH", environment)
            self.assertNotIn("PYTHONHOME", environment)
            self.assertEqual(environment["PYTHONNOUSERSITE"], "1")


if __name__ == "__main__":
    unittest.main()
