import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from streetgeo.client.build import build_client, sha256
from streetgeo.client import bootstrap
from streetgeo.client import tk_runtime
from streetgeo.core.checkpoints import save_checkpoint
from streetgeo.core.model import StreetGeoModel
from test_model import tiny_config


class ClientTests(unittest.TestCase):
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
            before = sha256(output / "best.pt")
            (output / "notes.txt").write_text("tester notes", encoding="utf-8")
            second = build_client(output)
            self.assertTrue(second["model_preserved"])
            self.assertEqual(sha256(output / "best.pt"), before)
            self.assertEqual((output / "notes.txt").read_text(), "tester notes")
            self.assertFalse((output / "streetgeo/training").exists())
            self.assertFalse((output / "streetgeo/ui/training.py").exists())
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
            result = subprocess.run([sys.executable, "app.py", "--self-test"], cwd=output, env=environment,
                                    capture_output=True, text=True, encoding="utf-8", timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(report["self_test"], "passed")
            self.assertEqual(report["device"], "cpu")
            self.assertEqual(Path(report["runtime_root"]), output)
            self.assertEqual(report["candidates"], 5)

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
                    with self.assertRaisesRegex(RuntimeError, "已经启动"):
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
