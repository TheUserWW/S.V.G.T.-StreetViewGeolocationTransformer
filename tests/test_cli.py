from contextlib import redirect_stderr, redirect_stdout
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from streetgeo.__main__ import main


class CommandLineTests(unittest.TestCase):
    def test_root_help_and_unknown_command_do_not_start_a_task(self):
        output = io.StringIO()
        with redirect_stdout(output), patch("streetgeo.__main__.importlib.import_module") as imported:
            main([])
            self.assertIn("refresh-map", output.getvalue())
            imported.assert_not_called()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["missing-command"])
        self.assertEqual(error.exception.code, 2)

    def test_subcommand_receives_help_and_space_containing_arguments(self):
        received = []
        task = SimpleNamespace(main=lambda: received.extend(sys.argv))
        original = sys.argv
        with patch("streetgeo.__main__.importlib.import_module", return_value=task) as imported:
            main(["train", "--train-manifest", "data set/train.jsonl", "--help"])
        imported.assert_called_once_with("streetgeo.training.train")
        self.assertEqual(received, ["streetgeo train", "--train-manifest", "data set/train.jsonl", "--help"])
        self.assertIs(sys.argv, original)


if __name__ == "__main__":
    unittest.main()
