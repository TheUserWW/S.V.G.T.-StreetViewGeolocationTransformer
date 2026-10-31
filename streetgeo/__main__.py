"""Unified command line: python -m streetgeo <command> [options]."""
import argparse
import importlib
import sys


COMMANDS = {
    "gui": ("streetgeo.ui.training", "Open the training workbench"),
    "map": ("streetgeo.ui.prediction", "Open the local prediction map"),
    "train": ("streetgeo.training.train", "Train or resume a model"),
    "predict": ("streetgeo.inference.predict", "Predict coordinates from local images"),
    "download": ("streetgeo.datasets.download_mapillary", "Download Mapillary street views"),
    "prepare": ("streetgeo.datasets.prepare", "Import and split datasets"),
    "split": ("streetgeo.datasets.split", "Split an existing JSONL manifest"),
    "check-gpu": ("streetgeo.cli.check_gpu", "Check CUDA training capacity"),
    "refresh-map": ("streetgeo.maps.refresh", "Refresh offline UN reference geography"),
    "worker": ("streetgeo.ui.worker", "Run a workbench inspection task"),
    "build-client": ("streetgeo.client.build", "Build the standalone prediction client folder"),
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="streetgeo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n" + "\n".join(f"  {name:13} {description}" for name, (_, description) in COMMANDS.items())
        + "\n\nUse: python -m streetgeo <command> --help",
    )
    parser.add_argument("command", choices=COMMANDS, nargs="?", help="Task to run")
    # Parse only the command so subcommand options (including --help) reach its parser.
    command = parser.parse_args(arguments[:1]).command
    if command is None:
        parser.print_help()
        return
    previous_argv = sys.argv
    try:
        sys.argv = [f"{parser.prog} {command}", *arguments[1:]]
        importlib.import_module(COMMANDS[command][0]).main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
