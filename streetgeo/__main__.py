"""Unified command line: python -m streetgeo <command> [options]."""
import argparse
import importlib
import sys
from streetgeo import PROJECT_TITLE


COMMANDS = {
    "train": ("streetgeo.training.train", "Train or resume a Mapillary model"),
    "download": ("streetgeo.datasets.download_mapillary", "Download Mapillary street views"),
    "prepare": ("streetgeo.datasets.prepare", "Import and split Mapillary"),
    "split": ("streetgeo.datasets.split", "Split a Mapillary JSONL manifest"),
    "check-gpu": ("streetgeo.cli.check_gpu", "Check CUDA training capacity"),
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="streetgeo", description=f"{PROJECT_TITLE}\n\n{__doc__}", formatter_class=argparse.RawDescriptionHelpFormatter,
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
