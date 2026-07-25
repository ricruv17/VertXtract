from pathlib import Path
import argparse

from main_viewer import run

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing .VERT data"
    )

    args = parser.parse_args()

    run(Path(args.directory))
