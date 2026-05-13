"""Command-line entry point for generating VMSD plots."""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from vmsd.data.loader import ReviewDataLoader
from vmsd.visualization.plots import VMSDPlotter


def parse_args():
    parser = argparse.ArgumentParser(description="Create plots from VMSD prediction file")
    parser.add_argument("--input", required=True, help="Prediction CSV/XLSX path")
    parser.add_argument("--output-dir", default="outputs/plots", help="Directory for generated plots")
    return parser.parse_args()


def main():
    args = parse_args()
    df = ReviewDataLoader().load(args.input)
    paths = VMSDPlotter(args.output_dir).create_all(df)
    print("Generated plots:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
