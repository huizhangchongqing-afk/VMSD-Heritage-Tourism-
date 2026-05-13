"""Command-line entry point for running the full VMSD pipeline."""

import argparse
from pathlib import Path
import sys

# Allow imports from code/vmsd when running from project root.
sys.path.append(str(Path(__file__).resolve().parent))

from vmsd.data.exporter import ResultExporter
from vmsd.pipeline import VMSDPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run VMSD heritage-tourism pipeline")
    parser.add_argument("--input", required=True, help="Path to CSV/XLSX dataset")
    parser.add_argument("--output", default="outputs/vmsd_predictions.csv", help="Output CSV/XLSX path")
    parser.add_argument("--config", default=None, help="Optional config.yaml path")
    parser.add_argument("--taxonomy", default=None, help="Optional label_taxonomy.yaml path")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = VMSDPipeline(config_path=args.config, taxonomy_path=args.taxonomy)
    predictions = pipeline.run(args.input)
    output_path = ResultExporter().save_dataframe(predictions, args.output)
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
