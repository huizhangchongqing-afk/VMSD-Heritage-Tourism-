"""Command-line entry point for training optional ML baseline."""

import argparse
from pathlib import Path
import pprint
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from vmsd.modeling.train_text_model import VMSDTextModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train VMSD text classifier")
    parser.add_argument("--input", required=True, help="CSV/XLSX dataset path")
    parser.add_argument("--model-output", default="outputs/vmsd_text_classifier.joblib", help="Model output path")
    parser.add_argument("--label-column", default="final_label", help="Label column to train on")
    return parser.parse_args()


def main():
    args = parse_args()
    report = VMSDTextModelTrainer().train(
        input_path=args.input,
        model_output_path=args.model_output,
        label_column=args.label_column,
    )
    pprint.pp(report)
    print(f"Saved model to: {args.model_output}")


if __name__ == "__main__":
    main()
