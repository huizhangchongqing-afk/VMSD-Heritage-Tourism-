"""Command-line entry point for evaluation."""

import argparse
from pathlib import Path
import pprint
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from vmsd.evaluation.evaluator import VMSDEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VMSD predictions")
    parser.add_argument("--input", required=True, help="Prediction file path")
    parser.add_argument("--label-column", default="final_label", help="Ground truth label column")
    parser.add_argument("--prediction-column", default="pred_vmsd_label", help="Prediction label column")
    parser.add_argument("--output", default="outputs/evaluation_report.json", help="Evaluation JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()
    report = VMSDEvaluator().evaluate(
        input_path=args.input,
        label_column=args.label_column,
        prediction_column=args.prediction_column,
        output_path=args.output,
    )
    pprint.pp(report)
    print(f"Saved evaluation report to: {args.output}")


if __name__ == "__main__":
    main()
