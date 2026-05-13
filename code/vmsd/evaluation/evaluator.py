"""Evaluation runner for prediction files."""

from pathlib import Path
from typing import Dict

import pandas as pd

from vmsd.data.exporter import ResultExporter
from vmsd.data.loader import ReviewDataLoader
from vmsd.evaluation.metrics import ClassificationMetrics


class VMSDEvaluator:
    """Evaluates predicted VMSD labels against a ground-truth label column."""

    def __init__(self):
        self.loader = ReviewDataLoader()
        self.metrics = ClassificationMetrics()
        self.exporter = ResultExporter()

    def evaluate(
        self,
        input_path: str | Path,
        label_column: str,
        prediction_column: str,
        output_path: str | Path | None = None,
    ) -> Dict:
        df = self.loader.load(input_path)
        if label_column not in df.columns:
            raise ValueError(f"Ground-truth label column not found: {label_column}")
        if prediction_column not in df.columns:
            raise ValueError(f"Prediction column not found: {prediction_column}")

        y_true = self.metrics.normalize_label_series(df[label_column])
        y_pred = self.metrics.normalize_label_series(df[prediction_column])
        report = self.metrics.compute(y_true, y_pred)
        report["rows_evaluated"] = int(len(df))
        report["label_column"] = label_column
        report["prediction_column"] = prediction_column

        if output_path:
            self.exporter.save_json(report, output_path)
        return report
