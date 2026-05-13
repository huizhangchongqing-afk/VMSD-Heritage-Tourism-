"""Evaluation metric helpers."""

from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


class ClassificationMetrics:
    """Computes binary VMSD classification metrics."""

    def compute(self, y_true, y_pred) -> Dict:
        labels = ["No", "Yes"]
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision_yes": round(float(precision_score(y_true, y_pred, pos_label="Yes", zero_division=0)), 4),
            "recall_yes": round(float(recall_score(y_true, y_pred, pos_label="Yes", zero_division=0)), 4),
            "f1_yes": round(float(f1_score(y_true, y_pred, pos_label="Yes", zero_division=0)), 4),
            "confusion_matrix_labels": labels,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
        }

    @staticmethod
    def normalize_label_series(series: pd.Series) -> pd.Series:
        """Normalize label text to Yes/No."""
        def norm(x):
            text = str(x).strip().lower()
            if text in {"yes", "y", "true", "1", "positive", "vmsd", "vmsd-positive"}:
                return "Yes"
            if text in {"no", "n", "false", "0", "negative", "non-vmsd", "vmsd-negative"}:
                return "No"
            return str(x).strip()

        return series.apply(norm)
