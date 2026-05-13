"""Train a lightweight text classifier for VMSD.

This ML model is optional. The main VMSD pipeline remains rule-based and
explainable, but this classifier is useful for comparison experiments.
"""

from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from vmsd.data.loader import ReviewDataLoader
from vmsd.evaluation.metrics import ClassificationMetrics
from vmsd.modeling.model_registry import ModelRegistry


class VMSDTextModelTrainer:
    """Trains TF-IDF + Logistic Regression on review text and image descriptions."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.loader = ReviewDataLoader()
        self.registry = ModelRegistry()
        self.metrics = ClassificationMetrics()

    def train(
        self,
        input_path: str | Path,
        model_output_path: str | Path,
        label_column: str = "final_label",
        text_column: str = "review_text",
        image_description_column: str = "image_description",
    ) -> Dict:
        df = self.loader.load(input_path)
        if label_column not in df.columns:
            raise ValueError(f"Label column not found: {label_column}")
        if text_column not in df.columns:
            raise ValueError(f"Text column not found: {text_column}")

        df = df.copy()
        df[text_column] = df[text_column].fillna("").astype(str)
        if image_description_column not in df.columns:
            df[image_description_column] = ""
        df[image_description_column] = df[image_description_column].fillna("").astype(str)

        X = (df[text_column] + " " + df[image_description_column]).str.strip()
        y = self.metrics.normalize_label_series(df[label_column])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=self.random_state,
            stratify=y if y.nunique() > 1 else None,
        )

        model = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        self.registry.save(model, model_output_path)
        return {
            "model_path": str(model_output_path),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "report": report,
        }
