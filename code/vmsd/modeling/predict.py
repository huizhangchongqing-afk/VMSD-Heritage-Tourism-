"""Prediction helper for trained text classifier."""

from pathlib import Path

import pandas as pd

from vmsd.data.loader import ReviewDataLoader
from vmsd.modeling.model_registry import ModelRegistry


class VMSDTextPredictor:
    """Runs a saved text classifier on a dataset."""

    def __init__(self, model_path: str | Path):
        self.model = ModelRegistry().load(model_path)
        self.loader = ReviewDataLoader()

    def predict_dataframe(
        self,
        input_path: str | Path,
        text_column: str = "review_text",
        image_description_column: str = "image_description",
    ) -> pd.DataFrame:
        df = self.loader.load(input_path).copy()
        df[text_column] = df[text_column].fillna("").astype(str)
        if image_description_column not in df.columns:
            df[image_description_column] = ""
        df[image_description_column] = df[image_description_column].fillna("").astype(str)

        X = (df[text_column] + " " + df[image_description_column]).str.strip()
        df["ml_pred_vmsd_label"] = self.model.predict(X)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            class_names = list(self.model.classes_)
            if "Yes" in class_names:
                yes_index = class_names.index("Yes")
                df["ml_pred_yes_probability"] = proba[:, yes_index]

        return df
