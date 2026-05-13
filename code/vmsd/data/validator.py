"""Dataset validation and light repair.

The validator does not fail hard for every missing column because real annotation
files often evolve during research. Required columns are created with safe values
when possible so the pipeline can still run.
"""

from typing import Dict, List

import pandas as pd


class DatasetValidator:
    """Validates and normalizes the expected VMSD dataset schema."""

    def __init__(self, columns: Dict[str, str]):
        self.columns = columns

    def validate_and_repair(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        required_defaults = {
            self.columns.get("id", "review_id"): lambda i: f"R{i + 1:04d}",
            self.columns.get("site", "heritage_site"): "Unknown Site",
            self.columns.get("rating", "review_rating"): 5,
            self.columns.get("text", "review_text"): "",
            self.columns.get("image_description", "image_description"): "",
        }

        for col, default in required_defaults.items():
            if col not in df.columns:
                if callable(default):
                    df[col] = [default(i) for i in range(len(df))]
                else:
                    df[col] = default

        # Normalize text-like fields so downstream regex never receives NaN.
        for col in [
            self.columns.get("text", "review_text"),
            self.columns.get("image_description", "image_description"),
            self.columns.get("site", "heritage_site"),
        ]:
            df[col] = df[col].fillna("").astype(str)

        # Ratings are expected to be numeric. Invalid values become 0.
        rating_col = self.columns.get("rating", "review_rating")
        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce").fillna(0)

        return df

    def missing_columns(self, df: pd.DataFrame) -> List[str]:
        """Return expected columns that are absent from the dataframe."""
        expected = [col for col in self.columns.values() if col]
        return [col for col in expected if col not in df.columns]
