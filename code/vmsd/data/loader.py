"""Dataset loading utilities.

Supports CSV and Excel because the VMSD data is usually handled as an Excel sheet.
"""

from pathlib import Path

import pandas as pd


class ReviewDataLoader:
    """Loads review data from CSV/XLSX files."""

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

    def load(self, file_path: str | Path) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported dataset format: {path.suffix}. Use CSV or Excel."
            )

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_excel(path)
