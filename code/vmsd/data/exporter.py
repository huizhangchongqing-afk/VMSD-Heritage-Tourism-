"""Output writers for VMSD results."""

from pathlib import Path
from typing import Dict

import json
import pandas as pd

from vmsd.utils.path_utils import ensure_parent_dir


class ResultExporter:
    """Exports predictions and reports to disk."""

    def save_dataframe(self, df: pd.DataFrame, output_path: str | Path) -> Path:
        path = ensure_parent_dir(output_path)
        if path.suffix.lower() == ".csv":
            df.to_csv(path, index=False)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df.to_excel(path, index=False)
        else:
            raise ValueError("Output file must be .csv or .xlsx")
        return path

    def save_json(self, data: Dict, output_path: str | Path) -> Path:
        path = ensure_parent_dir(output_path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path
