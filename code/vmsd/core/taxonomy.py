"""Taxonomy loader for IV and OM labels."""

from pathlib import Path
from typing import Dict, List

import yaml


class Taxonomy:
    """Loads and exposes the VMSD keyword taxonomy.

    The taxonomy is intentionally stored in YAML so that research categories can
    be edited without changing Python code.
    """

    def __init__(self, taxonomy_path: str | Path):
        self.taxonomy_path = Path(taxonomy_path)
        self._data = self._load_yaml()

    def _load_yaml(self) -> Dict:
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {self.taxonomy_path}")
        with self.taxonomy_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def intrinsic_value_keywords(self) -> List[str]:
        return self._data.get("intrinsic_value_keywords", [])

    @property
    def operational_categories(self) -> Dict[str, Dict]:
        return self._data.get("operational_management_taxonomy", {})

    @property
    def image_evidence_keywords(self) -> List[str]:
        return self._data.get("image_evidence_keywords", [])

    def all_operational_keywords(self) -> Dict[str, List[str]]:
        """Return category -> keywords mapping for OM detection."""
        return {
            category: details.get("keywords", [])
            for category, details in self.operational_categories.items()
        }
