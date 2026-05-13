"""Configuration management for the VMSD project."""

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigManager:
    """Loads the YAML config and provides safe dotted-key access.

    Example:
        config.get("columns.text") -> "review_text"
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Read nested config keys using dot notation."""
        current: Any = self.data
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    @property
    def columns(self) -> Dict[str, str]:
        return self.data.get("columns", {})
