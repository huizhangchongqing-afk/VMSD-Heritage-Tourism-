"""Model saving/loading helpers."""

from pathlib import Path

import joblib

from vmsd.utils.path_utils import ensure_parent_dir


class ModelRegistry:
    """Tiny registry wrapper around joblib."""

    def save(self, model, path: str | Path) -> Path:
        path = ensure_parent_dir(path)
        joblib.dump(model, path)
        return path

    def load(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return joblib.load(path)
