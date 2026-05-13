"""Path helpers used by scripts and package modules."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root when scripts are run from /code."""
    return Path(__file__).resolve().parents[3]


def ensure_parent_dir(path: str | Path) -> Path:
    """Create parent directory for a file path and return the Path object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return the Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
