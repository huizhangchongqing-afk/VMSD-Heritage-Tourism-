"""Minimal smoke test for VMSD scorer.

Run from project root:
    pytest tests/
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "code"))

from vmsd.pipeline import VMSDPipeline


def test_pipeline_runs_on_sample():
    root = Path(__file__).resolve().parents[1]
    pipeline = VMSDPipeline()
    df = pipeline.run(root / "dataset" / "sample_vmsd_reviews.csv")
    assert "pred_vmsd_label" in df.columns
    assert len(df) > 0
