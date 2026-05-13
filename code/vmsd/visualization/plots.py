"""Plot generation for VMSD outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vmsd.utils.path_utils import ensure_dir


class VMSDPlotter:
    """Creates paper/report-ready plots from prediction outputs."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = ensure_dir(output_dir)

    def create_all(self, df: pd.DataFrame) -> list[Path]:
        created = []
        if "pred_vmsd_label" in df.columns:
            created.append(self.plot_count(df, "pred_vmsd_label", "VMSD Label Distribution", "vmsd_label_distribution.png"))
        if "pred_vmsd_severity" in df.columns:
            created.append(self.plot_count(df, "pred_vmsd_severity", "VMSD Severity Distribution", "vmsd_severity_distribution.png"))
        if "pred_evidence_source" in df.columns:
            created.append(self.plot_count(df, "pred_evidence_source", "Evidence Source Distribution", "evidence_source_distribution.png"))
        if "heritage_site" in df.columns and "pred_vmsd_label" in df.columns:
            created.append(self.plot_sitewise_rate(df))
        if "pred_operational_aspects" in df.columns:
            created.append(self.plot_operational_aspects(df))
        return created

    def plot_count(self, df: pd.DataFrame, column: str, title: str, filename: str) -> Path:
        counts = df[column].fillna("Unknown").value_counts()
        path = self.output_dir / filename

        plt.figure(figsize=(8, 5))
        counts.plot(kind="bar")
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_sitewise_rate(self, df: pd.DataFrame) -> Path:
        temp = df.copy()
        temp["is_vmsd"] = temp["pred_vmsd_label"].astype(str).str.lower().eq("yes")
        site_rate = temp.groupby("heritage_site")["is_vmsd"].mean().sort_values(ascending=False)
        path = self.output_dir / "sitewise_vmsd_rate.png"

        plt.figure(figsize=(9, 5))
        site_rate.plot(kind="bar")
        plt.title("Site-wise VMSD Rate")
        plt.xlabel("Heritage Site")
        plt.ylabel("VMSD Rate")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_operational_aspects(self, df: pd.DataFrame) -> Path:
        aspect_counts = {}
        for value in df["pred_operational_aspects"].fillna(""):
            aspects = [a.strip() for a in str(value).split(";") if a.strip()]
            for aspect in aspects:
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1

        counts = pd.Series(aspect_counts).sort_values(ascending=False).head(15)
        path = self.output_dir / "operational_aspect_frequency.png"

        plt.figure(figsize=(10, 6))
        counts.plot(kind="bar")
        plt.title("Top Operational-Management Aspects")
        plt.xlabel("Operational Aspect")
        plt.ylabel("Frequency")
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        return path
