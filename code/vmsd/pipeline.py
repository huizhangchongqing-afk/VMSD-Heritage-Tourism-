"""End-to-end VMSD pipeline.

This class wires together loading, validation, cleaning, feature extraction,
scoring, and output generation.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from vmsd.config import ConfigManager
from vmsd.core.constants import NO_VALUES, YES_VALUES
from vmsd.core.entities import ReviewRecord
from vmsd.core.taxonomy import Taxonomy
from vmsd.data.loader import ReviewDataLoader
from vmsd.data.validator import DatasetValidator
from vmsd.features.image_feature_extractor import ImageFeatureExtractor
from vmsd.features.text_feature_extractor import TextFeatureExtractor
from vmsd.scoring.vmsd_scorer import VMSDScorer
from vmsd.utils.logging_utils import get_logger
from vmsd.utils.path_utils import project_root


class VMSDPipeline:
    """Main OOP pipeline for VMSD prediction."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        taxonomy_path: str | Path | None = None,
    ):
        root = project_root()
        self.config = ConfigManager(config_path or root / "config" / "config.yaml")
        self.taxonomy = Taxonomy(taxonomy_path or root / "config" / "label_taxonomy.yaml")
        self.logger = get_logger(self.__class__.__name__)

        self.columns = self.config.columns
        scoring_cfg = self.config.get("scoring", {})
        dataset_cfg = self.config.get("dataset", {})
        extractor_cfg = {**scoring_cfg, **dataset_cfg}

        self.loader = ReviewDataLoader()
        self.validator = DatasetValidator(self.columns)
        self.text_extractor = TextFeatureExtractor(
            intrinsic_keywords=self.taxonomy.intrinsic_value_keywords,
            operational_keywords=self.taxonomy.all_operational_keywords(),
            config=extractor_cfg,
        )
        self.image_extractor = ImageFeatureExtractor(
            operational_keywords=self.taxonomy.all_operational_keywords(),
            image_keywords=self.taxonomy.image_evidence_keywords,
            config=extractor_cfg,
        )
        self.scorer = VMSDScorer(
            text_extractor=self.text_extractor,
            image_extractor=self.image_extractor,
            scoring_config=scoring_cfg,
        )

    def run(self, input_path: str | Path) -> pd.DataFrame:
        """Run the full pipeline and return dataframe with prediction columns."""
        self.logger.info(f"Loading dataset: {input_path}")
        df = self.loader.load(input_path)
        df = self.validator.validate_and_repair(df)

        self.logger.info(f"Rows loaded: {len(df)}")
        prediction_rows = []

        for _, row in df.iterrows():
            review = self._row_to_review_record(row)
            result = self.scorer.score(review)
            prediction_rows.append(self._result_to_dict(result))

        predictions = pd.DataFrame(prediction_rows)
        output = pd.concat([df.reset_index(drop=True), predictions], axis=1)
        self.logger.info("Pipeline completed successfully.")
        return output

    def _row_to_review_record(self, row: pd.Series) -> ReviewRecord:
        c = self.columns
        return ReviewRecord(
            review_id=str(row.get(c.get("id", "review_id"), "")),
            heritage_site=str(row.get(c.get("site", "heritage_site"), "")),
            rating=float(row.get(c.get("rating", "review_rating"), 0) or 0),
            review_text=str(row.get(c.get("text", "review_text"), "") or ""),
            image_description=str(row.get(c.get("image_description", "image_description"), "") or ""),
            manual_intrinsic_value=self._parse_bool_or_none(row.get(c.get("intrinsic_value_positive", "intrinsic_value_positive"))),
            manual_text_om_issue=self._parse_bool_or_none(row.get(c.get("text_operational_issue", "text_operational_issue"))),
            manual_image_om_issue=self._parse_bool_or_none(row.get(c.get("image_operational_issue", "image_operational_issue"))),
            manual_vmsd_label=self._parse_label_or_none(row.get(c.get("vmsd_label", "vmsd_label"))),
            extra_fields=row.to_dict(),
        )

    @staticmethod
    def _parse_bool_or_none(value: object) -> Optional[bool]:
        if pd.isna(value):
            return None
        text = str(value).strip().lower()
        if text in YES_VALUES:
            return True
        if text in NO_VALUES:
            return False
        return None

    @staticmethod
    def _parse_label_or_none(value: object) -> Optional[str]:
        if pd.isna(value):
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _result_to_dict(result) -> Dict[str, object]:
        return {
            "pred_intrinsic_value": result.intrinsic_value,
            "pred_text_om_issue": result.text_om_issue,
            "pred_image_om_issue": result.image_om_issue,
            "pred_text_confidence": result.text_confidence,
            "pred_image_confidence": result.image_confidence,
            "pred_fusion_score": result.fusion_score,
            "pred_vmsd_label": result.vmsd_label,
            "pred_vmsd_severity": result.severity,
            "pred_operational_aspects": "; ".join(result.operational_aspects),
            "pred_evidence_source": result.evidence_source,
            "pred_evidence_terms": "; ".join(result.evidence_terms),
        }
