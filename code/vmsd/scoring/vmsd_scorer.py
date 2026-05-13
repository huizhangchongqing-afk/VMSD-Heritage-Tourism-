"""VMSD scoring logic.

This file implements the formal rule used in the paper:

    VMSD = IV AND (OM_text OR OM_image)

and the fusion equation:

    D_F = 1 - (1 - D_T) * (1 - D_I)
"""

from vmsd.core.constants import (
    EVIDENCE_BOTH,
    EVIDENCE_IMAGE,
    EVIDENCE_NONE,
    EVIDENCE_TEXT,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MODERATE,
    SEVERITY_NONE,
    VMSD_NO,
    VMSD_YES,
)
from vmsd.core.entities import ReviewRecord, VMSDResult
from vmsd.features.image_feature_extractor import ImageFeatureExtractor
from vmsd.features.text_feature_extractor import TextFeatureExtractor


class VMSDScorer:
    """Scores one review at a time using IV, OM-text, and OM-image evidence."""

    def __init__(
        self,
        text_extractor: TextFeatureExtractor,
        image_extractor: ImageFeatureExtractor,
        scoring_config: dict,
    ):
        self.text_extractor = text_extractor
        self.image_extractor = image_extractor
        self.scoring_config = scoring_config

    def score(self, review: ReviewRecord) -> VMSDResult:
        intrinsic_value = self.text_extractor.has_intrinsic_value(
            text=review.review_text,
            rating=review.rating,
            manual_flag=review.manual_intrinsic_value,
        )

        text_result = self.text_extractor.extract_operational_issue(
            text=review.review_text,
            manual_flag=review.manual_text_om_issue,
        )
        image_result = self.image_extractor.extract_operational_issue(
            image_description=review.image_description,
            manual_flag=review.manual_image_om_issue,
        )

        fusion_score = self.fuse_confidence(
            text_confidence=text_result.confidence,
            image_confidence=image_result.confidence,
        )

        is_vmsd = intrinsic_value and (text_result.has_issue or image_result.has_issue)
        label = VMSD_YES if is_vmsd else VMSD_NO
        severity = self.assign_severity(fusion_score, is_vmsd)
        evidence_source = self.get_evidence_source(text_result.has_issue, image_result.has_issue)

        operational_aspects = sorted(
            set(text_result.matched_aspects + image_result.matched_aspects)
        )
        evidence_terms = sorted(set(text_result.matched_terms + image_result.matched_terms))

        return VMSDResult(
            intrinsic_value=intrinsic_value,
            text_om_issue=text_result.has_issue,
            image_om_issue=image_result.has_issue,
            text_confidence=round(text_result.confidence, 4),
            image_confidence=round(image_result.confidence, 4),
            fusion_score=round(fusion_score, 4),
            vmsd_label=label,
            severity=severity,
            operational_aspects=operational_aspects,
            evidence_source=evidence_source,
            evidence_terms=evidence_terms,
        )

    @staticmethod
    def fuse_confidence(text_confidence: float, image_confidence: float) -> float:
        """Noisy-OR fusion: D_F = 1 - (1-D_T)(1-D_I)."""
        text_confidence = max(0.0, min(1.0, text_confidence))
        image_confidence = max(0.0, min(1.0, image_confidence))
        return 1.0 - (1.0 - text_confidence) * (1.0 - image_confidence)

    def assign_severity(self, fusion_score: float, is_vmsd: bool) -> str:
        if not is_vmsd:
            return SEVERITY_NONE

        low_threshold = self.scoring_config.get("low_severity_threshold", 0.35)
        moderate_threshold = self.scoring_config.get("moderate_severity_threshold", 0.65)

        if fusion_score < low_threshold:
            return SEVERITY_LOW
        if fusion_score < moderate_threshold:
            return SEVERITY_MODERATE
        return SEVERITY_HIGH

    @staticmethod
    def get_evidence_source(text_issue: bool, image_issue: bool) -> str:
        if text_issue and image_issue:
            return EVIDENCE_BOTH
        if text_issue:
            return EVIDENCE_TEXT
        if image_issue:
            return EVIDENCE_IMAGE
        return EVIDENCE_NONE
