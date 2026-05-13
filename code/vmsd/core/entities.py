"""Data objects used across the VMSD project.

Using dataclasses keeps the project clean: raw rows are converted into
structured review objects and scorer outputs are returned as structured results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ReviewRecord:
    """One review record after basic loading and normalization."""

    review_id: str
    heritage_site: str
    rating: float
    review_text: str
    image_description: str = ""
    manual_intrinsic_value: Optional[bool] = None
    manual_text_om_issue: Optional[bool] = None
    manual_image_om_issue: Optional[bool] = None
    manual_vmsd_label: Optional[str] = None
    extra_fields: Dict[str, object] = field(default_factory=dict)


@dataclass
class FeatureResult:
    """Feature extraction result for one evidence channel."""

    has_issue: bool
    confidence: float
    matched_aspects: List[str]
    matched_terms: List[str]


@dataclass
class VMSDResult:
    """Final VMSD scoring result for one review."""

    intrinsic_value: bool
    text_om_issue: bool
    image_om_issue: bool
    text_confidence: float
    image_confidence: float
    fusion_score: float
    vmsd_label: str
    severity: str
    operational_aspects: List[str]
    evidence_source: str
    evidence_terms: List[str]
