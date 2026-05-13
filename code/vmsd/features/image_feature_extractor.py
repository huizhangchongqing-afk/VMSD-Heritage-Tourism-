"""Feature extraction from image descriptions."""

from typing import List

from vmsd.core.entities import FeatureResult
from vmsd.features.keyword_rules import KeywordMatcher


class ImageFeatureExtractor:
    """Detects OM evidence from image descriptions.

    This supports the VMSD paper's image-description pathway, where image evidence
    can independently reveal operational issues missed by review text.
    """

    def __init__(self, operational_keywords: dict, image_keywords: List[str], config: dict):
        self.operational_keywords = operational_keywords
        self.image_keywords = image_keywords
        self.config = config
        self.matcher = KeywordMatcher()

    def extract_operational_issue(self, image_description: str, manual_flag: bool | None = None) -> FeatureResult:
        matched_categories, category_terms = self.matcher.match_categories(
            image_description,
            self.operational_keywords,
        )
        image_terms = self.matcher.match_terms(image_description, self.image_keywords)
        all_terms = list(dict.fromkeys(category_terms + image_terms))

        if manual_flag is True:
            confidence = self.config.get("manual_flag_confidence", 0.95)
            return FeatureResult(True, confidence, matched_categories, all_terms)
        if manual_flag is False:
            # An explicit negative annotation is respected over keyword guesses.
            return FeatureResult(False, 0.0, [], [])

        confidence = self._confidence_from_matches(all_terms)
        has_issue = confidence >= self.config.get("min_keyword_confidence", 0.25)
        return FeatureResult(has_issue, confidence, matched_categories, all_terms)

    def _confidence_from_matches(self, matched_terms: List[str]) -> float:
        if not matched_terms:
            return 0.0
        max_conf = self.config.get("max_keyword_confidence", 0.90)
        return min(max_conf, 0.25 + 0.15 * len(set(matched_terms)))
