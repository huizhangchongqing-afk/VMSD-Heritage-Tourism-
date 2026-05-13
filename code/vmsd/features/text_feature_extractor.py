"""Feature extraction from review text."""

from typing import List

from vmsd.core.entities import FeatureResult
from vmsd.features.keyword_rules import KeywordMatcher
from vmsd.preprocessing.text_cleaner import TextCleaner


class TextFeatureExtractor:
    """Extracts IV and OM features from review text."""

    def __init__(self, intrinsic_keywords: List[str], operational_keywords: dict, config: dict):
        self.intrinsic_keywords = intrinsic_keywords
        self.operational_keywords = operational_keywords
        self.config = config
        self.matcher = KeywordMatcher()
        self.cleaner = TextCleaner()

    def has_intrinsic_value(self, text: str, rating: float, manual_flag: bool | None = None) -> bool:
        """Detect whether a review expresses intrinsic heritage value."""
        high_rating_threshold = self.config.get("high_rating_threshold", 4)

        if manual_flag is True:
            return True
        if manual_flag is False:
            return False
        if rating >= high_rating_threshold:
            return True

        matched_terms = self.matcher.match_terms(text, self.intrinsic_keywords)
        return bool(matched_terms)

    def extract_operational_issue(self, text: str, manual_flag: bool | None = None) -> FeatureResult:
        """Detect operational-management issue in text."""
        matched_categories, matched_terms = self.matcher.match_categories(text, self.operational_keywords)

        if manual_flag is True:
            # Manual annotation is treated as high-confidence evidence.
            confidence = self.config.get("manual_flag_confidence", 0.95)
            return FeatureResult(True, confidence, matched_categories, matched_terms)
        if manual_flag is False:
            # An explicit negative annotation is respected over keyword guesses.
            return FeatureResult(False, 0.0, [], [])

        confidence = self._confidence_from_matches(matched_terms)
        has_issue = confidence >= self.config.get("min_keyword_confidence", 0.25)

        return FeatureResult(has_issue, confidence, matched_categories, matched_terms)

    def _confidence_from_matches(self, matched_terms: List[str]) -> float:
        """Convert number of matched terms into a bounded confidence score."""
        if not matched_terms:
            return 0.0
        max_conf = self.config.get("max_keyword_confidence", 0.90)
        return min(max_conf, 0.20 + 0.15 * len(set(matched_terms)))
