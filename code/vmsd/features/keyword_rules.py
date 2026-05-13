"""Reusable keyword matching rules."""

from typing import Dict, Iterable, List, Tuple

from vmsd.preprocessing.text_cleaner import TextCleaner


class KeywordMatcher:
    """Matches keywords and returns categories/terms in an explainable way."""

    def __init__(self):
        self.cleaner = TextCleaner()

    def match_terms(self, text: str, terms: Iterable[str]) -> List[str]:
        clean_text = self.cleaner.clean(text)
        matched = []
        for term in terms:
            clean_term = self.cleaner.clean(term)
            if clean_term and clean_term in clean_text:
                matched.append(term)
        return matched

    def match_categories(self, text: str, category_keywords: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
        matched_categories: List[str] = []
        matched_terms: List[str] = []

        for category, keywords in category_keywords.items():
            terms = self.match_terms(text, keywords)
            if terms:
                matched_categories.append(category)
                matched_terms.extend(terms)

        return matched_categories, matched_terms
