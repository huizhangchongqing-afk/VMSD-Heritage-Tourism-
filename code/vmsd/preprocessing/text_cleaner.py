"""Text preprocessing for review text and image descriptions."""

import re


class TextCleaner:
    """Simple deterministic cleaner.

    We avoid heavy NLP dependencies so the project runs easily in Colab.
    """

    _url_pattern = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
    _html_pattern = re.compile(r"<.*?>")
    _space_pattern = re.compile(r"\s+")

    def clean(self, text: object) -> str:
        if text is None:
            return ""
        text = str(text)
        text = self._url_pattern.sub(" ", text)
        text = self._html_pattern.sub(" ", text)
        text = text.replace("\n", " ").replace("\t", " ")
        text = self._space_pattern.sub(" ", text)
        return text.strip().lower()

    def contains_phrase(self, text: str, phrase: str) -> bool:
        """Phrase search with word boundaries where possible."""
        text = self.clean(text)
        phrase = self.clean(phrase)
        if not phrase:
            return False
        return phrase in text
