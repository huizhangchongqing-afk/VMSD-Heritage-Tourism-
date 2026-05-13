"""Cleaner dedicated to image-description evidence.

In this project, images are represented through manually/externally prepared
image descriptions. The image cleaner therefore treats image evidence as text.
"""

from vmsd.preprocessing.text_cleaner import TextCleaner


class ImageDescriptionCleaner(TextCleaner):
    """Currently identical to TextCleaner, separated for future extension."""

    pass
