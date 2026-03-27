"""Text processing utilities."""

from __future__ import annotations

import re
import unicodedata


def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip."""
    return re.sub(r"\s+", " ", text).strip()


def normalise_unicode(text: str) -> str:
    """Normalise Unicode to NFC form and replace common smart-quote variants."""
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2026": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def clean_text(text: str) -> str:
    """Full cleaning pipeline: unicode normalisation + whitespace collapse."""
    return normalise_whitespace(normalise_unicode(text))


def truncate(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to *max_length* characters, appending *suffix* if cut."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def count_tokens_approx(text: str) -> int:
    """Rough token count (whitespace split). Good enough for display purposes."""
    return len(text.split())
