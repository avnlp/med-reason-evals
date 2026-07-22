"""Text normalization helpers for comparing model answers.

Normalization is intentionally opinionated to make evaluation resilient to
formatting noise. The utilities support multiple modes so datasets can opt into
lightweight normalization for MCQ or more aggressive semantic matching.
"""

import re
import string
import unicodedata


def nfkc_casefold(text: str | None) -> str:
    """Apply Unicode NFKC normalization and case folding.

    NFKC is chosen over NFC for evaluation because it decomposes compatibility
    characters (e.g., converting full-width letters to ASCII equivalents),
    which models sometimes emit inconsistently. Case folding handles locale-
    insensitive comparison better than lower() for non-ASCII text.

    Args:
        text: The input text to normalize.

    Returns:
        The normalized, case-folded text.
    """
    # NFKC ensures 'Ａ' (full-width) and 'A' match; casefold handles 'ß' vs 'ss'
    return unicodedata.normalize("NFKC", text or "").casefold()


def normalize_spaces(text: str) -> str:
    """Collapse multiple whitespace to single spaces and trim edges.

    Args:
        text: The input text to normalize.

    Returns:
        The space-normalized text.
    """
    return re.sub(r"\s+", " ", text).strip()


def normalize_answer(text: str | None, mode: str = "basic") -> str:
    """Normalize an answer string for comparison.

    Args:
        text: The answer text to normalize.
        mode: The normalization mode to use ("basic", "semantic", "mcq").
            Defaults to "basic".

    Returns:
        The normalized text.

    Raises:
        ValueError: If an unknown mode is specified.
    """
    if text is None:
        if mode not in {"basic", "semantic", "mcq"}:
            raise ValueError(f"Unknown normalization mode: {mode}")
        return ""

    if mode == "mcq":
        # Treat single-character MCQ outputs as categorical labels.
        stripped = unicodedata.normalize("NFKC", text).strip()
        if len(stripped) == 1:
            if stripped.isalpha():
                return stripped.upper()
            return stripped
        # Multi-character text falls through to basic normalization

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if mode in {"basic", "mcq"}:
        return normalized

    if mode == "semantic":
        normalized = re.sub(r"\b(a|an|the)\b", "", normalized)
        normalized = "".join(char for char in normalized if not unicodedata.category(char).startswith("P"))
        return re.sub(r"\s+", " ", normalized).strip()

    raise ValueError(f"Unknown normalization mode: {mode}")
