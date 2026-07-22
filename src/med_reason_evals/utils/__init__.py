"""Text processing utilities for evaluation pipelines and reward functions.

These helpers bridge the gap between messy model outputs and structured evaluation.
The design prioritizes composability: each function does one thing well, making
them easy to chain or swap based on dataset requirements.

Key modules:
    text: Unicode-aware normalization for fuzzy string matching
    extraction: Heuristic-based answer extraction from free-form responses
    parsing: JSON recovery and boolean parsing from judge model outputs

All functions are pure (no side effects) and operate only on strings, making
them safe to use in multi-threaded evaluation pipelines.
"""

from med_reason_evals.utils.extraction import (
    extract_answer,
    extract_boxed_answer,
    extract_xml_answer,
)
from med_reason_evals.utils.parsing import (
    parse_json_response,
    parse_yes_no,
    strip_think_tags,
)
from med_reason_evals.utils.text import (
    nfkc_casefold,
    normalize_answer,
    normalize_spaces,
)


normalize_text = normalize_answer

__all__ = [
    "nfkc_casefold",
    "normalize_spaces",
    "normalize_answer",
    "normalize_text",
    "extract_xml_answer",
    "extract_boxed_answer",
    "extract_answer",
    "strip_think_tags",
    "parse_json_response",
    "parse_yes_no",
]
