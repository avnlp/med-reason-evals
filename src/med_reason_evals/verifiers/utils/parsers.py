"""Parsing utilities for extracting answers from model responses.

This module re-exports extraction functions from the med_reason_evals.utils
package, providing a unified interface for verifiers-specific parsing needs.

These parsers handle the answer extraction phase of evaluation:
- extract_boxed_answer: Parse \\boxed{} LaTeX notation
- extract_xml_answer: Parse <answer>...</answer> XML tags
- strip_think_tags: Remove <think>...</think> reasoning sections
- normalize_answer: Normalize text for fuzzy matching

The functions are used by:
1. vf.Parser/vf.XMLParser: Integrated into verifiers framework
2. Reward functions: Manual extraction for custom scoring logic
3. Evaluators: Setting up parser configurations

Note: This is a compatibility layer. Direct imports from med_reason_evals.utils
are also valid and may be preferred for non-verifiers code.
"""

from med_reason_evals.utils.extraction import (
    extract_boxed_answer,
    extract_xml_answer,
)
from med_reason_evals.utils.parsing import strip_think_tags
from med_reason_evals.utils.text import normalize_answer


__all__ = [
    "extract_boxed_answer",
    "extract_xml_answer",
    "normalize_answer",
    "strip_think_tags",
]
