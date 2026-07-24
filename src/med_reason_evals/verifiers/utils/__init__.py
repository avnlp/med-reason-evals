"""Utility functions for verifiers.

This module provides parsing and formatting utilities for extracting
answers from model responses and managing answer format specifications.

The utilities are organized into:

1. Prompts (prompts.py): System prompt templates for different answer formats
   (XML tags, LaTeX boxed notation) with optional chain-of-thought sections.

2. Parsers (parsers.py): Answer extraction functions that parse model outputs
   from various formats (XML tags, boxed LaTeX, think tags).

These utilities are used by both evaluators (for setting up the environment)
and reward functions (for extracting answers during scoring).
"""

from med_reason_evals.verifiers.utils.parsers import (
    extract_boxed_answer,
    extract_xml_answer,
    strip_think_tags,
)
from med_reason_evals.verifiers.utils.prompts import (
    BOXED_SYSTEM_PROMPT,
    THINK_BOXED_SYSTEM_PROMPT,
    THINK_XML_SYSTEM_PROMPT,
    XML_SYSTEM_PROMPT,
    AnswerFormat,
)


__all__ = [
    "AnswerFormat",
    "BOXED_SYSTEM_PROMPT",
    "THINK_BOXED_SYSTEM_PROMPT",
    "THINK_XML_SYSTEM_PROMPT",
    "XML_SYSTEM_PROMPT",
    "extract_boxed_answer",
    "extract_xml_answer",
    "strip_think_tags",
]
