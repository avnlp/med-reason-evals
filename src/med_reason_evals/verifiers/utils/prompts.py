"""System prompts and answer format definitions for medical reasoning evaluation.

This module defines standardized prompts for instructing models on how to format
their responses. Two main formats are supported:

1. XML Format: Uses <answer>...</answer> tags for structured extraction.
   Compatible with vf.XMLParser for automatic field extraction.

2. BOXED Format: Uses LaTeX \\boxed{} notation (math-style answers).
   Common in reasoning benchmarks and works well with chain-of-thought.

Each format has two variants:
- Standard: Direct answer without showing reasoning
- THINK variant: Requires reasoning in <think>...</think> tags first

The AnswerFormat enum provides type-safe selection of formats across
the codebase, ensuring consistency in prompt/parser pairing.
"""

from enum import Enum


class AnswerFormat(str, Enum):
    """Supported answer formats for model responses.

    Values:
        XML: Structured XML tags (<answer>...</answer>)
        BOXED: LaTeX boxed notation (\\boxed{answer})
        JSON: JSON format (defined but less commonly used)

    Note: XML and BOXED are the primary formats used in evaluators.
    """

    BOXED = "boxed"
    JSON = "json"
    XML = "xml"


# XML format prompts - Used with vf.XMLParser for structured extraction
# The <answer> tag content is automatically extracted as the parsed answer
THINK_XML_SYSTEM_PROMPT = (
    "Think step-by-step inside <think>...</think> tags. "
    "Then, give your final answer inside <answer>...</answer> XML tags."
)

XML_SYSTEM_PROMPT = (
    "Please reason step by step, then give your final answer within "
    "<answer>...</answer> XML tags."
)

# Boxed format prompts - LaTeX style common in math/reasoning benchmarks
# The \\boxed{} content is extracted using extract_boxed_answer()
THINK_BOXED_SYSTEM_PROMPT = (
    "Think step-by-step inside <think>...</think> tags. "
    "Then, give your final answer inside \\boxed{}."
)

BOXED_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
