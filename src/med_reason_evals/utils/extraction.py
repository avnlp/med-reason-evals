"""Extract final answers from model responses using layered heuristics.

The utilities in this module are tuned for evaluation pipelines that accept
loosely structured outputs. They favor explicit formatting cues such as XML
tags and LaTeX boxes before falling back to anchored phrases or the last token,
which mirrors how models often summarize their final choice at the end.
"""

import re


_BOXED_PATTERN = re.compile(
    # Matches \boxed{...} with balanced braces. The complex inner pattern handles
    # nested braces that models sometimes emit (e.g., \boxed{\frac{1}{2}}).
    r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
    re.DOTALL,
)

_ANSWER_PATTERN = re.compile(
    r"(?:\bfinal\s+answer\b|\banswer\b|\bans\b|\bchoice\b|\boption\b"
    r"|\bselected\b|\bi\s+choose\b|\bi\s+pick\b|\btherefore\b|\bthus\b"
    r"|\bso\b|\bconclusion\b|\bin\s+conclusion\b|\bmost\s+likely\b"
    r"|\bbest[-\s]+supported\s+answer\b)\s*[:\-–—]?\s*(?:is\s*)?"
    r"\(?\s*([A-Za-z]|\d{1,2})\s*\)?\s*[\)\.:\-]?",
    re.IGNORECASE,
)


_FREE_FORM_PATTERN = re.compile(
    r"(?:\bfinal\s+answer\b|\banswer\b|\bans\b|\bdiagnosis\s+is\b"
    r"|\bconclusion\b|\bin\s+conclusion\b|\bmost\s+likely\b"
    r"|\bbest[-\s]+supported\s+answer\b)\s*[:\-–—]?\s*(?:is\s*)?"
    r"\s*(?:[*_`~]+\s*)*"
    r"([A-Za-z0-9][^\n.!?]*?)"
    r"\s*(?:[*_`~]+\s*)*(?:[.!?]|\n|$)",
    re.IGNORECASE,
)

_TOKEN_PATTERN = re.compile(
    r"(?:\(|\b)([A-Za-z]|\d{1,2})(?:\)|\b)[\s.!?]*$",
)


def extract_boxed_answer(text: str) -> str:
    """Extract the content from the final LaTeX ``\\boxed{...}`` expression.

    Models that show their work sometimes include intermediate ``\\boxed``
    fragments. Selecting the final boxed span biases toward the model's last
    declared answer without requiring stricter output formatting.

    Args:
        text: The model's response text containing ``\\boxed{...}``.

    Returns:
        The content inside the boxed expression, or the original text if no
        boxed expression is found.
    """
    if not text:
        return text

    matches = list(_BOXED_PATTERN.finditer(text))
    if not matches:
        return text

    # Choose the last boxed span to avoid earlier scratch-work boxes.
    return matches[-1].group(1).strip()


def extract_xml_answer(text: str, field: str = "answer") -> str | None:
    """Extract content from a named XML tag.

    Args:
        text: The model's response text containing XML tags.
        field: The name of the XML field to extract (default: "answer").

    Returns:
        The content inside the XML tag, or None if not found.
    """
    if not text:
        return None

    pattern = re.compile(
        rf"<{re.escape(field)}>\s*(.*?)\s*</{re.escape(field)}>",
        re.DOTALL | re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None

    # Favor the last tag to mirror models that emit multiple <answer> tags.
    return matches[-1].group(1).strip()


def extract_answer(text: str | None) -> str | None:  # noqa: PLR0911
    """Extract a final answer from a model response using ordered heuristics.

    The strategy intentionally tries highly structured markers first to avoid
    misclassifying intermediate reasoning fragments as the final answer.

    Args:
        text: The model's response text.

    Returns:
        The extracted answer, or None if no answer is found.
    """
    if not text:
        return None

    search_text = text[-500:] if len(text) > 500 else text

    # Strategy 1: XML tags (highest priority).
    xml_result = extract_xml_answer(search_text, field="answer")
    if xml_result is not None:
        return xml_result

    # Strategy 2: Boxed LaTeX.
    boxed_result = extract_boxed_answer(search_text)
    if boxed_result != search_text:
        return boxed_result

    # Strategy 3: Anchored patterns (captures single letter/digit).
    anchored_matches = list(_ANSWER_PATTERN.finditer(search_text))
    if anchored_matches:
        return anchored_matches[-1].group(1).strip()

    # Strategy 3b: Free-form text after anchor phrases.
    anchored_freeform = list(_FREE_FORM_PATTERN.finditer(search_text))
    if anchored_freeform:
        return anchored_freeform[-1].group(1).strip()

    # Strategy 4: Last token (lowest priority).
    tokens = list(_TOKEN_PATTERN.finditer(search_text))
    if tokens:
        return tokens[-1].group(1).strip()

    return None
