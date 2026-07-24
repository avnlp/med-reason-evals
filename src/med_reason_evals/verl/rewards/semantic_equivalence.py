"""Verl reward function for normalized medical text matching.

This module provides reward computation for tasks where the answer
needs to be compared using normalized text matching (e.g., medical diagnosis
comparison).

Unlike exact string matching, this module normalizes answers before comparison
using medical text normalization (case-folding, punctuation removal, etc.).
The comparison is exact/substring matching after normalization, not semantic
similarity via embeddings.

Two matching strategies are supported:
    - exact: Answers must match exactly after normalization.
    - substring: The normalized answer must appear within the prediction.

Example:
    >>> ground_truth = {"target": "Acute Myocardial Infarction"}
    >>> compute_score("The diagnosis is <answer>AMI</answer>", ground_truth)
    0.0  # Extracts "AMI", which doesn't match after normalization
"""

import re
from typing import Any

from med_reason_evals.utils.extraction import extract_answer
from med_reason_evals.utils.text import normalize_answer


def exact_match(prediction: str, golden_answers: str | list[str]) -> bool:
    """Check if prediction exactly matches any golden answer.

    Args:
        prediction: The predicted answer.
        golden_answers: The ground truth answer(s).

    Returns:
        True if there's an exact match after normalization.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction, mode="semantic")
    if not normalized_prediction:
        return False

    for golden in golden_answers:
        if normalize_answer(golden, mode="semantic") == normalized_prediction:
            return True

    return False


def substring_match(prediction: str, golden_answers: str | list[str]) -> bool:
    """Check if any golden answer is a substring of the prediction.

    Args:
        prediction: The predicted answer.
        golden_answers: The ground truth answer(s).

    Returns:
        True if any golden answer is found in the prediction.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction, mode="semantic")
    if not normalized_prediction:
        return False

    for golden in golden_answers:
        normalized_golden = normalize_answer(golden, mode="semantic")
        if not normalized_golden:
            continue
        pattern = re.compile(
            rf"\b{re.escape(normalized_golden)}\b",
            re.IGNORECASE,
        )
        if pattern.search(normalized_prediction):
            return True

    return False


def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    method: str = "exact",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Compute the reward score for semantic equivalence.

    Args:
        solution_str: The model's solution text.
        ground_truth: Dict with 'target' or 'answer' key containing the ground truth.
        method: Matching method - 'exact' or 'substring'.
        format_score: Score to return if answer format is correct but wrong.
        score: Score to return for a correct answer.

    Returns:
        The reward score (0.0 to 1.0).

    Raises:
        ValueError: If an unknown method is specified.
    """
    if method not in ("exact", "substring"):
        raise ValueError(f"Unknown method: {method}. Expected 'exact' or 'substring'.")

    answer = extract_answer(solution_str)
    if answer is None:
        return 0.0

    # Support both 'target' and 'answer' keys
    golden = ground_truth.get("target") or ground_truth.get("answer")
    if not golden:
        return 0.0

    if method == "exact":
        if exact_match(answer, golden):
            return score
    elif method == "substring" and substring_match(answer, golden):
        return score

    # Answer was extracted but incorrect
    return format_score
