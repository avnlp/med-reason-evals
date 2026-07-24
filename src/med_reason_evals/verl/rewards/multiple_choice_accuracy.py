"""Verl reward function for multiple-choice accuracy.

This module provides reward computation for RL training on MCQ tasks.
It extracts answers from model completions and compares them to ground truth.

Answer extraction:
    - Uses regex to find answers in <answer>...</answer> tags
    - Falls back to extracting the last uppercase letter or digit

Comparison strategy:
    - Normalizes both extracted and correct answers (MCQ mode)
    - Supports letter matching (A, B, C, D) and numeric matching (1, 2, 3)
    - Optionally checks against full answer text for verbose responses

Scoring:
    - Returns 0.0 if no answer is extracted
    - Returns score parameter (default 1.0) for correct answers
    - Returns format_score (default 0.0) for extracted but incorrect answers
"""

from typing import Any

from med_reason_evals.utils.extraction import extract_answer
from med_reason_evals.utils.text import (
    nfkc_casefold,
    normalize_answer,
    normalize_spaces,
)


def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Compute the reward score for a multiple-choice answer.

    Args:
        solution_str: The model's solution text.
        ground_truth: Dict with 'answer' key containing the correct answer letter,
            and optionally 'answer_text' for the full answer text.
        format_score: Score to return if answer format is correct but wrong.
        score: Score to return for a correct answer.

    Returns:
        The reward score (0.0 to 1.0).
    """
    answer = extract_answer(solution_str)
    if answer is None:
        return 0.0

    correct_answer = ground_truth.get("answer", "")
    answer_text = ground_truth.get("answer_text", "")

    # Normalize for comparison
    norm_answer = normalize_answer(answer, mode="mcq")
    norm_correct = normalize_answer(correct_answer, mode="mcq")

    # Check letter/number match
    if norm_answer and norm_correct and norm_answer == norm_correct:
        return score

    # Check answer text match (for verbose responses)
    if answer_text:
        norm_text = nfkc_casefold(normalize_spaces(answer_text))
        norm_extracted = nfkc_casefold(normalize_spaces(answer))
        # Substring match for answer text
        if (
            norm_text
            and norm_extracted
            and (norm_text in norm_extracted or norm_extracted in norm_text)
        ):
            return score

    # Answer was extracted but incorrect
    if norm_answer:
        return format_score

    return 0.0
