"""Hybrid reward function for PubHealthBench MCQ/freeform scoring.

This module implements a dispatch-based reward function that selects the
appropriate scoring strategy based on question type. PubHealthBench contains
heterogeneous questions requiring different evaluation approaches.

Dispatch logic:
- is_mcq=True: Use accuracy_reward for standard letter/number matching
- is_mcq=False, None, or missing (default): Use
  binary_judge_reward_from_template for open-ended evaluation

This pattern allows a single reward function to handle mixed-format datasets
without requiring separate evaluator configurations. The question type is
passed through the info dictionary from dataset loading. Defaulting to
free-form routing is deliberate: free-form ground truths routed into MCQ
letter-extraction would raise or silently score incorrectly, whereas MCQ
answers routed into judge scoring merely take the slower, still-correct path.
"""

from __future__ import annotations

from typing import Any

from med_reason_evals.verifiers.rewards.llm_as_judge import (
    PUBHEALTHBENCH_JUDGE_TEMPLATE,
    binary_judge_reward_from_template,
)
from med_reason_evals.verifiers.rewards.multiple_choice_accuracy import (
    accuracy_reward,
)


async def pubhealthbench_reward(
    completion: Any,
    answer: str,
    info: dict[str, Any] | None,
    state,
    parser,
    judge,
    **kwargs: Any,
) -> float:
    """Score PubHealthBench answers using MCQ or judge rubric.

    Routes to appropriate reward function based on info["is_mcq"] flag.
    Falls back to free-form judge scoring if info is missing or is_mcq is not
    specified, since that is the safer default: a free-form ground truth
    mistakenly routed to MCQ letter-extraction can raise or score incorrectly,
    while an MCQ answer routed to the judge is still scored correctly.

    Args:
        completion: Model's completion (Messages or string)
        answer: Ground truth answer (letter for MCQ, text for free-form)
        info: Metadata dict containing is_mcq flag and other fields
        state: Rollout state from verifiers
        parser: Parser for extracting answer from completion
        judge: Judge function for LLM-as-judge scoring
        **kwargs: Additional arguments passed to underlying reward functions

    Returns:
        Float score 0.0-1.0 from selected reward function

    Raises:
        ValueError: If info["is_mcq"] is present but not a bool.
    """
    is_mcq = info.get("is_mcq") if info else None
    if is_mcq is not None and not isinstance(is_mcq, bool):
        raise ValueError(
            "info['is_mcq'] must be a bool or absent/None, got "
            f"{type(is_mcq).__name__}: {is_mcq!r}"
        )

    if is_mcq:
        return await accuracy_reward(
            completion=completion,
            answer=answer,
            info=info,
            parser=parser,
            **kwargs,
        )

    return await binary_judge_reward_from_template(
        completion=completion,
        answer=answer,
        state=state,
        info=info,
        parser=parser,
        judge=judge,
        template=PUBHEALTHBENCH_JUDGE_TEMPLATE,
        **kwargs,
    )
