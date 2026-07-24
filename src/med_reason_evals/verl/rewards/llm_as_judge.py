"""Verl reward function using LLM as judge.

This module provides reward computation using an LLM judge to evaluate
whether model responses are correct or equivalent to ground truth.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

from med_reason_evals.utils.extraction import extract_answer
from med_reason_evals.utils.parsing import parse_yes_no
from med_reason_evals.utils.retry import call_with_retry


DEFAULT_JUDGE_PROMPT = """Is the predicted answer correct (yes/no)?
Predicted answer: {prediction}
Correct answer: {ground_truth}
Answer [yes/no]."""


async def judge_answer(
    prediction: str,
    ground_truth: str,
    judge_client: AsyncOpenAI,
    judge_model: str = "openai/gpt-oss-120b",
    judge_prompt: str | None = None,
    max_tokens: int = 100,
    temperature: float = 0.0,
    **kwargs: Any,
) -> bool:
    """Use an LLM to judge if the prediction is correct.

    Args:
        prediction: The model's prediction.
        ground_truth: The correct answer.
        judge_client: The OpenAI client to use for judging.
        judge_model: The model to use for judging.
        judge_prompt: Custom template with {prediction} and {ground_truth} placeholders.
        max_tokens: Maximum tokens for judge response.
        temperature: Sampling temperature for judge.
        **kwargs: Additional arguments passed to the API call.

    Returns:
        True if the judge says the answer is correct.
    """
    prompt = (judge_prompt or DEFAULT_JUDGE_PROMPT).format(
        prediction=prediction,
        ground_truth=ground_truth,
    )

    response = await call_with_retry(
        judge_client.chat.completions.create,
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

    judge_response = response.choices[0].message.content or ""
    result = parse_yes_no(judge_response)
    return result if result is not None else False


async def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    judge_client: AsyncOpenAI,
    judge_model: str = "openai/gpt-oss-120b",
    judge_prompt: str | None = None,
    format_score: float = 0.0,
    score: float = 1.0,
    max_tokens: int = 100,
    temperature: float = 0.0,
    **kwargs: Any,
) -> float:
    """Compute the reward score using an LLM judge.

    This async function extracts the answer from the completion, uses an LLM
    to judge correctness, and returns a score.

    Args:
        solution_str: The model's solution text.
        ground_truth: Dict with 'target' or 'answer' key containing ground truth.
        judge_client: The OpenAI client to use for judging.
        judge_model: The model to use for judging.
        judge_prompt: Custom prompt template.
        format_score: Score if answer format is correct but wrong (0.0-1.0).
        score: Score to return for a correct answer (0.0-1.0).
        max_tokens: Maximum tokens for judge response.
        temperature: Sampling temperature for judge.
        **kwargs: Additional arguments passed to the API call.

    Returns:
        The reward score (0.0 to 1.0).
    """
    answer = extract_answer(solution_str)
    if answer is None:
        return 0.0

    # Get ground truth
    golden = ground_truth.get("target") or ground_truth.get("answer")
    if not golden:
        return format_score
    if isinstance(golden, list):
        golden = golden[0] if golden else ""

    is_correct = await judge_answer(
        prediction=answer,
        ground_truth=golden,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=judge_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

    return score if is_correct else format_score


def make_judge_reward(
    template: str,
    extract_answer_fn: Callable[[str], str | None] | None = None,
) -> Callable[..., Awaitable[float]]:
    """Create a judge-based reward function with a baked-in template.

    This factory creates a reward function that uses a specific judge prompt
    template. Useful for datasets with specific evaluation criteria.

    Args:
        template: The judge template with {prediction} and {ground_truth} placeholders.
        extract_answer_fn: Optional custom function to extract answers from completions.
                          Uses default extraction if None.

    Returns:
        An async reward function with the template baked in.
    """
    extractor = extract_answer_fn or extract_answer

    async def reward_fn(
        solution_str: str,
        ground_truth: dict[str, Any],
        judge_client: AsyncOpenAI,
        judge_model: str = "openai/gpt-oss-120b",
        format_score: float = 0.0,
        score: float = 1.0,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> float:
        """Compute score using the baked-in template."""
        answer = extractor(solution_str)
        if answer is None:
            return 0.0

        golden = ground_truth.get("target") or ground_truth.get("answer")
        if not golden:
            return format_score
        if isinstance(golden, list):
            golden = golden[0] if golden else ""

        is_correct = await judge_answer(
            prediction=answer,
            ground_truth=golden,
            judge_client=judge_client,
            judge_model=judge_model,
            judge_prompt=template,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        return score if is_correct else format_score

    return reward_fn
