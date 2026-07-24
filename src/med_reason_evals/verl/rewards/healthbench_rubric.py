"""Verl reward function for HealthBench rubric-based evaluation.

This module provides reward computation for HealthBench using multi-criteria
rubric evaluation with parallel judging.

HealthBench questions are evaluated against multiple criteria, each with an
associated point value (positive for correct, negative for errors). The final
score is normalized to [0, 1] based on the sum of earned points divided by
total positive points available.

Key features:
    - Parallel evaluation of all criteria using asyncio.Semaphore
    - Configurable concurrency via max_parallel_judges parameter
    - JSON-formatted judge responses with explanation and boolean result
    - Support for negative scoring (penalties for incorrect information)

Example criteria format:
    ground_truth = {
        "criteria": ["Provides accurate diagnosis", "Lists appropriate medications"],
        "points_list": [5, 3]
    }
"""

from __future__ import annotations

import asyncio
from typing import Any

from openai import AsyncOpenAI

from med_reason_evals.utils.parsing import parse_json_response
from med_reason_evals.utils.retry import call_with_retry


RUBRIC_JUDGE_TEMPLATE = """Your job is to evaluate a response against a rubric criterion.

# Response
{response}

# Rubric criterion
[{points}] {criterion}

# Instructions
Return a JSON object with:
- "explanation": Why the response does or does not meet the criterion.
- "criteria_met": Boolean indicating if the criterion is met.

Return just the JSON object."""


async def evaluate_criterion(
    response: str,
    criterion: str,
    points: int,
    judge_client: AsyncOpenAI,
    judge_model: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 500,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Evaluate a single rubric criterion.

    Args:
        response: The model's response to evaluate.
        criterion: The rubric criterion text.
        points: Points for this criterion.
        judge_client: The OpenAI client for judging.
        judge_model: The model to use for judging.
        semaphore: Semaphore to limit concurrent requests.
        max_tokens: Maximum tokens for judge response.
        temperature: Sampling temperature.

    Returns:
        Dict with 'criteria_met' (bool) and 'points' (int) keys.
    """
    async with semaphore:
        prompt = RUBRIC_JUDGE_TEMPLATE.format(
            response=response,
            criterion=criterion,
            points=points,
        )

        resp = await call_with_retry(
            judge_client.chat.completions.create,
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        judge_response = resp.choices[0].message.content or ""
        parsed = parse_json_response(judge_response)

        return {
            "criteria_met": bool(parsed.get("criteria_met", False)),
            "points": points,
            "explanation": parsed.get("explanation", ""),
        }


async def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    judge_client: AsyncOpenAI,
    judge_model: str = "openai/gpt-oss-120b",
    max_parallel_judges: int = 5,
    max_tokens: int = 500,
    temperature: float = 0.0,
) -> float:
    """Compute the reward score using multi-criteria rubric evaluation.

    Evaluates the response against all criteria in parallel and returns
    a normalized score based on points earned.

    Args:
        solution_str: The model's solution text.
        ground_truth: Dict with 'criteria' and 'points_list' keys.
        judge_client: The OpenAI client for judging.
        judge_model: The model to use for judging.
        max_parallel_judges: Maximum concurrent judge requests.
        max_tokens: Maximum tokens for each judge response.
        temperature: Sampling temperature.

    Returns:
        Normalized score between 0.0 and 1.0.
    """
    criteria = ground_truth.get("criteria", [])
    points_list = ground_truth.get("points_list", [])

    if not criteria or not points_list:
        return 0.0

    if len(criteria) != len(points_list):
        return 0.0

    # Calculate total positive points
    total_positive = sum(p for p in points_list if p > 0)
    if total_positive == 0:
        return 0.0

    # Evaluate all criteria in parallel with semaphore
    semaphore = asyncio.Semaphore(max_parallel_judges)
    tasks = [
        evaluate_criterion(
            response=solution_str,
            criterion=criterion,
            points=points,
            judge_client=judge_client,
            judge_model=judge_model,
            semaphore=semaphore,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for criterion, points in zip(criteria, points_list)
    ]

    results = await asyncio.gather(*tasks)

    # Calculate score
    earned = sum(r["points"] for r in results if r["criteria_met"])
    normalized_score = earned / total_positive

    # Clamp to [0, 1]
    return max(0.0, min(1.0, normalized_score))
