"""JudgeRubric reward function for multi-criteria evaluation.

This module implements HealthBench's rubric-based evaluation where responses
are scored against multiple weighted criteria rather than binary correct/incorrect.

Evaluation flow:
1. Parse completion to extract response text
2. Format conversation history for judge context
3. For each criterion (with points), call judge to assess if met
4. Aggregate: score = earned_points / total_positive_points
5. Store detailed results in info["rubric_results"] for debugging

Concurrency is controlled via semaphore (max_parallel_judges) to avoid
overwhelming the judge API with simultaneous requests.

The judge prompt includes the full conversation plus the specific rubric item,
allowing contextual evaluation of nuanced criteria.
"""

from __future__ import annotations

import asyncio
from typing import Any

from verifiers.types import Messages, State

from med_reason_evals.utils.parsing import parse_json_response


# Template for asking judge to evaluate a single rubric criterion
# Note: Judge receives full conversation context plus specific item to assess
HEALTHBENCH_JUDGE_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn
(i.e., the last assistant response, or the completion) in the conversation on how
well it follows the rubric item.

# Conversation
{conversation}

# Rubric item
{rubric_item}

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or
does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response
meets the criteria of the rubric item.

Return just the json object in markdown format. Do not include any other text in
the response.
""".strip()


def format_conversation(prompt: Messages, completion_text: str) -> str:
    """Format the conversation for a judge prompt."""
    lines: list[str] = []
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict):
                role = message.get("role", "")
                content = message.get("content", "")
                if role and content:
                    lines.append(f"{role}: {content}")
    lines.append(f"assistant: {completion_text}")
    return "\n\n".join(lines)


async def healthbench_rubric_reward(
    prompt: Messages,
    completion: Messages,
    info: dict[str, Any],
    state: State,
    judge,
    max_parallel_judges: int = 5,
    **kwargs: Any,
) -> float:
    """Evaluate response against multiple HealthBench rubric criteria."""
    if isinstance(completion, list) and completion:
        completion_text = completion[-1].get("content", "")
    else:
        completion_text = str(completion)

    criteria = info.get("criteria", [])
    points_list = info.get("points_list", [])

    if not criteria or not points_list:
        return 0.0

    total_positive = sum(points for points in points_list if points > 0)
    if total_positive == 0:
        return 0.0

    conversation = format_conversation(prompt, completion_text)
    # Semaphore limits concurrent judge API calls to avoid rate limits
    semaphore = asyncio.Semaphore(max_parallel_judges)

    async def _judge_single(idx: int, criterion: str, points: int) -> dict[str, Any]:
        # Acquire semaphore to limit concurrency
        async with semaphore:
            # Format: "[5] Response includes safety warning"
            rubric_item = f"[{points}] {criterion}"
            full_prompt = HEALTHBENCH_JUDGE_TEMPLATE.format(
                conversation=conversation,
                rubric_item=rubric_item,
            )
            raw_resp = await judge(
                prompt=[{"role": "user", "content": full_prompt}],
                completion="",
                answer="",
                state=state,
            )
            dict_resp = parse_json_response(str(raw_resp))
            criteria_met = (
                bool(dict_resp.get("criteria_met", False))
                if isinstance(dict_resp, dict)
                else False
            )
            return {
                "idx": idx,
                "points_possible": points,
                "criteria_met": criteria_met,
                "judge_explanation": dict_resp.get("explanation"),
            }

    # Launch all criteria evaluations concurrently (up to max_parallel_judges at once)
    tasks = [
        _judge_single(idx, criterion, points)
        for idx, (criterion, points) in enumerate(zip(criteria, points_list))
    ]
    judgments = await asyncio.gather(*tasks)

    earned_points = sum(
        j["points_possible"] if j["criteria_met"] else 0 for j in judgments
    )
    info.setdefault("rubric_results", []).extend(judgments)

    return float(max(0.0, min(1.0, earned_points / total_positive)))
