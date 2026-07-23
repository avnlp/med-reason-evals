"""LLM-as-Judge reward helpers for verifiers evaluation.

This module provides JudgeRubric-first reward functions that use an LLM
to evaluate whether model responses are equivalent to ground truth answers.

Unlike rule-based rewards (MCQ accuracy), LLM-as-judge handles open-ended tasks
like diagnosis or explanation evaluation where exact matching fails. The judge
model compares predicted vs. ground truth outputs and returns a yes/no verdict.

Architecture:
- binary_judge_reward_from_template: Core reward function with template injection
- make_binary_judge_reward: Factory that binds templates to create reusable rewards
- parse_yes_no: Robust parser for judge yes/no responses (handles variations)

Template-based approach allows task-specific prompts without code changes.
Templates receive {prediction} and {ground_truth} placeholders.

The main entry points are:
- `parse_yes_no`: Parse yes/no responses from a judge
- `binary_judge_reward_from_template`: Reward function using template-based judging
- `make_binary_judge_reward`: Factory to create reward functions with baked-in templates

Template constants:
- `MEDCASEREASONING_JUDGE_TEMPLATE`: For diagnosis evaluation (medical terminology)
- `PUBHEALTHBENCH_JUDGE_TEMPLATE`: For general prediction vs ground truth
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from med_reason_evals.utils.parsing import parse_yes_no


if TYPE_CHECKING:
    from verifiers.parsers.parser import Parser
    from verifiers.types import Info, Messages, State


MEDCASEREASONING_JUDGE_TEMPLATE = """\
Is our predicted diagnosis correct (yes/no)?
Predicted diagnosis: {prediction}, True diagnosis: {ground_truth}
Answer [yes/no]."""

PUBHEALTHBENCH_JUDGE_TEMPLATE = """\
Is the predicted answer correct (yes/no)?
Predicted answer: {prediction}
Correct answer: {ground_truth}
Answer [yes/no]."""


async def binary_judge_reward_from_template(
    completion: Messages,
    answer: str,
    state: State,
    info: Info,
    parser: Parser,
    judge: Callable[..., Awaitable[str]],
    template: str,
    **kwargs: Any,
) -> float:
    """Reward function using LLM-as-judge with a template.

    Works with JudgeRubric argument filtering - receives parser and judge
    from the rubric's class_objects.

    Steps:
    1. Parse prediction from completion using parser.parse(completion, last=True)
       and extract answer field if XMLParser
    2. If missing prediction, record in info["judge_feedback"] and return 0.0
    3. Build prompt using template.format(prediction=..., ground_truth=...)
    4. Call judge(prompt=prompt_str, completion=pred_str, answer=answer, state=state)
    5. Parse judge response via parse_yes_no
    6. Append structured debug info into info["judge_feedback"]
    7. Return 1.0 for True else 0.0

    Args:
        completion: The model's completion (Messages - list of ChatMessage or str).
        answer: The correct answer (ground truth).
        state: The rollout state.
        info: Dictionary for tracking metadata and feedback.
        parser: Parser instance for extracting answers from completion.
        judge: The judge function from JudgeRubric.
        template: Prompt template with {prediction} and {ground_truth} placeholders.
        **kwargs: Additional arguments (ignored).

    Returns:
        1.0 if the judge says correct, 0.0 otherwise.
    """
    prediction: str | None = None

    # Parse prediction from completion
    # Check if parser has XMLParser-style parse method
    if hasattr(parser, "parse"):
        try:
            # XMLParser.parse returns a SimpleNamespace with fields
            parsed = parser.parse(completion, last=True)  # type: ignore[call-arg]

            # Extract answer field if XMLParser (has answer_field attribute)
            if parsed is not None:
                answer_field = getattr(parser, "answer_field", "answer")
                # Ensure answer_field is a string (not MagicMock from tests)
                if isinstance(answer_field, str) and hasattr(parsed, answer_field):
                    prediction = getattr(parsed, answer_field)
                elif isinstance(parsed, str):
                    prediction = parsed
        except TypeError:
            # Fallback for parsers that don't support last= argument
            parsed = parser.parse(completion)  # type: ignore[call-arg]
            if parsed is not None:
                answer_field = getattr(parser, "answer_field", "answer")
                # Ensure answer_field is a string (not MagicMock from tests)
                if isinstance(answer_field, str) and hasattr(parsed, answer_field):
                    prediction = getattr(parsed, answer_field)
                elif isinstance(parsed, str):
                    prediction = parsed

    # Fallback: use parse_answer if prediction still None
    if prediction is None and hasattr(parser, "parse_answer"):
        prediction = parser.parse_answer(completion)

    # If missing prediction, record and return 0.0
    if prediction is None:
        info.setdefault("judge_feedback", []).append(
            {
                "prediction": None,
                "answer": answer,
                "raw_judge": "no prediction extracted",
                "parsed": None,
                "is_correct": False,
            }
        )
        return 0.0

    prediction = str(prediction)

    prompt_str = template.format(prediction=prediction, ground_truth=answer)

    # JudgeRubric.judge signature: judge(prompt, completion, answer, state)
    judge_response = await judge(
        prompt=prompt_str,
        completion=prediction,
        answer=answer,
        state=state,
    )
    judge_response_str = str(judge_response)

    is_correct = parse_yes_no(judge_response_str)

    info.setdefault("judge_feedback", []).append(
        {
            "prediction": prediction,
            "answer": answer,
            "raw_judge": judge_response_str,
            "parsed": is_correct,
            "is_correct": is_correct is True,
        }
    )

    # None (unparsable) is treated as incorrect
    return 1.0 if is_correct is True else 0.0


def make_binary_judge_reward(
    template: str,
) -> Callable[..., Awaitable[float]]:
    """Create a binary judge reward function with a template baked in.

    This is a factory function that returns a reward function closure
    with the template pre-configured. The returned function is compatible
    with JudgeRubric's argument filtering.

    Args:
        template: Prompt template with {prediction} and {ground_truth} placeholders.

    Returns:
        An async reward function that can be added to a JudgeRubric.

    Example:
        >>> rubric = JudgeRubric(judge_client=client, judge_model="gpt-4o-mini")
        >>> reward_func = make_binary_judge_reward(MEDCASEREASONING_JUDGE_TEMPLATE)
        >>> rubric.add_reward_func(reward_func, weight=1.0)
    """

    async def reward_func(
        completion: Messages,
        answer: str,
        state: State,
        info: Info,
        parser: Parser,
        judge: Callable[..., Awaitable[str]],
        **kwargs: Any,
    ) -> float:
        return await binary_judge_reward_from_template(
            completion=completion,
            answer=answer,
            state=state,
            info=info,
            parser=parser,
            judge=judge,
            template=template,
            **kwargs,
        )

    # Set a descriptive name for debugging
    reward_func.__name__ = f"binary_judge_reward_{template[:20].replace(' ', '_')}"
    reward_func.__doc__ = f"Binary judge reward with template: {template[:50]}..."

    return reward_func
