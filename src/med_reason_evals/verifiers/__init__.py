"""Verifiers evaluators and shared utilities.

This module provides a unified interface to all medical reasoning evaluators
built on top of the verifiers framework. Each evaluator wraps a specific
dataset and provides standardized evaluation capabilities through verifiers
environments.

The module exposes three categories of components:

1. Base Evaluators: Shared evaluator classes (BaseVerifierEvaluator,
   BaseMCQEvaluator, BaseJudgeEvaluator) that build verifiers environments
   for medical QA tasks, along with their configuration dataclasses
   (GroqGenConfig, JudgeConfig). Dataset-specific evaluation is otherwise
   invoked via each dataset module's `load_environment()` function (e.g.
   `med_reason_evals.verifiers.medqa.load_environment`).

2. Reward Functions: Scoring functions for evaluating model responses,
   including multiple-choice accuracy, LLM-as-judge, and hybrid approaches.

3. Utilities: Helper functions for parsing model outputs (boxed answers,
   XML tags) and managing answer formats.

Example:
    >>> import asyncio
    >>> from med_reason_evals.verifiers.medqa import load_environment
    >>> env = load_environment(use_think=True)
    >>> results = asyncio.run(env.evaluate(client, model="gpt-4", num_examples=100))
"""

from med_reason_evals.verifiers.base import (
    BaseJudgeEvaluator,
    BaseMCQEvaluator,
    BaseVerifierEvaluator,
    GroqGenConfig,
    JudgeConfig,
)
from med_reason_evals.verifiers.rewards import (
    MCQAccuracyResult,
    accuracy_reward,
    healthbench_rubric_reward,
    make_binary_judge_reward,
    multiple_choice_accuracy,
    parse_yes_no,
    pubhealthbench_reward,
)
from med_reason_evals.verifiers.utils import (
    AnswerFormat,
    extract_boxed_answer,
    strip_think_tags,
)


__all__ = [
    # Base evaluators
    "BaseJudgeEvaluator",
    "BaseMCQEvaluator",
    "BaseVerifierEvaluator",
    "GroqGenConfig",
    "JudgeConfig",
    # Rewards
    "MCQAccuracyResult",
    "accuracy_reward",
    "multiple_choice_accuracy",
    "parse_yes_no",
    "make_binary_judge_reward",
    "healthbench_rubric_reward",
    "pubhealthbench_reward",
    # Utils
    "AnswerFormat",
    "extract_boxed_answer",
    "strip_think_tags",
]
