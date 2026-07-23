"""Verifiers evaluators and shared utilities.

This module provides a unified interface to all medical reasoning evaluators
built on top of the verifiers framework. Each evaluator wraps a specific
dataset and provides standardized evaluation capabilities through verifiers
environments.

The module exposes three categories of components:

1. Evaluators: Dataset-specific evaluation classes (e.g., MedQAEvaluator,
   PubMedQAEvaluator) that build verifiers environments for medical QA tasks.

2. Reward Functions: Scoring functions for evaluating model responses,
   including multiple-choice accuracy, LLM-as-judge, and hybrid approaches.

3. Utilities: Helper functions for parsing model outputs (boxed answers,
   XML tags) and managing answer formats.

Example:
    >>> from med_reason_evals.verifiers import MedQAEvaluator
    >>> evaluator = MedQAEvaluator(use_think=True)
    >>> env = evaluator.environment()
    >>> import asyncio
    >>> results = asyncio.run(
    ...     evaluator.evaluate(client, model="gpt-4", num_examples=100)
    ... )
"""

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
