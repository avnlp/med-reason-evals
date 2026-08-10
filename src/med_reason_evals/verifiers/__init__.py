"""Verifiers evaluators and shared utilities.

This module provides a unified interface to all medical reasoning evaluators
built on top of the verifiers framework. Each evaluator wraps a specific
dataset and provides standardized evaluation capabilities through verifiers
environments.

The module exposes three categories of components:

1. Base Evaluators: Shared evaluator classes (BaseVerifierEvaluator,
   BaseMCQEvaluator, BaseJudgeEvaluator) that build verifiers environments
   for medical QA tasks, along with their configuration dataclasses
   (GroqGenConfig, JudgeConfig). Dataset-specific evaluators are invoked via
   each dataset module's evaluator class (e.g.
   `med_reason_evals.verifiers.medqa.MedQAEvaluator`).

2. Reward Functions: Scoring functions for evaluating model responses,
   including multiple-choice accuracy, LLM-as-judge, and hybrid approaches.

3. Utilities: Helper functions for parsing model outputs (boxed answers,
   XML tags) and managing answer formats.

Example:
    >>> import asyncio
    >>> import os
    >>> from openai import AsyncOpenAI
    >>> from med_reason_evals.verifiers.medqa import MedQAEvaluator
    >>> client = AsyncOpenAI(
    ...     api_key=os.environ["GROQ_API_KEY"],
    ...     base_url="https://api.groq.com/openai/v1",
    ... )
    >>> evaluator = MedQAEvaluator(use_think=True)
    >>> results = asyncio.run(
    ...     evaluator.evaluate(
    ...         client=client, model="openai/gpt-oss-120b", num_examples=100
    ...     )
    ... )
"""

from med_reason_evals.verifiers.base import (
    BaseJudgeEvaluator,
    BaseMCQEvaluator,
    BaseVerifierEvaluator,
    GroqGenConfig,
    JudgeConfig,
)
from med_reason_evals.verifiers.healthbench import HealthBenchEvaluator
from med_reason_evals.verifiers.med_mcqa import MedMCQAEvaluator
from med_reason_evals.verifiers.medcasereasoning import MedCaseReasoningEvaluator
from med_reason_evals.verifiers.medqa import MedQAEvaluator
from med_reason_evals.verifiers.medxpertqa import MedXpertQAEvaluator
from med_reason_evals.verifiers.metamedqa import MetaMedQAEvaluator
from med_reason_evals.verifiers.mmlu_pro_health import MMLUProHealthEvaluator
from med_reason_evals.verifiers.pubhealthbench import PubHealthBenchEvaluator
from med_reason_evals.verifiers.pubmedqa import PubMedQAEvaluator
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
    # Dataset evaluators
    "HealthBenchEvaluator",
    "MedCaseReasoningEvaluator",
    "MedMCQAEvaluator",
    "MedQAEvaluator",
    "MedXpertQAEvaluator",
    "MetaMedQAEvaluator",
    "MMLUProHealthEvaluator",
    "PubHealthBenchEvaluator",
    "PubMedQAEvaluator",
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
