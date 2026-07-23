"""Reward functions for verifiers evaluation.

This module provides reward functions for evaluating medical reasoning
model responses across different task types. Reward functions integrate
with verifiers' Rubric system to assign scores (typically 0.0-1.0) based
on answer correctness or quality.

Three main approaches are supported:

1. Multiple-Choice Accuracy (multiple_choice_accuracy.py):
   Extract and compare option letters/numbers with fuzzy matching
   for models that output full text or various formats.

2. LLM-as-Judge (llm_as_judge.py):
   Use a separate judge model to evaluate open-ended responses
   against ground truth, with templates for specific tasks.

3. Rubric-Based (judge_rubric.py, hybrid_pubhealthbench.py):
   Evaluate against multiple criteria with weighted scoring,
   supporting complex multi-point rubrics like HealthBench.

All reward functions follow the verifiers protocol: async functions
receiving (completion, answer, info, state, parser, **kwargs) and
returning float scores.
"""

from med_reason_evals.verifiers.rewards.hybrid_pubhealthbench import (
    pubhealthbench_reward,
)
from med_reason_evals.verifiers.rewards.judge_rubric import (
    healthbench_rubric_reward,
    parse_json_response,
)
from med_reason_evals.verifiers.rewards.llm_as_judge import (
    MEDCASEREASONING_JUDGE_TEMPLATE,
    PUBHEALTHBENCH_JUDGE_TEMPLATE,
    binary_judge_reward_from_template,
    make_binary_judge_reward,
    parse_yes_no,
)
from med_reason_evals.verifiers.rewards.multiple_choice_accuracy import (
    MCQAccuracyResult,
    accuracy_reward,
    multiple_choice_accuracy,
)


__all__ = [
    # MCQ Accuracy
    "MCQAccuracyResult",
    "accuracy_reward",
    "multiple_choice_accuracy",
    # LLM-as-Judge
    "MEDCASEREASONING_JUDGE_TEMPLATE",
    "PUBHEALTHBENCH_JUDGE_TEMPLATE",
    "binary_judge_reward_from_template",
    "make_binary_judge_reward",
    "parse_yes_no",
    # JudgeRubric
    "healthbench_rubric_reward",
    "parse_json_response",
    # Hybrid
    "pubhealthbench_reward",
]
