"""Verl reward functions for reinforcement learning training.

This package provides reward computation functions for RL training on medical datasets.
Each reward function implements a specific evaluation strategy:

Reward Functions:
    - multiple_choice_accuracy: Extracts and validates MCQ answers against ground truth.
    - llm_as_judge: Uses an LLM to judge correctness of open-ended responses.
    - semantic_equivalence: Matches answers using exact or substring comparison.
    - healthbench_rubric: Multi-criteria rubric evaluation with parallel judging.
    - hybrid_pubhealthbench: Routes between MCQ and judge-based evaluation.

Most reward functions follow a similar async signature:
    async compute_score(solution_str, ground_truth, **kwargs) -> float

Synchronous exceptions:
    - mcq_score (sync): Simple rule-based MCQ accuracy.
    - semantic_score (sync): Normalized text matching.

Async functions (require await):
    - llm_judge_score: Requires a judge_client; judge_model is optional.
    - healthbench_score: Requires a judge_client; judge_model is optional.
    - pubhealthbench_score: Requires a judge_client; judge_model is optional.

Returns a score between 0.0 and 1.0, where 1.0 indicates a correct answer.
"""

from med_reason_evals.verl.rewards.healthbench_rubric import (
    compute_score as healthbench_score,
)
from med_reason_evals.verl.rewards.hybrid_pubhealthbench import (
    PUBHEALTHBENCH_JUDGE_TEMPLATE,
)
from med_reason_evals.verl.rewards.hybrid_pubhealthbench import (
    compute_score as pubhealthbench_score,
)
from med_reason_evals.verl.rewards.llm_as_judge import (
    DEFAULT_JUDGE_PROMPT,
    judge_answer,
    make_judge_reward,
)
from med_reason_evals.verl.rewards.llm_as_judge import (
    compute_score as llm_judge_score,
)
from med_reason_evals.verl.rewards.multiple_choice_accuracy import (
    compute_score as mcq_accuracy_score,
)
from med_reason_evals.verl.rewards.multiple_choice_accuracy import (
    compute_score as mcq_score,
)
from med_reason_evals.verl.rewards.multiple_choice_accuracy import (
    extract_answer as mcq_extract_answer,
)
from med_reason_evals.verl.rewards.multiple_choice_accuracy import (
    normalize_answer as mcq_normalize_answer,
)
from med_reason_evals.verl.rewards.semantic_equivalence import (
    compute_score as semantic_score,
)
from med_reason_evals.verl.rewards.semantic_equivalence import (
    exact_match,
    substring_match,
)


__all__ = [
    # MCQ Accuracy
    "mcq_accuracy_score",
    "mcq_score",
    "mcq_extract_answer",
    "mcq_normalize_answer",
    # LLM-as-Judge
    "llm_judge_score",
    "judge_answer",
    "make_judge_reward",
    "DEFAULT_JUDGE_PROMPT",
    # HealthBench Rubric
    "healthbench_score",
    # PubHealthBench Hybrid
    "pubhealthbench_score",
    "PUBHEALTHBENCH_JUDGE_TEMPLATE",
    # Semantic Equivalence
    "semantic_score",
    "exact_match",
    "substring_match",
]
