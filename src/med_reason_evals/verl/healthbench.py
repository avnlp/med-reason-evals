"""HealthBench Verl module.

Provides HealthBenchEvaluator for HealthBench RL training
with rubric-based multi-criteria evaluation.
"""

import asyncio
from typing import Any

from datasets import IterableDataset

from med_reason_evals.data.healthbench import HealthBenchDataset
from med_reason_evals.verl.base import BaseJudgeEvaluator, GroqGenConfig, JudgeConfig
from med_reason_evals.verl.rewards.healthbench_rubric import (
    compute_score as rubric_score,
)


class HealthBenchEvaluator(BaseJudgeEvaluator):
    """HealthBench evaluator for Verl pipelines.

    Evaluates models on HealthBench using multi-criteria rubric evaluation
    with parallel judging.

    Attributes:
        DATASET_NAME: Name of the dataset.
    """

    DATASET_NAME = "healthbench"

    def __init__(
        self,
        difficulty: str = "regular",
        max_parallel_judges: int = 5,
        gen_config: GroqGenConfig | None = None,
        judge_config: JudgeConfig | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the HealthBench evaluator.

        Args:
            difficulty: Dataset difficulty ("regular", "consensus", "hard").
            max_parallel_judges: Maximum concurrent judge requests.
            gen_config: Configuration for generation.
            judge_config: Configuration for judging.
            streaming: Whether to stream the dataset.
        """
        super().__init__(
            gen_config=gen_config,
            judge_config=judge_config,
            streaming=streaming,
        )
        self.difficulty = difficulty
        self.max_parallel_judges = max_parallel_judges

    def _load_dataset(self) -> IterableDataset:
        """Load the HealthBench dataset.

        Returns:
            IterableDataset formatted for Verl.
        """
        dataset = HealthBenchDataset(
            difficulty=self.difficulty,
            streaming=self.streaming,
        )
        return dataset.get_verl_dataset()

    async def _evaluate_example(
        self,
        prompt: list[dict[str, str]],
        ground_truth: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Evaluate a single example using rubric criteria.

        Args:
            prompt: List of message dicts.
            ground_truth: Dict with 'criteria' and 'points_list' keys.
            metadata: Optional metadata (unused).

        Returns:
            Normalized score from 0.0 to 1.0.
        """
        # No system prompt for HealthBench - uses dataset prompts
        completion = await self.rollouts.generate(messages=prompt)

        return await rubric_score(
            solution_str=completion,
            ground_truth=ground_truth,
            judge_client=self.judge_client,
            judge_model=self.judge_config.model,
            max_parallel_judges=self.max_parallel_judges,
            max_tokens=self.judge_config.max_tokens,
            temperature=self.judge_config.temperature,
        )

    def _build_result(
        self,
        avg_score: float,
        num_examples: int = 0,
    ) -> dict[str, Any]:
        """Build the evaluation result dictionary.

        Args:
            avg_score: Average score across all evaluated examples.
            num_examples: Number of successfully evaluated examples.

        Returns:
            Dictionary with dataset, difficulty, num_examples, and avg_score.
        """
        return {
            "dataset": self.DATASET_NAME,
            "difficulty": self.difficulty,
            "num_examples": num_examples,
            "avg_score": avg_score,
        }


async def main() -> None:  # pragma: no cover
    """Run HealthBench Verl evaluation."""
    evaluator = HealthBenchEvaluator()
    results = await evaluator.evaluate(num_examples=2)
    print(f"\nHealthBench Verl Results: {results}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
