"""PubHealthBench Verl module.

Provides PubHealthBenchEvaluator for PubHealthBench RL training
with hybrid MCQ/Judge evaluation.
"""

import asyncio
from typing import Any

from datasets import Dataset, IterableDataset

from med_reason_evals.data.pubhealthbench import PubHealthBenchDataset
from med_reason_evals.verl.base import BaseJudgeEvaluator, GroqGenConfig, JudgeConfig
from med_reason_evals.verl.rewards.hybrid_pubhealthbench import (
    compute_score as hybrid_score,
)


class PubHealthBenchEvaluator(BaseJudgeEvaluator):
    """PubHealthBench evaluator for Verl pipelines.

    Evaluates models on PubHealthBench questions using hybrid evaluation:
    - MCQ accuracy for multiple-choice questions
    - LLM-as-Judge for freeform questions

    Attributes:
        DATASET_NAME: Name of the dataset.
        DEFAULT_SYSTEM_PROMPT: Default system prompt for generation.
    """

    DATASET_NAME = "pubhealthbench"
    DEFAULT_SYSTEM_PROMPT = (
        "You are a public health expert. Answer the following question. "
        "Think step by step and provide your final answer in <answer>...</answer> tags."
    )

    def __init__(
        self,
        split: str = "test",
        question_type: str = "all",
        gen_config: GroqGenConfig | None = None,
        judge_config: JudgeConfig | None = None,
        system_prompt: str | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the PubHealthBench evaluator.

        Args:
            split: Dataset split to use.
            question_type: Filter by type ("mcq", "freeform", "all").
            gen_config: Configuration for generation.
            judge_config: Configuration for judging.
            system_prompt: Optional override for system prompt.
            streaming: Whether to stream the dataset.
        """
        super().__init__(
            gen_config=gen_config,
            judge_config=judge_config,
            streaming=streaming,
        )
        self.split = split
        self.question_type = question_type
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def _load_dataset(self) -> Dataset | IterableDataset:
        """Load the PubHealthBench dataset.

        Returns:
            Dataset or IterableDataset formatted for Verl.
        """
        dataset = PubHealthBenchDataset(
            split=self.split,
            question_type=self.question_type,
            streaming=self.streaming,
        )
        return dataset.get_verl_dataset()

    async def _evaluate_example(
        self,
        prompt: list[dict[str, str]],
        ground_truth: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Evaluate a single example using hybrid scoring.

        Args:
            prompt: List of message dicts.
            ground_truth: Dict with answer information.
            metadata: Dict with 'is_mcq' key.

        Returns:
            Score from 0.0 to 1.0.
        """
        messages = [{"role": "system", "content": self.system_prompt}] + prompt
        completion = await self.rollouts.generate(
            messages=messages,
            **self.gen_config.sampling_args,
        )

        metadata = metadata or {}
        return await hybrid_score(
            solution_str=completion,
            ground_truth=ground_truth,
            metadata=metadata,
            judge_client=(None if metadata.get("is_mcq", False) else self.judge_client),
            judge_model=self.judge_config.model,
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
            Dictionary with dataset, split, question_type, num_examples, and avg_score.
        """
        return {
            "dataset": self.DATASET_NAME,
            "split": self.split,
            "question_type": self.question_type,
            "num_examples": num_examples,
            "avg_score": avg_score,
        }


async def main() -> None:
    """Run PubHealthBench Verl evaluation."""
    evaluator = PubHealthBenchEvaluator()
    results = await evaluator.evaluate(num_examples=100)
    print(f"\nPubHealthBench Verl Results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
