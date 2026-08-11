"""MedBullets Verl module.

Provides MedBulletsEvaluator for MedBullets RL training with Groq rollouts.
"""

import asyncio
from typing import Any

from datasets import Dataset, IterableDataset

from med_reason_evals.data.medbullets import MedBulletsDataset
from med_reason_evals.verl.base import BaseMCQEvaluator, GroqGenConfig


class MedBulletsEvaluator(BaseMCQEvaluator):
    """MedBullets evaluator for Verl pipelines.

    Evaluates models on MedBullets medical board exam questions.

    Attributes:
        DATASET_NAME: Name of the dataset.
        DEFAULT_SYSTEM_PROMPT: Default system prompt for generation.
    """

    DATASET_NAME = "medbullets"
    DEFAULT_SYSTEM_PROMPT = (
        "You are a medical expert. Answer the following multiple-choice question. "
        "Think step by step and provide your final answer in <answer>X</answer> tags."
    )

    def __init__(
        self,
        num_options: int = 4,
        gen_config: GroqGenConfig | None = None,
        system_prompt: str | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the MedBullets evaluator.

        Args:
            num_options: Number of answer options (4 or 5).
            gen_config: Configuration for generation.
            system_prompt: Optional override for system prompt.
            streaming: Whether to stream the dataset.
        """
        super().__init__(
            gen_config=gen_config,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            streaming=streaming,
        )
        self.num_options = num_options

    def _load_dataset(self) -> Dataset | IterableDataset:
        """Load the MedBullets dataset.

        Returns:
            Dataset or IterableDataset formatted for Verl.
        """
        dataset = MedBulletsDataset(
            num_options=self.num_options,
            streaming=self.streaming,
        )
        return dataset.get_verl_dataset()

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
            Dictionary with dataset, num_examples, num_options, and avg_score.
        """
        return {
            "dataset": self.DATASET_NAME,
            "num_examples": num_examples,
            "num_options": self.num_options,
            "avg_score": avg_score,
        }


async def main() -> None:
    """Run MedBullets Verl evaluation."""
    evaluator = MedBulletsEvaluator()
    results = await evaluator.evaluate(num_examples=100)
    print(f"\nMedBullets Verl Results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
