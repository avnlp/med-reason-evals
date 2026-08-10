"""MedMCQA Verl module.

Provides MedMCQAEvaluator for MedMCQA RL training with Groq rollouts.
"""

import asyncio
from typing import Any

from datasets import IterableDataset

from med_reason_evals.data.med_mcqa import MedMCQADataset
from med_reason_evals.verl.base import BaseMCQEvaluator, GroqGenConfig


class MedMCQAEvaluator(BaseMCQEvaluator):
    """MedMCQA evaluator for Verl pipelines.

    Evaluates models on MedMCQA multiple-choice questions from Indian medical exams.

    Attributes:
        DATASET_NAME: Name of the dataset.
        DEFAULT_SYSTEM_PROMPT: Default system prompt for generation.
    """

    DATASET_NAME = "med_mcqa"
    DEFAULT_SYSTEM_PROMPT = (
        "You are a medical expert. Answer the following multiple-choice question. "
        "Think step by step and provide your final answer in <answer>X</answer> tags."
    )

    def __init__(
        self,
        split: str = "validation",
        gen_config: GroqGenConfig | None = None,
        system_prompt: str | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the MedMCQA evaluator.

        Args:
            split: Dataset split to use ("train" or "validation").
            gen_config: Configuration for generation.
            system_prompt: Optional override for system prompt.
            streaming: Whether to stream the dataset.
        """
        super().__init__(
            gen_config=gen_config,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            streaming=streaming,
        )
        self.split = split

    def _load_dataset(self) -> IterableDataset:
        """Load the MedMCQA dataset.

        Returns:
            IterableDataset formatted for Verl.
        """
        dataset = MedMCQADataset(split=self.split, streaming=self.streaming)
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
            Dictionary with dataset, split, num_examples, avg_score, and accuracy.
        """
        return {
            "dataset": self.DATASET_NAME,
            "split": self.split,
            "num_examples": num_examples,
            "avg_score": avg_score,
            "accuracy": avg_score,
        }


async def main() -> None:  # pragma: no cover
    """Run MedMCQA Verl evaluation."""
    evaluator = MedMCQAEvaluator()
    results = await evaluator.evaluate(num_examples=100)
    print("\nMedMCQA Verl Results:")
    print(f"  Dataset: {results['dataset']}")
    print(f"  Split: {results['split']}")
    print(f"  Examples: {results['num_examples']}")
    print(f"  Accuracy: {results['accuracy']:.3f}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
