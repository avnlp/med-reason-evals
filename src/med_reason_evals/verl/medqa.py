"""MedQA Verl module.

Provides MedQAEvaluator for MedQA RL training with Groq rollouts.
"""

import asyncio
from typing import Any

from datasets import IterableDataset

from med_reason_evals.data.medqa import MedQADataset
from med_reason_evals.verl.base import BaseMCQEvaluator, GroqGenConfig


class MedQAEvaluator(BaseMCQEvaluator):
    """MedQA evaluator for Verl pipelines.

    Evaluates models on MedQA-USMLE-4 multiple-choice questions.

    Attributes:
        DATASET_NAME: Name of the dataset.
        DEFAULT_SYSTEM_PROMPT: Default system prompt for generation.
    """

    DATASET_NAME = "medqa"
    DEFAULT_SYSTEM_PROMPT = (
        "You are a medical expert. Answer the following multiple-choice question. "
        "Think step by step and provide your final answer in <answer>X</answer> tags."
    )

    def __init__(
        self,
        split: str = "test",
        gen_config: GroqGenConfig | None = None,
        system_prompt: str | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the MedQA evaluator.

        Args:
            split: Dataset split to use ("train" or "test").
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
        """Load the MedQA dataset.

        Returns:
            IterableDataset formatted for Verl.
        """
        dataset = MedQADataset(split=self.split, streaming=self.streaming)
        return dataset.get_verl_dataset()

    def _build_result(
        self,
        scores: list[float],
        avg_score: float,
    ) -> dict[str, Any]:
        """Build the evaluation result dictionary.

        Args:
            scores: List of individual scores.
            avg_score: Average score.

        Returns:
            Dictionary with dataset, split, num_examples, avg_score, and accuracy.
        """
        return {
            "dataset": self.DATASET_NAME,
            "split": self.split,
            "num_examples": len(scores),
            "avg_score": avg_score,
            "accuracy": avg_score,
        }


async def main() -> None:
    """Run MedQA Verl evaluation."""
    evaluator = MedQAEvaluator()
    results = await evaluator.evaluate(num_examples=100)
    print("\nMedQA Verl Results:")
    print(f"  Dataset: {results['dataset']}")
    print(f"  Split: {results['split']}")
    print(f"  Examples: {results['num_examples']}")
    print(f"  Accuracy: {results['accuracy']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
