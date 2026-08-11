"""MedXpertQA Verl module.

Provides MedXpertQAEvaluator for MedXpertQA RL training with Groq rollouts.
"""

import asyncio
from typing import Any

from datasets import Dataset, IterableDataset

from med_reason_evals.data.medxpertqa import MedXpertQADataset
from med_reason_evals.verl.base import BaseMCQEvaluator, GroqGenConfig


class MedXpertQAEvaluator(BaseMCQEvaluator):
    """MedXpertQA evaluator for Verl pipelines.

    Evaluates models on MedXpertQA expert-level medical questions.

    Attributes:
        DATASET_NAME: Name of the dataset.
        DEFAULT_SYSTEM_PROMPT: Default system prompt for generation.
    """

    DATASET_NAME = "medxpertqa"
    DEFAULT_SYSTEM_PROMPT = (
        "You are a medical expert. Answer the following multiple-choice question. "
        "Think step by step and provide your final answer in <answer>X</answer> tags."
    )

    def __init__(
        self,
        split: str = "test",
        question_type: str = "all",
        gen_config: GroqGenConfig | None = None,
        system_prompt: str | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the MedXpertQA evaluator.

        Args:
            split: Dataset split to use.
            question_type: Filter by type ("reasoning", "understanding", "all").
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
        self.question_type = question_type

    def _load_dataset(self) -> Dataset | IterableDataset:
        """Load the MedXpertQA dataset.

        Returns:
            Dataset or IterableDataset formatted for Verl.
        """
        dataset = MedXpertQADataset(
            split=self.split,
            question_type=self.question_type,
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
            Dictionary with dataset, split, question_type, num_examples, and
            avg_score.
        """
        return {
            "dataset": self.DATASET_NAME,
            "split": self.split,
            "question_type": self.question_type,
            "num_examples": num_examples,
            "avg_score": avg_score,
        }


async def main() -> None:  # pragma: no cover
    """Run MedXpertQA Verl evaluation."""
    evaluator = MedXpertQAEvaluator()
    results = await evaluator.evaluate(num_examples=100)
    print(f"\nMedXpertQA Verl Results: {results}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
