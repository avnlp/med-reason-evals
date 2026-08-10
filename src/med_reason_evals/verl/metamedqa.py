"""MetaMedQA Verl module.

Provides MetaMedQAEvaluator for MetaMedQA RL training with Groq rollouts.
"""

import asyncio
from typing import Any

from datasets import IterableDataset

from med_reason_evals.data.metamedqa import MetaMedQADataset
from med_reason_evals.verl.base import BaseMCQEvaluator, GroqGenConfig


class MetaMedQAEvaluator(BaseMCQEvaluator):
    """MetaMedQA evaluator for Verl pipelines.

    Evaluates models on MetaMedQA multilingual medical questions.

    Attributes:
        DATASET_NAME: Name of the dataset.
        DEFAULT_SYSTEM_PROMPT: Default system prompt for generation.
    """

    DATASET_NAME = "metamedqa"
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
        """Initialize the MetaMedQA evaluator.

        Args:
            split: Dataset split to use.
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
        """Load the MetaMedQA dataset.

        Returns:
            IterableDataset formatted for Verl.
        """
        dataset = MetaMedQADataset(split=self.split, streaming=self.streaming)
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


async def main() -> None:
    """Run MetaMedQA Verl evaluation."""
    evaluator = MetaMedQAEvaluator()
    results = await evaluator.evaluate(num_examples=100)
    print("\nMetaMedQA Verl Results:")
    print(f"  Dataset: {results['dataset']}")
    print(f"  Split: {results['split']}")
    print(f"  Examples: {results['num_examples']}")
    print(f"  Accuracy: {results['accuracy']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
