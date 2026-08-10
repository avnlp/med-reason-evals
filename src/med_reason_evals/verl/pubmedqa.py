"""PubMedQA Verl module.

Provides PubMedQAEvaluator for PubMedQA RL training with Groq rollouts.
"""

import asyncio
from typing import Any

from datasets import IterableDataset

from med_reason_evals.data.pubmedqa import PubMedQADataset
from med_reason_evals.verl.base import BaseMCQEvaluator, GroqGenConfig


class PubMedQAEvaluator(BaseMCQEvaluator):
    """PubMedQA evaluator for Verl pipelines.

    Evaluates models on PubMedQA yes/no/maybe biomedical questions.

    Attributes:
        DATASET_NAME: Name of the dataset.
        DEFAULT_SYSTEM_PROMPT: Default system prompt for generation.
    """

    DATASET_NAME = "pubmedqa"
    DEFAULT_SYSTEM_PROMPT = (
        "You are a medical expert. Answer the following yes/no/maybe question. "
        "Think step by step and provide your final answer in <answer>X</answer> tags "
        "where X is A (Yes), B (No), or C (Maybe)."
    )

    def __init__(
        self,
        gen_config: GroqGenConfig | None = None,
        system_prompt: str | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the PubMedQA evaluator.

        Args:
            gen_config: Configuration for generation.
            system_prompt: Optional override for system prompt.
            streaming: Whether to stream the dataset.
        """
        super().__init__(
            gen_config=gen_config,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            streaming=streaming,
        )

    def _load_dataset(self) -> IterableDataset:
        """Load the PubMedQA dataset.

        Returns:
            IterableDataset formatted for Verl.
        """
        dataset = PubMedQADataset(streaming=self.streaming)
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
            Dictionary with dataset, num_examples, and avg_score.
        """
        return {
            "dataset": self.DATASET_NAME,
            "num_examples": num_examples,
            "avg_score": avg_score,
        }


async def main() -> None:  # pragma: no cover
    """Run PubMedQA Verl evaluation."""
    evaluator = PubMedQAEvaluator()
    results = await evaluator.evaluate(num_examples=100)
    print(f"\nPubMedQA Verl Results: {results}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
