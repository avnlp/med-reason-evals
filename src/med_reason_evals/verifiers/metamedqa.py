"""MetaMedQA verifiers evaluator.

Evaluator for MetaMedQA, an aggregated collection of multiple medical QA datasets
combined into a unified benchmark. This provides broad coverage across different
medical domains and question styles.

MetaMedQA enables evaluation on diverse medical knowledge without needing
separate evaluators for each source dataset. It's useful for holistic assessment
of medical AI systems across varied content.

The evaluator uses standard MCQ patterns with XML answer extraction.

Reference: https://github.com/MetaMedQA/MetaMedQA
"""

import asyncio
import os

from datasets import Dataset
from openai import AsyncOpenAI

from med_reason_evals.data.metamedqa import MetaMedQADataset
from med_reason_evals.verifiers.base import BaseMCQEvaluator, GroqGenConfig
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


class MetaMedQAEvaluator(BaseMCQEvaluator):
    """Evaluator for MetaMedQA."""

    def __init__(
        self,
        use_think: bool = False,
        system_prompt: str | None = None,
        answer_format: AnswerFormat | str = AnswerFormat.XML,
        streaming: bool | None = None,
    ) -> None:
        """Initialize the MetaMedQA evaluator.

        Args:
            use_think: Whether to include thinking tags in the output.
            system_prompt: Custom system prompt to use. If None, uses default.
            answer_format: Format for answers (XML or BOXED).
            streaming: Whether to use streaming mode for dataset loading.
                Defaults to False if not specified.
        """
        super().__init__(
            use_think=use_think,
            system_prompt=system_prompt,
            answer_format=answer_format,
            streaming=streaming,
        )

    def _load_datasets(self) -> tuple[Dataset | None, Dataset]:
        test_dataset = MetaMedQADataset(split="test", streaming=self.streaming)
        return None, test_dataset.get_verifiers_dataset()


async def main() -> None:  # pragma: no cover
    """Run MetaMedQA evaluation with Groq API."""
    config = GroqGenConfig()
    client = AsyncOpenAI(
        api_key=os.getenv(config.api_key_env),
        base_url=config.base_url,
    )

    evaluator = MetaMedQAEvaluator()
    results = await evaluator.evaluate(
        client=client,
        model=config.model,
        num_examples=100,
    )

    print(f"MetaMedQA Results: {results}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
