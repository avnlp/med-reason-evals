"""MedXpertQA verifiers evaluator.

Evaluator for the MedXpertQA dataset, a challenging medical reasoning
benchmark with questions at different difficulty levels. This evaluator
supports both text-based and image-based questions through the question_type
parameter.

MedXpertQA tests medical knowledge across various clinical scenarios
with expert-level complexity. The evaluator uses the standard MCQ pattern
with XML or boxed answer extraction.

Reference: https://github.com/medxpertqa/MedXpertQA
"""

import asyncio
import os

from datasets import Dataset
from openai import AsyncOpenAI

from med_reason_evals.data.medxpertqa import MedXpertQADataset
from med_reason_evals.verifiers.base import BaseMCQEvaluator, GroqGenConfig
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


class MedXpertQAEvaluator(BaseMCQEvaluator):
    """Evaluator for MedXpertQA."""

    def __init__(
        self,
        question_type: str = "all",
        use_think: bool = False,
        system_prompt: str | None = None,
        answer_format: AnswerFormat | str = AnswerFormat.XML,
        streaming: bool | None = None,
    ) -> None:
        """Initialize the MedXpertQA evaluator.

        Args:
            question_type: Type of questions to load ("all" or specific type).
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
        self.question_type = question_type

    def _load_datasets(self) -> tuple[Dataset | None, Dataset]:
        test_dataset = MedXpertQADataset(
            split="test",
            streaming=self.streaming,
            question_type=self.question_type,
        )
        return None, test_dataset.get_verifiers_dataset()


async def main() -> None:  # pragma: no cover
    """Run MedXpertQA evaluation with Groq API."""
    config = GroqGenConfig()
    client = AsyncOpenAI(
        api_key=os.getenv(config.api_key_env),
        base_url=config.base_url,
    )

    evaluator = MedXpertQAEvaluator()
    results = await evaluator.evaluate(
        client=client,
        model=config.model,
        num_examples=100,
    )

    print(f"MedXpertQA Results: {results}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
