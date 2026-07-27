"""MedQA verifiers evaluator.

Evaluator for MedQA-USMLE-4-options, a multiple-choice question dataset
based on the United States Medical Licensing Examination (USMLE).

This dataset tests medical knowledge at the level required for medical
licensure in the United States, covering:
- Basic science knowledge
- Clinical reasoning
- Patient management
- Diagnostic interpretation

The 4-option format aligns with standard medical board exams, making
this a practical benchmark for medical AI capabilities.

Reference: https://github.com/jind11/MedQA
"""

import asyncio
import os

from datasets import Dataset
from openai import AsyncOpenAI

from med_reason_evals.data.medqa import MedQADataset
from med_reason_evals.verifiers.base import BaseMCQEvaluator, GroqGenConfig
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


class MedQAEvaluator(BaseMCQEvaluator):
    """Evaluator for MedQA-USMLE-4-options."""

    def __init__(
        self,
        use_think: bool = False,
        system_prompt: str | None = None,
        answer_format: AnswerFormat | str = AnswerFormat.XML,
        streaming: bool | None = None,
    ) -> None:
        """Initialize the MedQA evaluator.

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
        train_dataset = MedQADataset(split="train", streaming=self.streaming)
        test_dataset = MedQADataset(split="test", streaming=self.streaming)
        return (
            train_dataset.get_verifiers_dataset(),
            test_dataset.get_verifiers_dataset(),
        )


async def main() -> None:
    """Run MedQA evaluation with Groq API."""
    config = GroqGenConfig()
    client = AsyncOpenAI(
        api_key=os.getenv(config.api_key_env),
        base_url=config.base_url,
    )

    evaluator = MedQAEvaluator()
    results = await evaluator.evaluate(
        client=client,
        model=config.model,
        num_examples=100,
    )

    print(f"MedQA Results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
