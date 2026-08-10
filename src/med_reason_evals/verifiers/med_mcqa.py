"""MedMCQA verifiers evaluator.

Evaluator for MedMCQA (Medical Multiple Choice Question Answering), a large-scale
MCQ dataset covering diverse medical topics. Questions are sourced from Indian
medical entrance exams (AIIMS, NEET) with 4 options each.

The dataset covers:
- Anatomy, Physiology, Biochemistry
- Pathology, Microbiology, Pharmacology
- Medicine, Surgery, Pediatrics, and more

This is one of the largest medical QA datasets, making it valuable for
training and evaluating medical language models.

Reference: https://github.com/medmcqa/medmcqa
"""

import asyncio
import os

from datasets import Dataset
from openai import AsyncOpenAI

from med_reason_evals.data.med_mcqa import MedMCQADataset
from med_reason_evals.verifiers.base import BaseMCQEvaluator, GroqGenConfig
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


class MedMCQAEvaluator(BaseMCQEvaluator):
    """Evaluator for MedMCQA."""

    def __init__(
        self,
        use_think: bool = False,
        system_prompt: str | None = None,
        answer_format: AnswerFormat | str = AnswerFormat.XML,
        streaming: bool | None = None,
    ) -> None:
        """Initialize the MedMCQA evaluator.

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
        train_dataset = MedMCQADataset(split="train", streaming=self.streaming)
        val_dataset = MedMCQADataset(split="validation", streaming=self.streaming)
        return (
            train_dataset.get_verifiers_dataset(),
            val_dataset.get_verifiers_dataset(),
        )


async def main() -> None:  # pragma: no cover
    """Run MedMCQA evaluation with Groq API."""
    config = GroqGenConfig()
    client = AsyncOpenAI(
        api_key=os.getenv(config.api_key_env),
        base_url=config.base_url,
    )

    evaluator = MedMCQAEvaluator()
    results = await evaluator.evaluate(
        client=client,
        model=config.model,
        num_examples=100,
    )

    print(f"MedMCQA Results: {results}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
