"""MedBullets verifiers evaluator.

Evaluator for MedBullets, a bank of USMLE Step 2/3 style questions used for
medical board preparation. The same questions ship in a four-option and a
five-option variant, so ``num_options`` selects the difficulty by varying the
number of distractors.

Unlike the other MCQ evaluators, MedBullets defaults to the BOXED answer
format (``\\boxed{}``), which pairs well with chain-of-thought reasoning.

Reference: https://step2.medbullets.com/
"""

import asyncio
import os

from datasets import Dataset
from openai import AsyncOpenAI

from med_reason_evals.data.medbullets import MedBulletsDataset
from med_reason_evals.verifiers.base import BaseMCQEvaluator, GroqGenConfig
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


class MedBulletsEvaluator(BaseMCQEvaluator):
    """Evaluator for MedBullets."""

    def __init__(
        self,
        num_options: int = 4,
        use_think: bool = False,
        system_prompt: str | None = None,
        answer_format: AnswerFormat | str = AnswerFormat.BOXED,
        streaming: bool | None = None,
    ) -> None:
        """Initialize the MedBullets evaluator.

        Args:
            num_options: Number of answer options per question (4 or 5).
                Validated when the dataset is loaded.
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
        self.num_options = num_options

    def _load_datasets(self) -> tuple[Dataset | None, Dataset]:
        # MedBullets publishes test splits only, so there is no train dataset.
        test_dataset = MedBulletsDataset(
            num_options=self.num_options,
            streaming=self.streaming,
        )
        return None, test_dataset.get_verifiers_dataset()


async def main() -> None:  # pragma: no cover
    """Run MedBullets evaluation with Groq API."""
    config = GroqGenConfig()
    client = AsyncOpenAI(
        api_key=os.getenv(config.api_key_env),
        base_url=config.base_url,
    )

    evaluator = MedBulletsEvaluator()
    results = await evaluator.evaluate(
        client=client,
        model=config.model,
        num_examples=100,
    )

    print(f"MedBullets Results: {results}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
