"""PubMedQA verifiers evaluator.

Evaluator for PubMedQA, a biomedical question answering dataset based on
PubMed abstracts. Questions are answered as yes/no/maybe from medical
literature contexts.

This evaluator uses the canonical 500-example human-annotated test split
from ``openlifescienceai/pubmedqa``.

The yes/no/maybe format tests models' ability to synthesize evidence
from scientific abstracts and make determinations about biomedical claims.

Reference: https://pubmedqa.github.io/
"""

import asyncio
import os

from datasets import Dataset
from openai import AsyncOpenAI

from med_reason_evals.data.pubmedqa import PubMedQADataset
from med_reason_evals.verifiers.base import BaseMCQEvaluator, GroqGenConfig
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


class PubMedQAEvaluator(BaseMCQEvaluator):
    """Evaluator for PubMedQA."""

    def __init__(
        self,
        use_think: bool = False,
        system_prompt: str | None = None,
        answer_format: AnswerFormat | str = AnswerFormat.XML,
        streaming: bool | None = None,
    ) -> None:
        """Initialize the PubMedQA evaluator.

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
        test_dataset = PubMedQADataset(
            streaming=self.streaming,
        )
        return None, test_dataset.get_verifiers_dataset()


async def main() -> None:  # pragma: no cover
    """Run PubMedQA evaluation with Groq API."""
    config = GroqGenConfig()
    client = AsyncOpenAI(
        api_key=os.getenv(config.api_key_env),
        base_url=config.base_url,
    )

    evaluator = PubMedQAEvaluator()
    results = await evaluator.evaluate(
        client=client,
        model=config.model,
        num_examples=100,
    )

    print(f"PubMedQA Results: {results}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
