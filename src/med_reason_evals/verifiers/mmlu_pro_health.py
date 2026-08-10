"""MMLUProHealth verifiers evaluator.

Evaluator for MMLU-Pro Health subset, a professionally curated collection
of health-related questions from the MMLU-Pro benchmark. MMLU-Pro improves
upon the original MMLU with better quality control and more challenging distractors.

Default configuration differs from other evaluators:
- use_think=True: Encourages step-by-step reasoning
- answer_format=BOXED: Uses LaTeX boxed notation for answers

This reflects the benchmark's focus on reasoning-heavy medical questions
where chain-of-thought significantly improves performance.

Reference: https://github.com/MMLU-Pro/MMLU-Pro
"""

import asyncio
import os

from datasets import Dataset
from openai import AsyncOpenAI

from med_reason_evals.data.mmlu_pro_health import MMLUProHealthDataset
from med_reason_evals.verifiers.base import BaseMCQEvaluator, GroqGenConfig
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


class MMLUProHealthEvaluator(BaseMCQEvaluator):
    """Evaluator for MMLUProHealth."""

    def __init__(
        self,
        use_think: bool = True,
        system_prompt: str | None = None,
        answer_format: AnswerFormat | str = AnswerFormat.BOXED,
        streaming: bool | None = None,
    ) -> None:
        """Initialize the MMLUProHealth evaluator.

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
        test_dataset = MMLUProHealthDataset(split="test", streaming=self.streaming)
        return None, test_dataset.get_verifiers_dataset()


async def main() -> None:  # pragma: no cover
    """Run MMLUProHealth evaluation with Groq API."""
    config = GroqGenConfig()
    client = AsyncOpenAI(
        api_key=os.getenv(config.api_key_env),
        base_url=config.base_url,
    )

    evaluator = MMLUProHealthEvaluator()
    results = await evaluator.evaluate(
        client=client,
        model=config.model,
        num_examples=100,
    )

    print(f"MMLUProHealth Results: {results}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
