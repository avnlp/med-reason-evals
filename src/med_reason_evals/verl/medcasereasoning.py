"""MedCaseReasoning Verl module.

Provides MedCaseReasoningEvaluator for MedCaseReasoning RL training
with LLM-as-Judge evaluation.
"""

import asyncio
from typing import Any

from datasets import IterableDataset

from med_reason_evals.data.medcasereasoning import MedCaseReasoningDataset
from med_reason_evals.verl.base import BaseJudgeEvaluator, GroqGenConfig, JudgeConfig
from med_reason_evals.verl.rewards.llm_as_judge import compute_score as judge_score


class MedCaseReasoningEvaluator(BaseJudgeEvaluator):
    """MedCaseReasoning evaluator for Verl pipelines.

    Evaluates models on MedCaseReasoning diagnosis prediction using
    LLM-as-Judge for open-ended evaluation.

    Attributes:
        DATASET_NAME: Name of the dataset.
        DEFAULT_SYSTEM_PROMPT: Default system prompt for generation.
        JUDGE_TEMPLATE: Template for judge prompts.
    """

    DATASET_NAME = "medcasereasoning"
    DEFAULT_SYSTEM_PROMPT = (
        "Read the following case presentation and give the most likely diagnosis. "
        "First, provide your internal reasoning within <think>...</think> tags. "
        "Then, output the final diagnosis within <answer>...</answer> tags."
    )

    JUDGE_TEMPLATE = """Is the predicted diagnosis correct (yes/no)?
Predicted diagnosis: {prediction}
True diagnosis: {ground_truth}
Answer [yes/no]."""

    def __init__(
        self,
        split: str = "val",
        gen_config: GroqGenConfig | None = None,
        judge_config: JudgeConfig | None = None,
        system_prompt: str | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the MedCaseReasoning evaluator.

        Args:
            split: Dataset split to use ("train" or "val").
            gen_config: Configuration for generation.
            judge_config: Configuration for judging.
            system_prompt: Optional override for system prompt.
            streaming: Whether to stream the dataset.
        """
        super().__init__(
            gen_config=gen_config,
            judge_config=judge_config,
            streaming=streaming,
        )
        self.split = split
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def _load_dataset(self) -> IterableDataset:
        """Load the MedCaseReasoning dataset.

        Returns:
            IterableDataset formatted for Verl.
        """
        dataset = MedCaseReasoningDataset(
            split=self.split,
            streaming=self.streaming,
        )
        return dataset.get_verl_dataset()

    async def _evaluate_example(
        self,
        prompt: list[dict[str, str]],
        ground_truth: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Evaluate a single example using LLM-as-Judge.

        Args:
            prompt: List of message dicts.
            ground_truth: Dict with 'target' or 'answer' key.
            metadata: Optional metadata (unused).

        Returns:
            Score from 0.0 to 1.0.
        """
        messages = [{"role": "system", "content": self.system_prompt}] + prompt
        completion = await self.rollouts.generate(messages=messages)

        return await judge_score(
            solution_str=completion,
            ground_truth=ground_truth,
            judge_client=self.judge_client,
            judge_model=self.judge_config.model,
            judge_prompt=self.JUDGE_TEMPLATE,
        )

    def _build_result(
        self,
        avg_score: float,
        num_examples: int = 0,
    ) -> dict[str, Any]:
        """Build the evaluation result dictionary.

        Args:
            avg_score: The average score.
            num_examples: Number of successfully evaluated examples.

        Returns:
            Dictionary with dataset, split, num_examples, and avg_score.
        """
        return {
            "dataset": self.DATASET_NAME,
            "split": self.split,
            "num_examples": num_examples,
            "avg_score": avg_score,
        }


async def main() -> None:
    """Run MedCaseReasoning Verl evaluation."""
    evaluator = MedCaseReasoningEvaluator()
    results = await evaluator.evaluate(num_examples=50)
    print(f"\nMedCaseReasoning Verl Results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
