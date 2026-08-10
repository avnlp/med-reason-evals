"""HealthBench verifiers evaluator.

Evaluator for HealthBench, a multi-criteria evaluation framework for
health-related question answering. Unlike binary correct/incorrect scoring,
HealthBench uses detailed rubrics with multiple criteria worth different points.

Key features:
- Multi-criteria rubric evaluation with weighted scoring
- Parallel judge evaluation with semaphore-controlled concurrency
- Criteria extracted from dataset metadata (criteria, points_list fields)
- Returns normalized score 0.0-1.0 based on points earned vs. possible

Reference: https://github.com/openai/healthbench (or similar)
"""

import asyncio
import os

import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI
from verifiers import JudgeRubric

from med_reason_evals.data.healthbench import HealthBenchDataset
from med_reason_evals.verifiers.base import (
    BaseJudgeEvaluator,
    GroqGenConfig,
    JudgeConfig,
    env_dataset_streaming_default,
)
from med_reason_evals.verifiers.rewards.judge_rubric import healthbench_rubric_reward


class HealthBenchEvaluator(BaseJudgeEvaluator):
    """Evaluator for HealthBench."""

    def __init__(
        self,
        difficulty: str = "regular",
        max_parallel_judges: int = 5,
        streaming: bool | None = None,
        judge_config: JudgeConfig | None = None,
        judge_api_key: str | None = None,
    ) -> None:
        """Initialize the HealthBench evaluator.

        Args:
            difficulty: Difficulty level of the questions ("regular" or other).
            max_parallel_judges: Maximum number of parallel judge evaluations.
            streaming: Whether to use streaming mode for dataset loading.
                Defaults to False if not specified.
            judge_config: Configuration for the judge model.
            judge_api_key: API key for the judge model.
        """
        super().__init__(judge_config=judge_config, judge_api_key=judge_api_key)
        self.difficulty = difficulty
        self.max_parallel_judges = max_parallel_judges
        self.streaming = (
            env_dataset_streaming_default() if streaming is None else streaming
        )

    def _load_datasets(self) -> tuple[Dataset | None, Dataset]:
        dataset = HealthBenchDataset(
            difficulty=self.difficulty,
            streaming=self.streaming,
        )
        return None, dataset.get_verifiers_dataset()

    def _build_parser_and_prompt(self) -> tuple[vf.Parser, str | None]:
        return vf.Parser(), ""

    def _add_judge_reward_funcs(self, rubric: JudgeRubric, parser: vf.Parser) -> None:
        async def reward_func(
            prompt,
            completion,
            info,
            state,
            judge,
            **kwargs,
        ) -> float:
            return await healthbench_rubric_reward(
                prompt=prompt,
                completion=completion,
                info=info,
                state=state,
                judge=judge,
                max_parallel_judges=self.max_parallel_judges,
                **kwargs,
            )

        rubric.add_reward_func(reward_func, weight=1.0)


async def main() -> None:  # pragma: no cover
    """Run HealthBench evaluation with Groq API."""
    config = GroqGenConfig()
    client = AsyncOpenAI(
        api_key=os.getenv(config.api_key_env),
        base_url=config.base_url,
    )

    evaluator = HealthBenchEvaluator()
    results = await evaluator.evaluate(
        client=client,
        model=config.model,
        num_examples=20,
    )

    print(f"HealthBench Results: {results}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
