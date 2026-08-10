"""PubHealthBench verifiers evaluator.

Evaluator for PubHealthBench, a hybrid dataset with both multiple-choice
and free-form questions about public health topics. This requires a unique
reward function that switches between MCQ accuracy and LLM-as-judge based
on question type.

The hybrid approach:
- MCQ questions: Use accuracy_reward for letter/number matching
- Free-form questions: Use binary_judge_reward_from_template for open evaluation

Question type is determined at runtime via info["is_mcq"] metadata from
the dataset, allowing a single evaluator to handle heterogeneous formats.

This demonstrates advanced reward function composition patterns.
"""

import asyncio
import os

import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI
from verifiers import JudgeRubric

from med_reason_evals.data.pubhealthbench import PubHealthBenchDataset
from med_reason_evals.verifiers.base import (
    BaseJudgeEvaluator,
    GroqGenConfig,
    env_dataset_streaming_default,
)
from med_reason_evals.verifiers.rewards.hybrid_pubhealthbench import (
    pubhealthbench_reward,
)
from med_reason_evals.verifiers.utils.prompts import (
    THINK_XML_SYSTEM_PROMPT,
    XML_SYSTEM_PROMPT,
    AnswerFormat,
)


class PubHealthBenchEvaluator(BaseJudgeEvaluator):
    """Evaluator for PubHealthBench."""

    def __init__(
        self,
        question_type: str = "all",
        use_think: bool = False,
        system_prompt: str | None = None,
        answer_format: AnswerFormat | str = AnswerFormat.XML,
        streaming: bool | None = None,
        judge_config=None,
        judge_api_key: str | None = None,
    ) -> None:
        """Initialize the PubHealthBench evaluator.

        Args:
            question_type: Type of questions to load ("all" or specific type).
            use_think: Whether to include thinking tags in the output.
            system_prompt: Custom system prompt to use. If None, uses default.
            answer_format: Format for answers (only XML supported).
            streaming: Whether to use streaming mode for dataset loading.
                Defaults to False if not specified.
            judge_config: Configuration for the judge model.
            judge_api_key: API key for the judge model.
        """
        super().__init__(judge_config=judge_config, judge_api_key=judge_api_key)
        self.question_type = question_type
        self.use_think = use_think
        self.system_prompt = system_prompt
        self.answer_format = answer_format
        self.streaming = (
            env_dataset_streaming_default() if streaming is None else streaming
        )

    def _load_datasets(self) -> tuple[Dataset | None, Dataset]:
        test_dataset = PubHealthBenchDataset(
            split="test",
            streaming=self.streaming,
            question_type=self.question_type,
        )
        return None, test_dataset.get_verifiers_dataset()

    def _build_parser_and_prompt(self) -> tuple[vf.Parser, str | None]:
        answer_format = (
            AnswerFormat(self.answer_format)
            if isinstance(self.answer_format, str)
            else self.answer_format
        )
        if answer_format != AnswerFormat.XML:
            raise ValueError("PubHealthBench only supports XML answer format.")
        system_prompt = (
            self.system_prompt
            if self.system_prompt is not None
            else (THINK_XML_SYSTEM_PROMPT if self.use_think else XML_SYSTEM_PROMPT)
        )
        parser_fields = ["think", "answer"] if self.use_think else ["answer"]
        parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
        return parser, system_prompt

    def _add_judge_reward_funcs(self, rubric: JudgeRubric, parser: vf.Parser) -> None:
        rubric.add_reward_func(pubhealthbench_reward, weight=1.0)


async def main() -> None:
    """Run PubHealthBench evaluation with Groq API."""
    config = GroqGenConfig()
    client = AsyncOpenAI(
        api_key=os.getenv(config.api_key_env),
        base_url=config.base_url,
    )

    evaluator = PubHealthBenchEvaluator()
    results = await evaluator.evaluate(
        client=client,
        model=config.model,
        num_examples=100,
    )

    print(f"PubHealthBench Results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
