"""MedCaseReasoning verifiers evaluator.

Evaluator for clinical case reasoning tasks where models must diagnose
medical conditions from case presentations. Unlike MCQ tasks, this
requires free-form diagnosis with LLM-as-judge evaluation.

Models receive detailed case histories and must output:
1. Reasoning within <think>...</think> tags
2. Final diagnosis within <answer>...</answer> tags

The judge compares predicted vs. true diagnoses using a specialized
template that accounts for medical terminology variations.

This is the most complex evaluation type, requiring both reasoning
capability and medical domain knowledge.
"""

import asyncio
import os

import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI
from verifiers import JudgeRubric

from med_reason_evals.data.medcasereasoning import MedCaseReasoningDataset
from med_reason_evals.verifiers.base import BaseJudgeEvaluator, GroqGenConfig
from med_reason_evals.verifiers.rewards.llm_as_judge import (
    MEDCASEREASONING_JUDGE_TEMPLATE,
    make_binary_judge_reward,
)


SYSTEM_PROMPT = (
    "Read the following case presentation and give the most likely diagnosis. "
    "First, provide your internal reasoning for the diagnosis within the tags <think> ... </think>. "
    "Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>."
)


class MedCaseReasoningEvaluator(BaseJudgeEvaluator):
    """Evaluator for MedCaseReasoning."""

    def _load_datasets(self) -> tuple[Dataset | None, Dataset]:
        train_dataset = MedCaseReasoningDataset(split="train", streaming=False)
        val_dataset = MedCaseReasoningDataset(split="val", streaming=False)
        return (
            train_dataset.get_verifiers_dataset(),
            val_dataset.get_verifiers_dataset(),
        )

    def _build_parser_and_prompt(self) -> tuple[vf.Parser, str | None]:
        parser = vf.XMLParser(fields=["think", "answer"], answer_field="answer")
        return parser, SYSTEM_PROMPT

    def _add_judge_reward_funcs(self, rubric: JudgeRubric, parser: vf.Parser) -> None:
        reward_func = make_binary_judge_reward(MEDCASEREASONING_JUDGE_TEMPLATE)
        rubric.add_reward_func(reward_func, weight=1.0)


async def main() -> None:
    """Run MedCaseReasoning evaluation with Groq API."""
    config = GroqGenConfig()
    client = AsyncOpenAI(
        api_key=os.getenv(config.api_key_env),
        base_url=config.base_url,
    )

    evaluator = MedCaseReasoningEvaluator()
    results = await evaluator.evaluate(
        client=client,
        model=config.model,
        num_examples=50,
    )

    print(f"MedCaseReasoning Results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
