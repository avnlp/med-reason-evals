"""Shared evaluator base classes for verifiers pipelines.

This module defines the foundational classes for building medical evaluators
using the verifiers framework. It provides three base evaluator types:

- BaseVerifierEvaluator: Abstract base with common environment lifecycle
- BaseMCQEvaluator: For multiple-choice datasets with XML/boxed answer parsing
- BaseJudgeEvaluator: For open-ended tasks requiring LLM-as-judge scoring

Each evaluator follows a builder pattern where _load_datasets(),
_build_parser_and_prompt(), and _build_rubric() are implemented by subclasses
to customize dataset loading, answer extraction, and scoring logic.

The module also includes configuration dataclasses for Groq API settings and
judge model parameters, enabling consistent model access across evaluators.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI
from verifiers import JudgeRubric

from med_reason_evals.utils.retry import wrap_openai_call
from med_reason_evals.verifiers.rewards.multiple_choice_accuracy import (
    accuracy_reward,
)
from med_reason_evals.verifiers.utils.parsers import extract_boxed_answer
from med_reason_evals.verifiers.utils.prompts import (
    BOXED_SYSTEM_PROMPT,
    THINK_BOXED_SYSTEM_PROMPT,
    THINK_XML_SYSTEM_PROMPT,
    XML_SYSTEM_PROMPT,
    AnswerFormat,
)


@dataclass
class GroqGenConfig:
    """Default Groq generation configuration."""

    api_key_env: str = "GROQ_API_KEY"
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-120b"
    sampling_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeConfig:
    """Default Groq judge configuration."""

    api_key_env: str = "GROQ_API_KEY"
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-120b"
    sampling_args: dict[str, Any] = field(default_factory=dict)


def make_async_openai_client(*, api_key: str | None, base_url: str) -> AsyncOpenAI:
    """Create an AsyncOpenAI client with the given base URL."""
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def env_dataset_streaming_default() -> bool:
    """Return the default streaming setting for verifiers datasets.

    Verifiers environments require datasets.Dataset (not IterableDataset) because
    Environment.get_dataset relies on len()/shuffle()/select().

    Note: While HuggingFace datasets support streaming for large datasets,
    the verifiers framework needs random access for train/val splitting
    and shuffling operations. This function centralizes the default to
    ensure consistency across all evaluators.
    """
    return False


class BaseVerifierEvaluator(ABC):
    """Base evaluator that builds a verifiers environment."""

    def __init__(self) -> None:
        """Initialize the base evaluator."""
        self._env: vf.Environment | None = None

    @abstractmethod
    def _load_datasets(self) -> tuple[Dataset | None, Dataset]:
        """Return (train_dataset, eval_dataset) for the evaluator."""

    @abstractmethod
    def _build_parser_and_prompt(self) -> tuple[vf.Parser, str | None]:
        """Return the parser and system prompt for the evaluator."""

    @abstractmethod
    def _build_rubric(self, parser: vf.Parser) -> vf.Rubric:
        """Return the rubric for the evaluator."""

    def environment(self) -> vf.Environment:
        """Return the cached verifiers environment.

        Implements lazy initialization pattern - environment is built on first
        access and cached for subsequent calls. This avoids expensive dataset
        loading during evaluator construction.

        Environment construction:
        1. Load train/eval datasets (train may be None for test-only evaluators)
        2. Build parser and select system prompt based on answer format
        3. Construct rubric with reward functions for scoring
        4. Create SingleTurnEnv with all components wired together

        Returns:
            Configured verifiers Environment ready for evaluation
        """
        if self._env is not None:
            return self._env

        train_ds, eval_ds = self._load_datasets()
        parser, system_prompt = self._build_parser_and_prompt()
        rubric = self._build_rubric(parser)

        # SingleTurnEnv handles both train+eval and eval-only configurations
        # train_ds=None indicates this evaluator only supports evaluation (no training)
        if train_ds is None:
            self._env = vf.SingleTurnEnv(
                eval_dataset=eval_ds,
                system_prompt=system_prompt,
                parser=parser,
                rubric=rubric,
            )
        else:
            self._env = vf.SingleTurnEnv(
                dataset=train_ds,
                eval_dataset=eval_ds,
                system_prompt=system_prompt,
                parser=parser,
                rubric=rubric,
            )

        return self._env

    def evaluate(
        self,
        client: AsyncOpenAI,
        model: str,
        num_examples: int,
        **kwargs: Any,
    ) -> Any:
        """Evaluate the model against the evaluator's environment."""
        env = self.environment()
        return env.evaluate(
            client=client,
            model=model,
            num_examples=num_examples,
            **kwargs,
        )


class BaseMCQEvaluator(BaseVerifierEvaluator):
    """Base evaluator for multiple-choice datasets."""

    def __init__(
        self,
        use_think: bool = False,
        system_prompt: str | None = None,
        answer_format: AnswerFormat | str = AnswerFormat.XML,
        streaming: bool | None = None,
    ) -> None:
        """Initialize the MCQ evaluator.

        Args:
            use_think: Whether to require thinking tags in responses.
            system_prompt: Custom system prompt. If None, uses default based on
                answer_format and use_think.
            answer_format: Format for answers (XML or BOXED).
            streaming: Whether to stream dataset loading. Uses default if None.
        """
        super().__init__()
        self.use_think = use_think
        self.system_prompt = system_prompt
        self.answer_format = answer_format
        self.streaming = (
            env_dataset_streaming_default() if streaming is None else streaming
        )

    def _build_parser_and_prompt(self) -> tuple[vf.Parser, str | None]:
        # Normalize answer_format to enum for type safety
        answer_format = (
            AnswerFormat(self.answer_format)
            if isinstance(self.answer_format, str)
            else self.answer_format
        )

        if answer_format == AnswerFormat.XML:
            # XML format: Use vf.XMLParser for structured field extraction
            system_prompt = self.system_prompt or (
                THINK_XML_SYSTEM_PROMPT if self.use_think else XML_SYSTEM_PROMPT
            )
            parser_fields = ["think", "answer"] if self.use_think else ["answer"]
            parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
        elif answer_format == AnswerFormat.BOXED:
            # BOXED format: Use vf.Parser with extract_boxed_answer function
            system_prompt = self.system_prompt or (
                THINK_BOXED_SYSTEM_PROMPT if self.use_think else BOXED_SYSTEM_PROMPT
            )
            # ThinkParser handles both think tags and answer extraction
            parser = (
                vf.ThinkParser(extract_boxed_answer)
                if self.use_think
                else vf.Parser(extract_boxed_answer)
            )
        else:
            raise ValueError(f"Unsupported answer format: {answer_format}")

        return parser, system_prompt

    def _build_rubric(self, parser: vf.Parser) -> vf.Rubric:
        return vf.Rubric(funcs=[accuracy_reward], weights=[1.0], parser=parser)


class BaseJudgeEvaluator(BaseVerifierEvaluator):
    """Base evaluator for judge-based datasets."""

    def __init__(
        self,
        judge_config: JudgeConfig | None = None,
        judge_api_key: str | None = None,
    ) -> None:
        """Initialize the judge evaluator.

        Args:
            judge_config: Configuration for the judge model. Uses default if None.
            judge_api_key: API key for the judge model. Uses env var if None.
        """
        super().__init__()
        self.judge_config = judge_config or JudgeConfig()
        self.judge_api_key = judge_api_key

    @abstractmethod
    def _add_judge_reward_funcs(self, rubric: JudgeRubric, parser: vf.Parser) -> None:
        """Attach reward functions to the judge rubric."""

    def _build_rubric(self, parser: vf.Parser) -> vf.Rubric:
        api_key = self.judge_api_key or os.getenv(self.judge_config.api_key_env)
        judge_client = make_async_openai_client(
            api_key=api_key,
            base_url=self.judge_config.base_url,
        )
        judge_rubric = JudgeRubric(
            judge_client=judge_client,
            judge_model=self.judge_config.model,
            judge_prompt="{question}",
            judge_sampling_args=self.judge_config.sampling_args,
            parser=parser,
        )
        judge_rubric.judge = wrap_openai_call(judge_rubric.judge)
        self._add_judge_reward_funcs(judge_rubric, parser)
        return judge_rubric
