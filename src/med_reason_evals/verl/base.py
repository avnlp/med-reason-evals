"""Shared evaluator base classes for Verl pipelines.

This module provides base classes for Verl evaluators that use Groq rollouts
for generation and reward functions for scoring.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from datasets import IterableDataset
from openai import AsyncOpenAI

from med_reason_evals.verl.rollouts import GroqRollouts


@dataclass
class GroqGenConfig:
    """Groq generation configuration.

    Attributes:
        api_key_env: Environment variable name for the API key.
        base_url: The API base URL.
        model: The model to use for generation.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        sampling_args: Additional sampling arguments.
    """

    api_key_env: str = "GROQ_API_KEY"
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-120b"
    max_tokens: int = 2048
    temperature: float = 0.0
    sampling_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeConfig:
    """Groq judge configuration.

    Attributes:
        api_key_env: Environment variable name for the API key.
        base_url: The API base URL.
        model: The model to use for judging.
        max_tokens: Maximum tokens for judge responses.
        temperature: Sampling temperature.
        sampling_args: Additional sampling arguments.
    """

    api_key_env: str = "GROQ_API_KEY"
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-120b"
    max_tokens: int = 500
    temperature: float = 0.0
    sampling_args: dict[str, Any] = field(default_factory=dict)


class BaseVerlEvaluator(ABC):
    """Base evaluator for Verl pipelines.

    This class provides the foundation for all Verl evaluators, handling
    dataset loading, generation via GroqRollouts, and evaluation loops.

    Attributes:
        gen_config: Configuration for generation.
        streaming: Whether to stream the dataset.
    """

    def __init__(
        self,
        gen_config: GroqGenConfig | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the evaluator.

        Args:
            gen_config: Configuration for generation. Uses defaults if None.
            streaming: Whether to stream the dataset.
        """
        self.gen_config = gen_config or GroqGenConfig()
        self.streaming = streaming
        self._rollouts: GroqRollouts | None = None

    @property
    def rollouts(self) -> GroqRollouts:
        """Lazy-initialized GroqRollouts instance.

        Returns:
            A GroqRollouts instance configured with gen_config settings.

        Raises:
            ValueError: If the API key environment variable is not set.
        """
        if self._rollouts is None:
            api_key = os.getenv(self.gen_config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"Environment variable {self.gen_config.api_key_env} is not set"
                )
            self._rollouts = GroqRollouts(
                model=self.gen_config.model,
                api_key=api_key,
                base_url=self.gen_config.base_url,
                max_tokens=self.gen_config.max_tokens,
                temperature=self.gen_config.temperature,
                **self.gen_config.sampling_args,
            )
        return self._rollouts

    @abstractmethod
    def _load_dataset(self) -> IterableDataset:
        """Return the dataset for evaluation.

        Returns:
            An IterableDataset yielding examples with 'prompt', 'ground_truth',
            and optionally 'metadata' keys.
        """

    @abstractmethod
    async def _evaluate_example(
        self,
        prompt: list[dict[str, str]],
        ground_truth: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Evaluate a single example and return score.

        Args:
            prompt: List of message dicts with 'role' and 'content' keys.
            ground_truth: Dict containing the correct answer/target.
            metadata: Optional metadata about the example.

        Returns:
            A score between 0.0 and 1.0.
        """

    async def evaluate(
        self,
        num_examples: int = 100,
        progress_interval: int = 10,
    ) -> dict[str, Any]:
        """Run evaluation on the dataset.

        Iterates through the dataset, evaluates each example, and returns
        aggregated results.

        Args:
            num_examples: Maximum number of examples to evaluate.
            progress_interval: Print progress every N examples.

        Returns:
            A dictionary with evaluation results including:
            - dataset: Dataset name
            - num_examples: Number of examples evaluated
            - avg_score: Average score across all examples
        """
        dataset = self._load_dataset()
        scores: list[float] = []
        count = 0

        for example in dataset:
            if count >= num_examples:
                break

            prompt = example.get("prompt", [])
            ground_truth = example.get("ground_truth", {})
            metadata = example.get("metadata", {})

            if not prompt or not ground_truth:
                continue

            score = await self._evaluate_example(prompt, ground_truth, metadata)
            scores.append(score)
            count += 1

            if count % progress_interval == 0:
                avg = sum(scores) / len(scores) if scores else 0
                print(f"Processed {count} examples, avg score: {avg:.3f}")

        avg_score = sum(scores) / len(scores) if scores else 0
        return self._build_result(scores, avg_score)

    @abstractmethod
    def _build_result(
        self,
        scores: list[float],
        avg_score: float,
    ) -> dict[str, Any]:
        """Build the result dictionary.

        Args:
            scores: List of all individual scores.
            avg_score: The average score.

        Returns:
            A dictionary with standardized result fields.
        """


class BaseMCQEvaluator(BaseVerlEvaluator):
    """Base evaluator for multiple-choice datasets.

    Provides a standard implementation for MCQ evaluation that:
    1. Generates completions with an optional system prompt
    2. Scores using the MCQ accuracy reward function
    """

    def __init__(
        self,
        gen_config: GroqGenConfig | None = None,
        system_prompt: str | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the MCQ evaluator.

        Args:
            gen_config: Configuration for generation.
            system_prompt: Optional system prompt to prepend to messages.
            streaming: Whether to stream the dataset.
        """
        super().__init__(gen_config, streaming)
        self.system_prompt = system_prompt

    async def _evaluate_example(
        self,
        prompt: list[dict[str, str]],
        ground_truth: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Evaluate a single MCQ example.

        Generates a completion and scores it using MCQ accuracy.

        Args:
            prompt: List of message dicts.
            ground_truth: Dict with 'answer' key containing correct answer.
            metadata: Optional metadata (unused for MCQ).

        Returns:
            Score from 0.0 to 1.0.
        """
        from med_reason_evals.verl.rewards.multiple_choice_accuracy import (
            compute_score as mcq_score,
        )

        messages = self._build_messages(prompt)
        completion = await self.rollouts.generate(messages=messages)
        return mcq_score(completion, ground_truth)

    def _build_messages(self, prompt: list[dict[str, str]]) -> list[dict[str, str]]:
        """Build message list with optional system prompt.

        Args:
            prompt: The user prompt messages.

        Returns:
            Messages with system prompt prepended if configured.
        """
        if self.system_prompt:
            return [{"role": "system", "content": self.system_prompt}] + prompt
        return prompt


class BaseJudgeEvaluator(BaseVerlEvaluator):
    """Base evaluator for LLM-as-Judge datasets.

    Provides infrastructure for datasets that require LLM-based evaluation,
    including a configured judge client.
    """

    def __init__(
        self,
        gen_config: GroqGenConfig | None = None,
        judge_config: JudgeConfig | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the judge evaluator.

        Args:
            gen_config: Configuration for generation.
            judge_config: Configuration for judging. Uses defaults if None.
            streaming: Whether to stream the dataset.
        """
        super().__init__(gen_config, streaming)
        self.judge_config = judge_config or JudgeConfig()
        self._judge_client: AsyncOpenAI | None = None

    @property
    def judge_client(self) -> AsyncOpenAI:
        """Lazy-initialized judge client.

        Returns:
            An AsyncOpenAI client configured for judging.

        Raises:
            ValueError: If the API key environment variable is not set.
        """
        if self._judge_client is None:
            api_key = os.getenv(self.judge_config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"Environment variable {self.judge_config.api_key_env} is not set"
                )
            self._judge_client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.judge_config.base_url,
            )
        return self._judge_client
