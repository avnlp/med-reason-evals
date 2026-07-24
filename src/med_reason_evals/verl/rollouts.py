"""Groq rollout helper for Verl pipelines.

This module provides a simple interface for generating completions from
Groq-hosted models, designed for use in Verl reward computation pipelines.

The GroqRollouts class wraps Groq's OpenAI-compatible API to provide:
    - Synchronous configuration with async generation methods
    - Batch generation support for efficient evaluation
    - Sensible defaults for medical question-answering tasks

Usage:
    >>> rollouts = GroqRollouts(model="llama-3.3-70b-versatile")
    >>> completion = await rollouts.generate(
    ...     messages=[{"role": "user", "content": "..."}]
    ... )
    >>> completions = await rollouts.generate_batch([messages1, messages2, ...])
"""

import os
from typing import Any

from openai import AsyncOpenAI

from med_reason_evals.utils.retry import call_with_retry


class GroqRollouts:
    """Helper class for generating rollouts using Groq's OpenAI-compatible API.

    This class provides a simple interface for generating completions from
    Groq-hosted models, designed for use in Verl reward computation pipelines.
    """

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the Groq rollouts helper.

        Args:
            model: The model to use for generation. Defaults to llama-3.3-70b-versatile.
            api_key: The Groq API key. Defaults to GROQ_API_KEY env var.
            base_url: The API base URL. Defaults to Groq's OpenAI-compatible endpoint.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional arguments passed to the AsyncOpenAI client.
        """
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set and no api_key provided."
            )

        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,
            **kwargs,
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a completion for the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Override the default model for this request.
            max_tokens: Override the default max_tokens for this request.
            temperature: Override the default temperature for this request.
            **kwargs: Additional arguments passed to the API call.

        Returns:
            The generated completion text.
        """
        response = await call_with_retry(
            self._client.chat.completions.create,
            model=model or self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            **kwargs,
        )

        if not response.choices:
            raise RuntimeError(
                "Groq API returned an empty choices list. "
                "This may indicate a provider or protocol failure."
            )
        return response.choices[0].message.content or ""

    async def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_concurrency: int = 10,
        **kwargs: Any,
    ) -> list[str]:
        """Generate completions for a batch of message lists.

        Args:
            messages_batch: List of message lists.
            model: Override the default model for this request.
            max_tokens: Override the default max_tokens for this request.
            temperature: Override the default temperature for this request.
            max_concurrency: Maximum concurrent API requests.
            **kwargs: Additional arguments passed to each API call.

        Returns:
            List of generated completion texts.

        Raises:
            RuntimeError: If any generation fails; pending tasks are cancelled.
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_generate(messages: list[dict[str, str]]) -> str:
            async with semaphore:
                return await self.generate(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )

        tasks = [
            asyncio.create_task(_bounded_generate(messages))
            for messages in messages_batch
        ]

        try:
            return await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


def get_default_rollouts() -> GroqRollouts:
    """Get a default GroqRollouts instance.

    Returns:
        A GroqRollouts instance configured with environment defaults.
    """
    return GroqRollouts()
