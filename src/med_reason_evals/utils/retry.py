"""Retry utilities for OpenAI-compatible API calls."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, TypeVar

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_random_exponential


ReturnType = TypeVar("ReturnType")
AsyncCallable = Callable[..., Awaitable[ReturnType]]

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 50
DEFAULT_MAX_WAIT = 600.0
DEFAULT_RETRY_LOG = True


def _read_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value}") from exc


def _read_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {value}") from exc


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean for {name}: {value}")


def _retry_settings() -> tuple[int, float]:
    max_attempts = _read_int_env(
        "MED_REASON_RETRY_MAX_ATTEMPTS",
        DEFAULT_MAX_ATTEMPTS,
    )
    max_wait = _read_float_env("MED_REASON_RETRY_MAX_WAIT", DEFAULT_MAX_WAIT)
    return max_attempts, max_wait


def _retry_logging_enabled() -> bool:
    return _read_bool_env("MED_REASON_RETRY_LOG", DEFAULT_RETRY_LOG)


def _retryable_exception_types() -> tuple[type[BaseException], ...]:
    return (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        InternalServerError,
        httpx.TimeoutException,
        asyncio.TimeoutError,
    )


def _log_retry(retry_state: Any) -> None:
    if not _retry_logging_enabled():
        return
    if retry_state.outcome is None:
        return
    exception = retry_state.outcome.exception()
    LOGGER.warning(
        "Retrying OpenAI call after %s (attempt %s/%s).",
        exception,
        retry_state.attempt_number,
        retry_state.retry_object.stop.max_attempt_number,
    )


def retry_openai() -> AsyncRetrying:
    """Return a configured AsyncRetrying instance for OpenAI-compatible calls."""
    max_attempts, max_wait = _retry_settings()
    return AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=1, min=1, max=max_wait),
        retry=retry_if_exception_type(_retryable_exception_types()),
        before_sleep=_log_retry,
        reraise=True,
    )


async def call_with_retry(
    func: AsyncCallable,
    *args: Any,
    **kwargs: Any,
) -> ReturnType:
    """Call an async OpenAI-compatible API function with retries."""
    try:
        async for attempt in retry_openai():
            with attempt:
                return await func(*args, **kwargs)
    except Exception as exc:
        if _retry_logging_enabled():
            LOGGER.error("OpenAI call failed: %s", exc)
        raise
    raise RuntimeError("Retry loop terminated unexpectedly.")


def wrap_openai_call(
    func: Callable[..., Awaitable[ReturnType]],
) -> Callable[..., Awaitable[ReturnType]]:
    """Wrap an async callable with the default OpenAI retry policy."""

    async def _wrapped(*args: Any, **kwargs: Any) -> ReturnType:
        return await call_with_retry(func, *args, **kwargs)

    return _wrapped
