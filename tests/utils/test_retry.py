"""Tests for retry utilities."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError, RateLimitError

from med_reason_evals.utils.retry import (
    _log_retry,
    _read_bool_env,
    _read_float_env,
    _read_int_env,
    call_with_retry,
    wrap_openai_call,
)


@pytest.mark.asyncio
async def test_call_with_retry_retries_until_success(monkeypatch: Any) -> None:
    """Retries transient failures before succeeding."""
    attempts = {"count": 0}

    async def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            request = httpx.Request("GET", "https://example.com")
            raise APIConnectionError(message="nope", request=request)
        return "ok"

    monkeypatch.setenv("MED_REASON_RETRY_MAX_ATTEMPTS", "3")

    result = await call_with_retry(flaky)

    assert result == "ok"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_call_with_retry_does_not_retry_non_retryable(
    monkeypatch: Any,
) -> None:
    """Non retryable errors should raise immediately."""
    attempts = {"count": 0}

    async def boom() -> str:
        attempts["count"] += 1
        raise RuntimeError("no")

    monkeypatch.setenv("MED_REASON_RETRY_MAX_ATTEMPTS", "3")

    with pytest.raises(RuntimeError):
        await call_with_retry(boom)

    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_call_with_retry_propagates_retryable_error(monkeypatch: Any) -> None:
    """Retryable errors should be raised after the final attempt."""
    attempts = {"count": 0}

    async def always_rate_limited() -> str:
        attempts["count"] += 1
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(429, request=request)
        raise RateLimitError(
            "no",
            response=response,
            body=None,
        )

    monkeypatch.setenv("MED_REASON_RETRY_MAX_ATTEMPTS", "2")

    with pytest.raises(RateLimitError):
        await call_with_retry(always_rate_limited)

    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_call_with_retry_handles_asyncio_timeout(monkeypatch: Any) -> None:
    """Asyncio timeout errors are treated as retryable."""
    attempts = {"count": 0}

    async def timeout_once() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise asyncio.TimeoutError("timeout")
        return "ok"

    monkeypatch.setenv("MED_REASON_RETRY_MAX_ATTEMPTS", "2")

    result = await call_with_retry(timeout_once)

    assert result == "ok"
    assert attempts["count"] == 2


class TestReadIntEnv:
    """Tests for _read_int_env helper."""

    def test_default_returned(self, monkeypatch: Any) -> None:
        """Default is returned when env var is not set."""
        monkeypatch.delenv("MY_TEST_INT", raising=False)
        assert _read_int_env("MY_TEST_INT", 42) == 42

    def test_valid_int(self, monkeypatch: Any) -> None:
        """Valid integer string is parsed."""
        monkeypatch.setenv("MY_TEST_INT", "10")
        assert _read_int_env("MY_TEST_INT", 42) == 10

    def test_invalid_raises(self, monkeypatch: Any) -> None:
        """Non-integer string raises ValueError."""
        monkeypatch.setenv("MY_TEST_INT", "abc")
        with pytest.raises(ValueError, match="Invalid integer"):
            _read_int_env("MY_TEST_INT", 42)


class TestReadFloatEnv:
    """Tests for _read_float_env helper."""

    def test_default_returned(self, monkeypatch: Any) -> None:
        """Default is returned when env var is not set."""
        monkeypatch.delenv("MY_TEST_FLOAT", raising=False)
        assert _read_float_env("MY_TEST_FLOAT", 3.14) == 3.14

    def test_valid_float(self, monkeypatch: Any) -> None:
        """Valid float string is parsed."""
        monkeypatch.setenv("MY_TEST_FLOAT", "2.5")
        assert _read_float_env("MY_TEST_FLOAT", 3.14) == 2.5

    def test_invalid_raises(self, monkeypatch: Any) -> None:
        """Non-float string raises ValueError."""
        monkeypatch.setenv("MY_TEST_FLOAT", "xyz")
        with pytest.raises(ValueError, match="Invalid float"):
            _read_float_env("MY_TEST_FLOAT", 3.14)


class TestReadBoolEnv:
    """Tests for _read_bool_env helper."""

    def test_default_returned(self, monkeypatch: Any) -> None:
        """Default is returned when env var is not set."""
        monkeypatch.delenv("MY_TEST_BOOL", raising=False)
        assert _read_bool_env("MY_TEST_BOOL", True) is True

    @pytest.mark.parametrize("value", ["1", "true", "yes", "y", "on", "TRUE", "Yes"])
    def test_truthy_values(self, monkeypatch: Any, value: str) -> None:
        """Truthy strings return True."""
        monkeypatch.setenv("MY_TEST_BOOL", value)
        assert _read_bool_env("MY_TEST_BOOL", False) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "n", "off", "FALSE", "No"])
    def test_falsy_values(self, monkeypatch: Any, value: str) -> None:
        """Falsy strings return False."""
        monkeypatch.setenv("MY_TEST_BOOL", value)
        assert _read_bool_env("MY_TEST_BOOL", True) is False

    def test_invalid_raises(self, monkeypatch: Any) -> None:
        """Invalid boolean string raises ValueError."""
        monkeypatch.setenv("MY_TEST_BOOL", "maybe")
        with pytest.raises(ValueError, match="Invalid boolean"):
            _read_bool_env("MY_TEST_BOOL", True)


class TestLogRetry:
    """Tests for _log_retry callback."""

    def test_log_retry_with_none_outcome(self) -> None:
        """Should return early when outcome is None."""
        retry_state = MagicMock()
        retry_state.outcome = None
        # Should not raise
        _log_retry(retry_state)

    def test_log_retry_with_logging_enabled(
        self, monkeypatch: Any, caplog: Any
    ) -> None:
        """Should log a warning when logging is enabled."""
        monkeypatch.setenv("MED_REASON_RETRY_LOG", "true")

        retry_state = MagicMock()
        retry_state.outcome.exception.return_value = RuntimeError("test error")
        retry_state.attempt_number = 2
        retry_state.retry_object.stop.max_attempt_number = 5

        with caplog.at_level(logging.WARNING, logger="med_reason_evals.utils.retry"):
            _log_retry(retry_state)

        assert "Retrying" in caplog.text

    def test_log_retry_with_logging_disabled(
        self, monkeypatch: Any, caplog: Any
    ) -> None:
        """Should not log when logging is disabled."""
        monkeypatch.setenv("MED_REASON_RETRY_LOG", "false")

        retry_state = MagicMock()
        retry_state.outcome.exception.return_value = RuntimeError("test error")

        with caplog.at_level(logging.WARNING, logger="med_reason_evals.utils.retry"):
            _log_retry(retry_state)

        assert "Retrying" not in caplog.text


class TestWrapOpenAICall:
    """Tests for wrap_openai_call decorator."""

    @pytest.mark.asyncio
    async def test_wrap_openai_call_success(self, monkeypatch: Any) -> None:
        """Wrapped function works for successful calls."""
        monkeypatch.setenv("MED_REASON_RETRY_MAX_ATTEMPTS", "2")

        async def my_func(x: int) -> int:
            return x * 2

        wrapped = wrap_openai_call(my_func)
        result = await wrapped(5)

        assert result == 10
