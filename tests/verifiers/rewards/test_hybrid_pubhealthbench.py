"""Tests for verifiers hybrid PubHealthBench reward function.

Tests cover the pubhealthbench_reward dispatch function that routes
between MCQ accuracy and LLM-as-judge scoring based on question type.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from med_reason_evals.verifiers.rewards.hybrid_pubhealthbench import (
    pubhealthbench_reward,
)


class TestPubHealthBenchReward:
    """Tests for pubhealthbench_reward function."""

    @pytest.mark.asyncio
    async def test_mcq_branch_routes_to_accuracy_reward(self):
        """MCQ questions should route to accuracy_reward."""
        with patch(
            "med_reason_evals.verifiers.rewards.hybrid_pubhealthbench.accuracy_reward",
            new_callable=AsyncMock,
            return_value=1.0,
        ) as mock_accuracy:
            result = await pubhealthbench_reward(
                completion="A",
                answer="A",
                info={"is_mcq": True},
                state=MagicMock(),
                parser=MagicMock(),
                judge=MagicMock(),
            )

            assert result == 1.0
            mock_accuracy.assert_called_once()

    @pytest.mark.asyncio
    async def test_judge_branch_routes_to_binary_judge(self):
        """Freeform questions should route to binary_judge_reward_from_template."""
        with patch(
            "med_reason_evals.verifiers.rewards.hybrid_pubhealthbench.binary_judge_reward_from_template",
            new_callable=AsyncMock,
            return_value=0.5,
        ) as mock_judge:
            result = await pubhealthbench_reward(
                completion="Some text",
                answer="Expected answer",
                info={"is_mcq": False},
                state=MagicMock(),
                parser=MagicMock(),
                judge=MagicMock(),
            )

            assert result == 0.5
            mock_judge.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_to_mcq_when_info_missing(self):
        """Should default to MCQ when info is None."""
        with patch(
            "med_reason_evals.verifiers.rewards.hybrid_pubhealthbench.accuracy_reward",
            new_callable=AsyncMock,
            return_value=1.0,
        ) as mock_accuracy:
            result = await pubhealthbench_reward(
                completion="A",
                answer="A",
                info=None,
                state=MagicMock(),
                parser=MagicMock(),
                judge=MagicMock(),
            )

            assert result == 1.0
            mock_accuracy.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_to_mcq_when_is_mcq_not_in_info(self):
        """Should default to MCQ when is_mcq not present in info dict."""
        with patch(
            "med_reason_evals.verifiers.rewards.hybrid_pubhealthbench.accuracy_reward",
            new_callable=AsyncMock,
            return_value=1.0,
        ) as mock_accuracy:
            result = await pubhealthbench_reward(
                completion="A",
                answer="A",
                info={},
                state=MagicMock(),
                parser=MagicMock(),
                judge=MagicMock(),
            )

            assert result == 1.0
            mock_accuracy.assert_called_once()
