"""Tests for Verl hybrid PubHealthBench reward function.

Tests cover the compute_score dispatch function that routes between
MCQ accuracy and LLM-as-judge scoring based on metadata.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from med_reason_evals.verl.rewards.hybrid_pubhealthbench import (
    PUBHEALTHBENCH_JUDGE_TEMPLATE,
    compute_score,
)


class TestComputeScore:
    """Tests for compute_score function."""

    @pytest.mark.asyncio
    async def test_compute_score_mcq_branch(self):
        """MCQ metadata should route to mcq_score."""
        with patch(
            "med_reason_evals.verl.rewards.hybrid_pubhealthbench.mcq_score",
            return_value=1.0,
        ) as mock_mcq:
            result = await compute_score(
                solution_str="<answer>A</answer>",
                ground_truth={"answer": "A"},
                metadata={"is_mcq": True},
                judge_client=MagicMock(),
            )

            assert result == 1.0
            mock_mcq.assert_called_once()

    @pytest.mark.asyncio
    async def test_compute_score_judge_branch(self):
        """Freeform metadata should route to judge_score."""
        with patch(
            "med_reason_evals.verl.rewards.hybrid_pubhealthbench.judge_score",
            new_callable=AsyncMock,
            return_value=0.8,
        ) as mock_judge:
            result = await compute_score(
                solution_str="<answer>Some answer</answer>",
                ground_truth={"target": "Expected"},
                metadata={"is_mcq": False},
                judge_client=MagicMock(),
            )

            assert result == 0.8
            mock_judge.assert_called_once()

    @pytest.mark.asyncio
    async def test_compute_score_default_mcq(self):
        """Empty metadata should default to MCQ scoring."""
        with patch(
            "med_reason_evals.verl.rewards.hybrid_pubhealthbench.mcq_score",
            return_value=0.0,
        ) as mock_mcq:
            result = await compute_score(
                solution_str="<answer>B</answer>",
                ground_truth={"answer": "A"},
                metadata={},
                judge_client=MagicMock(),
            )

            assert result == 0.0
            mock_mcq.assert_called_once()

    def test_judge_template_constant(self):
        """Verify the judge template contains expected placeholders."""
        assert "{prediction}" in PUBHEALTHBENCH_JUDGE_TEMPLATE
        assert "{ground_truth}" in PUBHEALTHBENCH_JUDGE_TEMPLATE
        assert "yes/no" in PUBHEALTHBENCH_JUDGE_TEMPLATE
