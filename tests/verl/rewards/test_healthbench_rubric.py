"""Tests for Verl HealthBench rubric reward function.

This module tests the healthbench_rubric reward functions used in verl for
rubric-based evaluation with LLM judges. Tests cover:
    - evaluate_criterion: Single criterion evaluation with semaphore concurrency
    - compute_score: Multi-criteria scoring with normalization and clamping
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from med_reason_evals.verl.rewards.healthbench_rubric import (
    RUBRIC_JUDGE_TEMPLATE,
    compute_score,
    evaluate_criterion,
)


class TestRubricJudgeTemplate:
    """Tests for RUBRIC_JUDGE_TEMPLATE constant."""

    def test_template_has_response_placeholder(self):
        """Template should have {response} placeholder."""
        assert "{response}" in RUBRIC_JUDGE_TEMPLATE

    def test_template_has_criterion_placeholder(self):
        """Template should have {criterion} placeholder."""
        assert "{criterion}" in RUBRIC_JUDGE_TEMPLATE

    def test_template_has_points_placeholder(self):
        """Template should have {points} placeholder."""
        assert "{points}" in RUBRIC_JUDGE_TEMPLATE

    def test_template_can_be_formatted(self):
        """Template should be formattable without errors."""
        result = RUBRIC_JUDGE_TEMPLATE.format(
            response="The patient has diabetes.",
            criterion="Mentions diabetes diagnosis",
            points=5,
            conversation="What is the diagnosis?",
        )
        assert "The patient has diabetes." in result
        assert "[5] Mentions diabetes diagnosis" in result
        assert "What is the diagnosis?" in result

    def test_template_requests_json_format(self):
        """Template should instruct to return JSON."""
        assert "JSON" in RUBRIC_JUDGE_TEMPLATE

    def test_template_mentions_criteria_met(self):
        """Template should mention criteria_met field."""
        assert "criteria_met" in RUBRIC_JUDGE_TEMPLATE

    def test_template_mentions_explanation(self):
        """Template should mention explanation field."""
        assert "explanation" in RUBRIC_JUDGE_TEMPLATE


class TestEvaluateCriterion:
    """Tests for evaluate_criterion function."""

    @pytest.mark.asyncio
    async def test_evaluates_criterion_with_true_response(self):
        """Should return criteria_met=True when judge confirms."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = '{"criteria_met": true, "explanation": "Good answer"}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        semaphore = asyncio.Semaphore(1)
        result = await evaluate_criterion(
            response="The diagnosis is diabetes.",
            criterion="Mentions diabetes",
            points=5,
            judge_client=mock_client,
            judge_model="test-model",
            semaphore=semaphore,
        )

        assert result["criteria_met"] is True
        assert result["points"] == 5
        assert result["explanation"] == "Good answer"

    @pytest.mark.asyncio
    async def test_evaluates_criterion_with_false_response(self):
        """Should return criteria_met=False when judge rejects."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = (
            '{"criteria_met": false, "explanation": "Missing information"}'
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        semaphore = asyncio.Semaphore(1)
        result = await evaluate_criterion(
            response="The patient is fine.",
            criterion="Mentions diabetes",
            points=5,
            judge_client=mock_client,
            judge_model="test-model",
            semaphore=semaphore,
        )

        assert result["criteria_met"] is False
        assert result["points"] == 5
        assert result["explanation"] == "Missing information"

    @pytest.mark.asyncio
    async def test_handles_markdown_json_response(self):
        """Should parse JSON wrapped in markdown code blocks."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = (
            '```json\n{"criteria_met": true, "explanation": "Correct"}\n```'
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        semaphore = asyncio.Semaphore(1)
        result = await evaluate_criterion(
            response="Answer",
            criterion="Check",
            points=3,
            judge_client=mock_client,
            judge_model="test-model",
            semaphore=semaphore,
        )

        assert result["criteria_met"] is True
        assert result["explanation"] == "Correct"

    @pytest.mark.asyncio
    async def test_handles_unparsable_response(self):
        """Should default to criteria_met=False when response is unparsable."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not JSON"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        semaphore = asyncio.Semaphore(1)
        result = await evaluate_criterion(
            response="Answer",
            criterion="Check",
            points=3,
            judge_client=mock_client,
            judge_model="test-model",
            semaphore=semaphore,
        )

        assert result["criteria_met"] is False
        assert result["points"] == 3
        assert result["explanation"] == ""

    @pytest.mark.asyncio
    async def test_handles_missing_criteria_met_field(self):
        """Should default to False when criteria_met field is missing."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"explanation": "Some explanation"}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        semaphore = asyncio.Semaphore(1)
        result = await evaluate_criterion(
            response="Answer",
            criterion="Check",
            points=3,
            judge_client=mock_client,
            judge_model="test-model",
            semaphore=semaphore,
        )

        assert result["criteria_met"] is False
        assert result["explanation"] == "Some explanation"

    @pytest.mark.asyncio
    async def test_uses_correct_model_and_parameters(self):
        """Should call API with correct model and parameters."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": true}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        semaphore = asyncio.Semaphore(1)
        await evaluate_criterion(
            response="The answer",
            criterion="The criterion",
            points=10,
            judge_client=mock_client,
            judge_model="custom-model",
            semaphore=semaphore,
            max_tokens=1000,
            temperature=0.5,
        )

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "custom-model"
        assert call_args.kwargs["max_tokens"] == 1000
        assert call_args.kwargs["temperature"] == 0.5
        assert len(call_args.kwargs["messages"]) == 1
        assert call_args.kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_prompt_contains_response_and_criterion(self):
        """Prompt should contain response text and criterion."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": true}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        semaphore = asyncio.Semaphore(1)
        await evaluate_criterion(
            response="Patient has type 2 diabetes",
            criterion="Identifies diabetes type",
            points=5,
            judge_client=mock_client,
            judge_model="test-model",
            semaphore=semaphore,
        )

        call_args = mock_client.chat.completions.create.call_args
        prompt_content = call_args.kwargs["messages"][0]["content"]
        assert "Patient has type 2 diabetes" in prompt_content
        assert "Identifies diabetes type" in prompt_content
        assert "[5]" in prompt_content

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Semaphore should limit concurrent API calls."""
        mock_client = MagicMock()
        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def delayed_response(*args, **kwargs):
            nonlocal active_count, max_active
            async with lock:
                active_count += 1
                max_active = max(max_active, active_count)
            await asyncio.sleep(0.05)  # Simulate API delay
            async with lock:
                active_count -= 1
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"criteria_met": true}'
            return mock_response

        mock_client.chat.completions.create = AsyncMock(side_effect=delayed_response)

        semaphore = asyncio.Semaphore(2)  # Allow max 2 concurrent

        # Start 5 concurrent evaluations
        tasks = [
            evaluate_criterion(
                response=f"Answer {i}",
                criterion=f"Criterion {i}",
                points=1,
                judge_client=mock_client,
                judge_model="test-model",
                semaphore=semaphore,
            )
            for i in range(5)
        ]

        await asyncio.gather(*tasks)

        assert max_active <= 2

    @pytest.mark.asyncio
    async def test_handles_empty_response_content(self):
        """Should handle empty response content gracefully."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        semaphore = asyncio.Semaphore(1)
        result = await evaluate_criterion(
            response="Answer",
            criterion="Check",
            points=3,
            judge_client=mock_client,
            judge_model="test-model",
            semaphore=semaphore,
        )

        assert result["criteria_met"] is False
        assert result["points"] == 3


class TestComputeScore:
    """Tests for compute_score function."""

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_criteria(self):
        """Should return 0.0 when criteria list is empty."""
        mock_client = MagicMock()
        ground_truth: dict[str, Any] = {"criteria": [], "points_list": []}

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_points_list(self):
        """Should return 0.0 when points_list is missing."""
        mock_client = MagicMock()
        ground_truth: dict[str, Any] = {"criteria": ["Criterion 1"]}

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_returns_zero_when_criteria_missing(self):
        """Should return 0.0 when criteria key is missing."""
        mock_client = MagicMock()
        ground_truth: dict[str, Any] = {"points_list": [1, 2, 3]}

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_raises_when_mismatched_lengths(self):
        """Should raise ValueError when criteria and points_list lengths differ."""
        mock_client = MagicMock()
        ground_truth: dict[str, Any] = {
            "criteria": ["Crit1", "Crit2", "Crit3"],
            "points_list": [1, 2],  # Mismatched length
        }

        with pytest.raises(ValueError, match="length mismatch"):
            await compute_score(
                solution_str="Answer",
                ground_truth=ground_truth,
                judge_client=mock_client,
                judge_model="test-model",
            )

    @pytest.mark.asyncio
    async def test_returns_zero_when_all_points_are_zero_or_negative(self):
        """Should return 0.0 when total positive points is 0."""
        mock_client = MagicMock()
        ground_truth: dict[str, Any] = {
            "criteria": ["Crit1", "Crit2"],
            "points_list": [0, -5],  # No positive points
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_all_criteria_pass_perfect_score(self):
        """Should return 1.0 when all criteria are met."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": true}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        ground_truth: dict[str, Any] = {
            "criteria": ["Crit1", "Crit2", "Crit3"],
            "points_list": [1, 2, 3],  # Total = 6
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        # All pass: earned = 6, total = 6, score = 1.0
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_partial_criteria_pass(self):
        """Should calculate correct score for partial pass."""
        mock_client = MagicMock()
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            # First and third pass, second fails
            if call_count == 2:
                mock_response.choices[0].message.content = '{"criteria_met": false}'
            else:
                mock_response.choices[0].message.content = '{"criteria_met": true}'
            return mock_response

        mock_client.chat.completions.create = AsyncMock(side_effect=side_effect)

        ground_truth: dict[str, Any] = {
            "criteria": ["Crit1", "Crit2", "Crit3"],
            "points_list": [2, 3, 5],  # Total = 10
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        # Earned: 2 + 0 + 5 = 7, total = 10, score = 0.7
        assert result == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_all_criteria_fail_zero_score(self):
        """Should return 0.0 when all criteria fail."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": false}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        ground_truth: dict[str, Any] = {
            "criteria": ["Crit1", "Crit2"],
            "points_list": [5, 5],  # Total = 10
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_negative_points_as_penalties(self):
        """Negative points should be included in earned calculation."""
        mock_client = MagicMock()
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            # First fails, second (penalty) is met
            if call_count == 1:
                mock_response.choices[0].message.content = '{"criteria_met": false}'
            else:
                mock_response.choices[0].message.content = '{"criteria_met": true}'
            return mock_response

        mock_client.chat.completions.create = AsyncMock(side_effect=side_effect)

        ground_truth: dict[str, Any] = {
            "criteria": ["Must mention X", "Must not mention Y"],
            "points_list": [5, -2],  # -2 is a penalty
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        # Earned: 0 + (-2) = -2, total_positive = 5
        # Score = -2/5 = -0.4, clamped to 0.0
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_score_clamped_to_zero(self):
        """Negative scores should be clamped to 0.0."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": true}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        ground_truth: dict[str, Any] = {
            "criteria": ["Positive", "Penalty"],
            "points_list": [2, -5],
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        # Earned: 2 + (-5) = -3, total_positive = 2
        # Score = -3/2 = -1.5, clamped to 0.0
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_score_always_within_bounds(self):
        """Score should always be within [0.0, 1.0]."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": true}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        ground_truth: dict[str, Any] = {
            "criteria": ["Crit1", "Crit2"],
            "points_list": [5, 3],
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        assert 0.0 <= result <= 1.0

    @pytest.mark.asyncio
    async def test_normalization_uses_positive_points_only(self):
        """Score normalization should only use positive points."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": true}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        ground_truth: dict[str, Any] = {
            "criteria": ["Positive 1", "Negative", "Positive 2"],
            "points_list": [3, -2, 2],  # Total positive = 5
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        # All met: earned = 3 + (-2) + 2 = 3, total_positive = 5
        # Score = 3/5 = 0.6
        assert result == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_semaphore_limits_parallel_judges(self):
        """Semaphore should limit concurrent judge requests."""
        mock_client = MagicMock()
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracking_response(*args, **kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.02)  # Small delay to allow overlap detection
            async with lock:
                current_concurrent -= 1
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"criteria_met": true}'
            return mock_response

        mock_client.chat.completions.create = AsyncMock(side_effect=tracking_response)

        ground_truth: dict[str, Any] = {
            "criteria": [f"Criterion {i}" for i in range(10)],
            "points_list": [1] * 10,
        }

        await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
            max_parallel_judges=3,
        )

        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_default_max_parallel_judges(self):
        """Default max_parallel_judges should be 5."""
        mock_client = MagicMock()
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracking_response(*args, **kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.02)
            async with lock:
                current_concurrent -= 1
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"criteria_met": true}'
            return mock_response

        mock_client.chat.completions.create = AsyncMock(side_effect=tracking_response)

        ground_truth: dict[str, Any] = {
            "criteria": [f"Criterion {i}" for i in range(15)],
            "points_list": [1] * 15,
        }

        await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        assert max_concurrent <= 5

    @pytest.mark.asyncio
    async def test_custom_judge_model(self):
        """Should use custom judge model when specified."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": true}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        ground_truth: dict[str, Any] = {
            "criteria": ["Crit1"],
            "points_list": [5],
        }

        await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="custom-judge-model",
            max_tokens=1000,
            temperature=0.5,
        )

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "custom-judge-model"
        assert call_args.kwargs["max_tokens"] == 1000
        assert call_args.kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_single_criterion(self):
        """Single criterion should work correctly."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": true}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        ground_truth: dict[str, Any] = {
            "criteria": ["Single criterion"],
            "points_list": [10],
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_large_number_of_criteria(self):
        """Large number of criteria should be handled efficiently."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": true}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        num_criteria = 50
        ground_truth: dict[str, Any] = {
            "criteria": [f"Criterion {i}" for i in range(num_criteria)],
            "points_list": [1] * num_criteria,
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
            max_parallel_judges=10,
        )

        assert result == 1.0
        assert mock_client.chat.completions.create.call_count == num_criteria


class TestEdgeCases:
    """Edge case tests for robustness."""

    @pytest.mark.asyncio
    async def test_zero_points_criterion(self):
        """Criterion with zero points should not affect total_positive."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": true}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        ground_truth: dict[str, Any] = {
            "criteria": ["Important", "Zero weight", "Also important"],
            "points_list": [5, 0, 5],  # Zero in the middle
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        # total_positive = 5 + 5 = 10 (0 is not positive)
        # earned = 5 + 0 + 5 = 10
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_all_negative_points_with_penalty_met(self):
        """All negative points met equals penalties applied and score clamped to 0."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": true}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        ground_truth: dict[str, Any] = {
            "criteria": ["Must not do X", "Must not do Y"],
            "points_list": [-2, -3],  # All penalties
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        # total_positive = 0, so returns 0.0 early
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_mixed_response_parsing(self):
        """Different criteria can have different response formats."""
        mock_client = MagicMock()
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            # Mix of formats
            if call_count == 1:
                mock_response.choices[0].message.content = '{"criteria_met": true}'
            elif call_count == 2:
                mock_response.choices[
                    0
                ].message.content = '```json\n{"criteria_met": false}\n```'
            else:
                mock_response.choices[
                    0
                ].message.content = '{"criteria_met": true, "explanation": "Good"}'
            return mock_response

        mock_client.chat.completions.create = AsyncMock(side_effect=side_effect)

        ground_truth: dict[str, Any] = {
            "criteria": ["Crit1", "Crit2", "Crit3"],
            "points_list": [2, 3, 5],  # Total = 10
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        # Earned: 2 + 0 + 5 = 7, total = 10, score = 0.7
        assert result == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_handles_empty_solution_string(self):
        """Empty solution string should still trigger evaluation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": true}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        ground_truth: dict[str, Any] = {
            "criteria": ["Crit1"],
            "points_list": [5],
        }

        result = await compute_score(
            solution_str="",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_very_large_points_values(self):
        """Should handle large point values without overflow issues."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"criteria_met": true}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        ground_truth: dict[str, Any] = {
            "criteria": ["Crit1", "Crit2"],
            "points_list": [1000, 2000],  # Large values
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_concurrent_evaluation_order(self):
        """Results should be correctly aggregated regardless of completion order."""
        mock_client = MagicMock()
        delays = [0.03, 0.01, 0.02]  # Different delays
        call_index = 0

        async def delayed_response(*args, **kwargs):
            nonlocal call_index
            delay = delays[call_index % len(delays)]
            call_index += 1
            await asyncio.sleep(delay)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"criteria_met": true}'
            return mock_response

        mock_client.chat.completions.create = AsyncMock(side_effect=delayed_response)

        ground_truth: dict[str, Any] = {
            "criteria": ["Crit1", "Crit2", "Crit3"],
            "points_list": [1, 2, 3],  # Total = 6
        }

        result = await compute_score(
            solution_str="Answer",
            ground_truth=ground_truth,
            judge_client=mock_client,
            judge_model="test-model",
            max_parallel_judges=3,
        )

        assert result == 1.0
