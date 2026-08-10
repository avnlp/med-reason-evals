"""Tests for HealthBench Verl evaluator.

Tests cover main() function, dataset loading, result building, and _evaluate_example.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from med_reason_evals.verl import HealthBenchEvaluator
from med_reason_evals.verl.base import GroqGenConfig


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestHealthBenchMain:
    """Tests for HealthBench main() function."""

    async def test_main_prints_results(self, mocker):
        """Test main() function prints results correctly."""
        mock_print = mocker.patch("builtins.print")

        mock_evaluator_class = mocker.patch(
            "med_reason_evals.verl.healthbench.HealthBenchEvaluator"
        )
        mock_evaluator = MagicMock()
        mock_results = {
            "dataset": "healthbench",
            "difficulty": "regular",
            "num_examples": 2,
            "avg_score": 0.85,
        }
        mock_evaluator.evaluate = AsyncMock(return_value=mock_results)
        mock_evaluator_class.return_value = mock_evaluator

        from med_reason_evals.verl.healthbench import main

        await main()

        mock_evaluator_class.assert_called_once()
        mock_evaluator.evaluate.assert_called_once_with(num_examples=2)
        mock_print.assert_called_once()
        assert "HealthBench Verl Results:" in mock_print.call_args[0][0]


class TestHealthBenchLoadDataset:
    """Tests for HealthBench _load_dataset method."""

    def test_load_dataset(self, mocker, healthbench_verl_dataset: Dataset):
        """Test _load_dataset method."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.healthbench.HealthBenchDataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = healthbench_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = HealthBenchEvaluator(difficulty="hard")
            result = evaluator._load_dataset()

        mock_dataset_class.assert_called_once_with(difficulty="hard", streaming=True)
        assert result == mock_verl_dataset

    def test_load_dataset_difficulties(self, mocker, healthbench_verl_dataset: Dataset):
        """Test _load_dataset with different difficulties."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.healthbench.HealthBenchDataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = healthbench_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        difficulties = ["regular", "consensus", "hard"]
        for difficulty in difficulties:
            with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
                evaluator = HealthBenchEvaluator(difficulty=difficulty)
                evaluator._load_dataset()

            mock_dataset_class.assert_called_with(difficulty=difficulty, streaming=True)

    def test_load_dataset_has_required_fields(
        self, mocker, healthbench_verl_dataset: Dataset
    ):
        """Test _load_dataset returns dataset with required Verl schema fields."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.healthbench.HealthBenchDataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = healthbench_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = HealthBenchEvaluator()
            result = evaluator._load_dataset()

        # Verify Verl schema: prompt, ground_truth, metadata
        assert "prompt" in result.column_names
        assert "ground_truth" in result.column_names
        assert "metadata" in result.column_names


class TestHealthBenchBuildResult:
    """Tests for HealthBench _build_result method."""

    def test_build_result(self, assert_verl_result_shape: Callable):
        """Test _build_result method."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = HealthBenchEvaluator(difficulty="consensus")
            scores = [0.8, 0.9]
            avg_score = 0.85

            result = evaluator._build_result(avg_score, num_examples=len(scores))

            assert_verl_result_shape(
                result,
                required_keys=["dataset", "difficulty", "num_examples", "avg_score"],
                expected_dataset="healthbench",
            )
            assert result["difficulty"] == "consensus"
            assert result["num_examples"] == 2


class TestHealthBenchEvaluateExample:
    """Tests for HealthBench _evaluate_example method."""

    async def test_evaluate_example(self, mocker):
        """Test _evaluate_example method."""
        mock_rubric_score = mocker.patch(
            "med_reason_evals.verl.healthbench.rubric_score"
        )
        mock_rubric_score.return_value = 0.9

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = HealthBenchEvaluator(max_parallel_judges=3)
            evaluator._rollouts = MagicMock()
            evaluator._rollouts.generate = AsyncMock(return_value="Some answer")

            prompt = [{"role": "user", "content": "Evaluate this case..."}]
            ground_truth = {
                "criteria": ["criterion1"],
                "points_list": [10],
            }

            score = await evaluator._evaluate_example(prompt, ground_truth)

            assert score == 0.9
            mock_rubric_score.assert_called_once()
            call_kwargs = mock_rubric_score.call_args.kwargs
            assert call_kwargs["solution_str"] == "Some answer"
            assert call_kwargs["ground_truth"] == ground_truth
            assert call_kwargs["max_parallel_judges"] == 3
            # The question must reach the judge for contextual scoring, and
            # judge concurrency must be shared via an evaluator-wide semaphore.
            assert call_kwargs["conversation"] == "user: Evaluate this case..."
            assert isinstance(call_kwargs["semaphore"], asyncio.Semaphore)

    async def test_evaluate_example_with_metadata_ignored(self, mocker):
        """Test _evaluate_example with metadata explicitly passed.

        Metadata is accepted but not used for rubric scoring, so the score is
        identical to an example evaluated without it.
        """
        mock_rubric_score = mocker.patch(
            "med_reason_evals.verl.healthbench.rubric_score"
        )
        mock_rubric_score.return_value = 0.75

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = HealthBenchEvaluator()
            evaluator._rollouts = MagicMock()
            evaluator._rollouts.generate = AsyncMock(return_value="Response")

            prompt = [{"role": "user", "content": "Question"}]
            ground_truth = {"criteria": [], "points_list": []}

            score = await evaluator._evaluate_example(
                prompt,
                ground_truth,
                metadata={"prompt_id": "hb_001", "difficulty": "regular"},
            )

            assert score == 0.75
            mock_rubric_score.assert_called_once()

    async def test_evaluate_example_forwards_sampling_args(self, mocker):
        """Test gen_config.sampling_args are forwarded to generate."""
        mock_rubric_score = mocker.patch(
            "med_reason_evals.verl.healthbench.rubric_score"
        )
        mock_rubric_score.return_value = 0.8

        gen_config = GroqGenConfig(sampling_args={"temperature": 0.7})
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = HealthBenchEvaluator(gen_config=gen_config)
            evaluator._rollouts = MagicMock()
            evaluator._rollouts.generate = AsyncMock(return_value="Answer")

            prompt = [{"role": "user", "content": "Question"}]
            await evaluator._evaluate_example(
                prompt,
                {"criteria": ["criterion1"], "points_list": [1]},
            )

        evaluator._rollouts.generate.assert_called_once_with(
            messages=prompt,
            temperature=0.7,
        )
