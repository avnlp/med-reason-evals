"""Tests for MedCaseReasoning Verl evaluator.

Tests cover main() function, dataset loading, result building, and _evaluate_example.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from med_reason_evals.verl import MedCaseReasoningEvaluator


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestMedCaseReasoningMain:
    """Tests for MedCaseReasoning main() function."""

    async def test_main_prints_results(self, mocker):
        """Test main() function prints results correctly."""
        mock_print = mocker.patch("builtins.print")

        mock_evaluator_class = mocker.patch(
            "med_reason_evals.verl.medcasereasoning.MedCaseReasoningEvaluator"
        )
        mock_evaluator = MagicMock()
        mock_results = {
            "dataset": "medcasereasoning",
            "split": "val",
            "num_examples": 50,
            "avg_score": 0.7,
        }
        mock_evaluator.evaluate = AsyncMock(return_value=mock_results)
        mock_evaluator_class.return_value = mock_evaluator

        from med_reason_evals.verl.medcasereasoning import main

        await main()

        mock_evaluator_class.assert_called_once()
        mock_evaluator.evaluate.assert_called_once_with(num_examples=50)
        mock_print.assert_called_once()
        assert "MedCaseReasoning Verl Results:" in mock_print.call_args[0][0]


class TestMedCaseReasoningLoadDataset:
    """Tests for MedCaseReasoning _load_dataset method."""

    def test_load_dataset(self, mocker, medcasereasoning_verl_dataset: Dataset):
        """Test _load_dataset method."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.medcasereasoning.MedCaseReasoningDataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = medcasereasoning_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedCaseReasoningEvaluator(split="train")
            result = evaluator._load_dataset()

        mock_dataset_class.assert_called_once_with(split="train", streaming=True)
        assert result == mock_verl_dataset

    def test_load_dataset_has_required_fields(
        self, mocker, medcasereasoning_verl_dataset: Dataset
    ):
        """Test _load_dataset returns dataset with required Verl schema fields."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.medcasereasoning.MedCaseReasoningDataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = medcasereasoning_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedCaseReasoningEvaluator()
            result = evaluator._load_dataset()

        # Verify Verl schema: prompt, ground_truth, metadata
        assert "prompt" in result.column_names
        assert "ground_truth" in result.column_names
        assert "metadata" in result.column_names


class TestMedCaseReasoningBuildResult:
    """Tests for MedCaseReasoning _build_result method."""

    def test_build_result(self, assert_verl_result_shape: Callable):
        """Test _build_result method."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedCaseReasoningEvaluator(split="val")
            scores = [1.0, 0.0, 0.5]
            avg_score = 1.5 / 3.0

            result = evaluator._build_result(avg_score, num_examples=len(scores))

            assert_verl_result_shape(
                result,
                required_keys=["dataset", "split", "num_examples"],
                expected_dataset="medcasereasoning",
            )
            assert result["split"] == "val"
            assert result["num_examples"] == 3


class TestMedCaseReasoningEvaluateExample:
    """Tests for MedCaseReasoning _evaluate_example method."""

    async def test_evaluate_example(self, mocker):
        """Test _evaluate_example method."""
        mock_judge_score = mocker.patch(
            "med_reason_evals.verl.medcasereasoning.judge_score"
        )
        mock_judge_score.return_value = 1.0

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedCaseReasoningEvaluator()
            # Mock rollouts
            evaluator._rollouts = MagicMock()
            evaluator._rollouts.generate = AsyncMock(
                return_value="<answer>Diabetes</answer>"
            )

            prompt = [{"role": "user", "content": "Patient presents with..."}]
            ground_truth = {"target": "Type 2 Diabetes"}

            score = await evaluator._evaluate_example(prompt, ground_truth)

            assert score == 1.0
            mock_judge_score.assert_called_once()
            call_kwargs = mock_judge_score.call_args.kwargs
            assert call_kwargs["solution_str"] == "<answer>Diabetes</answer>"
            assert call_kwargs["ground_truth"] == ground_truth

    async def test_evaluate_example_no_metadata(self, mocker):
        """Test _evaluate_example without metadata."""
        mock_judge_score = mocker.patch(
            "med_reason_evals.verl.medcasereasoning.judge_score"
        )
        mock_judge_score.return_value = 0.5

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedCaseReasoningEvaluator()
            evaluator._rollouts = MagicMock()
            evaluator._rollouts.generate = AsyncMock(return_value="Response")

            prompt = [{"role": "user", "content": "Question"}]
            ground_truth = {"target": "Answer"}

            # Call without metadata (should default to None)
            score = await evaluator._evaluate_example(prompt, ground_truth)

            assert score == 0.5
