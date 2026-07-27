"""Tests for MedQA Verl evaluator.

Tests cover main() function, dataset loading, and result building.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from med_reason_evals.verl import MedQAEvaluator


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestMedQAMain:
    """Tests for MedQA main() function."""

    async def test_main_prints_results(self, mocker):
        """Test main() function prints results correctly."""
        mock_print = mocker.patch("builtins.print")

        mock_evaluator_class = mocker.patch(
            "med_reason_evals.verl.medqa.MedQAEvaluator"
        )
        mock_evaluator = MagicMock()
        mock_results = {
            "dataset": "medqa",
            "split": "test",
            "num_examples": 100,
            "accuracy": 0.75,
        }
        mock_evaluator.evaluate = AsyncMock(return_value=mock_results)
        mock_evaluator_class.return_value = mock_evaluator

        from med_reason_evals.verl.medqa import main

        await main()

        mock_evaluator_class.assert_called_once()
        mock_evaluator.evaluate.assert_called_once_with(num_examples=100)
        assert mock_print.call_count == 5
        calls = mock_print.call_args_list
        assert "MedQA Verl Results:" in calls[0][0][0]
        assert "0.750" in str(calls[4])

    async def test_main_zero_accuracy(self, mocker):
        """Test main() with 0% accuracy."""
        mock_print = mocker.patch("builtins.print")

        mock_evaluator_class = mocker.patch(
            "med_reason_evals.verl.medqa.MedQAEvaluator"
        )
        mock_evaluator = MagicMock()
        mock_results = {
            "dataset": "medqa",
            "split": "test",
            "num_examples": 100,
            "accuracy": 0.0,
        }
        mock_evaluator.evaluate = AsyncMock(return_value=mock_results)
        mock_evaluator_class.return_value = mock_evaluator

        from med_reason_evals.verl.medqa import main

        await main()

        calls = mock_print.call_args_list
        assert "0.000" in str(calls[4])

    async def test_main_perfect_accuracy(self, mocker):
        """Test main() with 100% accuracy."""
        mock_print = mocker.patch("builtins.print")

        mock_evaluator_class = mocker.patch(
            "med_reason_evals.verl.medqa.MedQAEvaluator"
        )
        mock_evaluator = MagicMock()
        mock_results = {
            "dataset": "medqa",
            "split": "test",
            "num_examples": 100,
            "accuracy": 1.0,
        }
        mock_evaluator.evaluate = AsyncMock(return_value=mock_results)
        mock_evaluator_class.return_value = mock_evaluator

        from med_reason_evals.verl.medqa import main

        await main()

        calls = mock_print.call_args_list
        assert "1.000" in str(calls[4])


class TestMedQALoadDataset:
    """Tests for MedQA _load_dataset method."""

    def test_load_dataset(self, mocker, medqa_verl_dataset: Dataset):
        """Test _load_dataset method."""
        mock_dataset_class = mocker.patch("med_reason_evals.verl.medqa.MedQADataset")
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = medqa_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedQAEvaluator(split="train")
            result = evaluator._load_dataset()

        mock_dataset_class.assert_called_once_with(split="train", streaming=True)
        mock_dataset_instance.get_verl_dataset.assert_called_once()
        assert result == mock_verl_dataset

    def test_load_dataset_has_required_fields(
        self, mocker, medqa_verl_dataset: Dataset
    ):
        """Test _load_dataset returns dataset with required Verl schema fields."""
        mock_dataset_class = mocker.patch("med_reason_evals.verl.medqa.MedQADataset")
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = medqa_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedQAEvaluator()
            result = evaluator._load_dataset()

        # Verify Verl schema: prompt, ground_truth, metadata
        assert "prompt" in result.column_names
        assert "ground_truth" in result.column_names
        assert "metadata" in result.column_names

    def test_load_dataset_non_streaming(self, mocker, medqa_verl_dataset: Dataset):
        """Test _load_dataset with streaming=False."""
        mock_dataset_class = mocker.patch("med_reason_evals.verl.medqa.MedQADataset")
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = medqa_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedQAEvaluator(streaming=False)
            result = evaluator._load_dataset()

        mock_dataset_class.assert_called_once_with(split="test", streaming=False)
        assert result == mock_verl_dataset


class TestMedQABuildResult:
    """Tests for MedQA _build_result method."""

    def test_build_result(self, assert_verl_result_shape: Callable):
        """Test _build_result method."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedQAEvaluator(split="test")
            scores = [1.0, 0.0, 1.0, 1.0]
            avg_score = 0.75

            result = evaluator._build_result(scores, avg_score)

            assert_verl_result_shape(
                result,
                required_keys=[
                    "dataset",
                    "split",
                    "num_examples",
                    "avg_score",
                    "accuracy",
                ],
                expected_dataset="medqa",
            )
            assert result["split"] == "test"
            assert result["num_examples"] == 4
            assert result["accuracy"] == 0.75

    def test_build_result_empty_scores(self):
        """Test _build_result with empty scores."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedQAEvaluator()
            scores = []
            avg_score = 0.0

            result = evaluator._build_result(scores, avg_score)

            assert result["num_examples"] == 0
            assert result["avg_score"] == 0.0
            assert result["accuracy"] == 0.0
