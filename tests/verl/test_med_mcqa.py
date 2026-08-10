"""Tests for MedMCQA Verl evaluator.

Tests cover main() function, dataset loading, and result building.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from med_reason_evals.verl import MedMCQAEvaluator


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestMedMCQAMain:
    """Tests for MedMCQA main() function."""

    async def test_main_prints_results(self, mocker):
        """Test main() function prints results correctly."""
        mock_print = mocker.patch("builtins.print")

        mock_evaluator_class = mocker.patch(
            "med_reason_evals.verl.med_mcqa.MedMCQAEvaluator"
        )
        mock_evaluator = MagicMock()
        mock_results = {
            "dataset": "med_mcqa",
            "split": "validation",
            "num_examples": 100,
            "accuracy": 0.8,
        }
        mock_evaluator.evaluate = AsyncMock(return_value=mock_results)
        mock_evaluator_class.return_value = mock_evaluator

        from med_reason_evals.verl.med_mcqa import main

        await main()

        mock_evaluator_class.assert_called_once()
        mock_evaluator.evaluate.assert_called_once_with(num_examples=100)
        assert mock_print.call_count == 5
        calls = mock_print.call_args_list
        assert "MedMCQA Verl Results:" in calls[0][0][0]
        assert "0.800" in str(calls[4])

    async def test_main_zero_accuracy(self, mocker):
        """Test main() with 0% accuracy."""
        mock_print = mocker.patch("builtins.print")

        mock_evaluator_class = mocker.patch(
            "med_reason_evals.verl.med_mcqa.MedMCQAEvaluator"
        )
        mock_evaluator = MagicMock()
        mock_results = {
            "dataset": "med_mcqa",
            "split": "validation",
            "num_examples": 100,
            "accuracy": 0.0,
        }
        mock_evaluator.evaluate = AsyncMock(return_value=mock_results)
        mock_evaluator_class.return_value = mock_evaluator

        from med_reason_evals.verl.med_mcqa import main

        await main()

        calls = mock_print.call_args_list
        assert "0.000" in str(calls[4])

    async def test_main_perfect_accuracy(self, mocker):
        """Test main() with 100% accuracy."""
        mock_print = mocker.patch("builtins.print")

        mock_evaluator_class = mocker.patch(
            "med_reason_evals.verl.med_mcqa.MedMCQAEvaluator"
        )
        mock_evaluator = MagicMock()
        mock_results = {
            "dataset": "med_mcqa",
            "split": "validation",
            "num_examples": 100,
            "accuracy": 1.0,
        }
        mock_evaluator.evaluate = AsyncMock(return_value=mock_results)
        mock_evaluator_class.return_value = mock_evaluator

        from med_reason_evals.verl.med_mcqa import main

        await main()

        calls = mock_print.call_args_list
        assert "1.000" in str(calls[4])


class TestMedMCQALoadDataset:
    """Tests for MedMCQA _load_dataset method."""

    def test_load_dataset(self, mocker, med_mcqa_verl_dataset: Dataset):
        """Test _load_dataset method."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.med_mcqa.MedMCQADataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = med_mcqa_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedMCQAEvaluator(split="train")
            result = evaluator._load_dataset()

        mock_dataset_class.assert_called_once_with(split="train", streaming=True)
        mock_dataset_instance.get_verl_dataset.assert_called_once()
        assert result == mock_verl_dataset

    def test_load_dataset_has_required_fields(
        self, mocker, med_mcqa_verl_dataset: Dataset
    ):
        """Test _load_dataset returns dataset with required Verl schema fields."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.med_mcqa.MedMCQADataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = med_mcqa_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedMCQAEvaluator()
            result = evaluator._load_dataset()

        # Verify Verl schema: prompt, ground_truth, metadata
        assert "prompt" in result.column_names
        assert "ground_truth" in result.column_names
        assert "metadata" in result.column_names

    def test_load_dataset_non_streaming(self, mocker, med_mcqa_verl_dataset: Dataset):
        """Test _load_dataset with streaming=False."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.med_mcqa.MedMCQADataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = med_mcqa_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedMCQAEvaluator(streaming=False)
            result = evaluator._load_dataset()

        mock_dataset_class.assert_called_once_with(split="validation", streaming=False)
        assert result == mock_verl_dataset


class TestMedMCQABuildResult:
    """Tests for MedMCQA _build_result method."""

    def test_build_result(self, assert_verl_result_shape: Callable):
        """Test _build_result method."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedMCQAEvaluator(split="validation")
            avg_score = 0.667

            result = evaluator._build_result(avg_score, num_examples=3)

            assert_verl_result_shape(
                result,
                required_keys=[
                    "dataset",
                    "split",
                    "num_examples",
                    "avg_score",
                    "accuracy",
                ],
                expected_dataset="med_mcqa",
            )
            assert result["split"] == "validation"
            assert result["num_examples"] == 3
            assert result["accuracy"] == 0.667

    def test_build_result_zero_examples(self):
        """Test _build_result with zero examples."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedMCQAEvaluator()
            avg_score = 0.0

            result = evaluator._build_result(avg_score, num_examples=0)

            assert result["num_examples"] == 0
            assert result["avg_score"] == 0.0
            assert result["accuracy"] == 0.0

    def test_build_result_single_score(self):
        """Test _build_result with a single score."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedMCQAEvaluator()
            avg_score = 1.0

            result = evaluator._build_result(avg_score, num_examples=1)

            assert result["num_examples"] == 1
            assert result["avg_score"] == 1.0
