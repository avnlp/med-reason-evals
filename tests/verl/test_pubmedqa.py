"""Tests for PubMedQA Verl evaluator.

Tests cover main() function, dataset loading, result building, and subset parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from med_reason_evals.verl import PubMedQAEvaluator


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestPubMedQAMain:
    """Tests for PubMedQA main() function."""

    async def test_main_prints_results(self, mocker):
        """Test main() function prints results correctly."""
        mock_print = mocker.patch("builtins.print")

        mock_evaluator_class = mocker.patch(
            "med_reason_evals.verl.pubmedqa.PubMedQAEvaluator"
        )
        mock_evaluator = MagicMock()
        mock_results = {
            "dataset": "pubmedqa",
            "num_examples": 100,
            "avg_score": 0.78,
        }
        mock_evaluator.evaluate = AsyncMock(return_value=mock_results)
        mock_evaluator_class.return_value = mock_evaluator

        from med_reason_evals.verl.pubmedqa import main

        await main()

        mock_evaluator_class.assert_called_once()
        mock_evaluator.evaluate.assert_called_once_with(num_examples=100)
        mock_print.assert_called_once()
        assert "PubMedQA Verl Results:" in mock_print.call_args[0][0]


class TestPubMedQALoadDataset:
    """Tests for PubMedQA _load_dataset method."""

    def test_load_dataset(self, mocker, pubmedqa_verl_dataset: Dataset):
        """Test _load_dataset method."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.pubmedqa.PubMedQADataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = pubmedqa_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = PubMedQAEvaluator()
            result = evaluator._load_dataset()

        mock_dataset_class.assert_called_once_with(streaming=True)
        assert result == mock_verl_dataset

    def test_load_dataset_has_required_fields(
        self, mocker, pubmedqa_verl_dataset: Dataset
    ):
        """Test _load_dataset returns dataset with required Verl schema fields."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.pubmedqa.PubMedQADataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = pubmedqa_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = PubMedQAEvaluator()
            result = evaluator._load_dataset()

        # Verify Verl schema: prompt, ground_truth, metadata
        assert "prompt" in result.column_names
        assert "ground_truth" in result.column_names
        assert "metadata" in result.column_names


class TestPubMedQABuildResult:
    """Tests for PubMedQA _build_result method."""

    def test_build_result(self, assert_verl_result_shape: Callable):
        """Test _build_result method."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = PubMedQAEvaluator()
            avg_score = 1.0

            result = evaluator._build_result(avg_score, num_examples=3)

            assert_verl_result_shape(
                result,
                required_keys=["dataset", "num_examples", "avg_score"],
                expected_dataset="pubmedqa",
            )
            assert result["num_examples"] == 3
