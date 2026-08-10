"""Tests for MedXpertQA Verl evaluator.

Tests cover main() function, dataset loading, result building,
and question_type parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from med_reason_evals.verl import MedXpertQAEvaluator


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestMedXpertQAMain:
    """Tests for MedXpertQA main() function."""

    async def test_main_prints_results(self, mocker):
        """Test main() function prints results correctly."""
        mock_print = mocker.patch("builtins.print")

        mock_evaluator_class = mocker.patch(
            "med_reason_evals.verl.medxpertqa.MedXpertQAEvaluator"
        )
        mock_evaluator = MagicMock()
        mock_results = {
            "dataset": "medxpertqa",
            "split": "test",
            "question_type": "all",
            "num_examples": 100,
            "avg_score": 0.6,
        }
        mock_evaluator.evaluate = AsyncMock(return_value=mock_results)
        mock_evaluator_class.return_value = mock_evaluator

        from med_reason_evals.verl.medxpertqa import main

        await main()

        mock_evaluator_class.assert_called_once()
        mock_evaluator.evaluate.assert_called_once_with(num_examples=100)
        mock_print.assert_called_once()
        assert "MedXpertQA Verl Results:" in mock_print.call_args[0][0]


class TestMedXpertQALoadDataset:
    """Tests for MedXpertQA _load_dataset method."""

    def test_load_dataset(self, mocker, medxpertqa_verl_dataset: Dataset):
        """Test _load_dataset method."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.medxpertqa.MedXpertQADataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = medxpertqa_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedXpertQAEvaluator(split="train", question_type="reasoning")
            result = evaluator._load_dataset()

        mock_dataset_class.assert_called_once_with(
            split="train", question_type="reasoning", streaming=True
        )
        assert result == mock_verl_dataset

    def test_load_dataset_question_types(
        self, mocker, medxpertqa_verl_dataset: Dataset
    ):
        """Test _load_dataset with different question types."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.medxpertqa.MedXpertQADataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = medxpertqa_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        question_types = ["all", "reasoning", "understanding"]
        for qtype in question_types:
            with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
                evaluator = MedXpertQAEvaluator(question_type=qtype)
                evaluator._load_dataset()

            mock_dataset_class.assert_called_with(
                split="test", question_type=qtype, streaming=True
            )

    def test_load_dataset_has_required_fields(
        self, mocker, medxpertqa_verl_dataset: Dataset
    ):
        """Test _load_dataset returns dataset with required Verl schema fields."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.medxpertqa.MedXpertQADataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = medxpertqa_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedXpertQAEvaluator()
            result = evaluator._load_dataset()

        # Verify Verl schema: prompt, ground_truth, metadata
        assert "prompt" in result.column_names
        assert "ground_truth" in result.column_names
        assert "metadata" in result.column_names


class TestMedXpertQABuildResult:
    """Tests for MedXpertQA _build_result method."""

    def test_build_result(self, assert_verl_result_shape: Callable):
        """Test _build_result method."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedXpertQAEvaluator(split="test", question_type="reasoning")
            scores = [1.0, 1.0]
            avg_score = 1.0

            result = evaluator._build_result(avg_score, num_examples=len(scores))

            assert_verl_result_shape(
                result,
                required_keys=[
                    "dataset",
                    "split",
                    "question_type",
                    "num_examples",
                    "avg_score",
                ],
                expected_dataset="medxpertqa",
            )
            assert result["question_type"] == "reasoning"
            assert result["num_examples"] == 2

    def test_build_result_mixed_scores(self):
        """Test _build_result with mixed scores."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedXpertQAEvaluator()
            scores = [0.0, 0.5, 1.0, 0.0, 0.5]
            avg_score = 2.0 / 5.0

            result = evaluator._build_result(avg_score, num_examples=len(scores))

            assert result["num_examples"] == 5
            assert result["avg_score"] == pytest.approx(0.4, rel=1e-3)
