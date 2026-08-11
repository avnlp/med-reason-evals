"""Tests for MedBullets Verl evaluator.

Tests cover main() function, dataset loading, and result building.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from med_reason_evals.verl import MedBulletsEvaluator


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestMedBulletsMain:
    """Tests for MedBullets main() function."""

    async def test_main_prints_results(self, mocker):
        """Test main() function prints results correctly."""
        mock_print = mocker.patch("builtins.print")

        mock_evaluator_class = mocker.patch(
            "med_reason_evals.verl.medbullets.MedBulletsEvaluator"
        )
        mock_evaluator = MagicMock()
        mock_results = {
            "dataset": "medbullets",
            "num_examples": 100,
            "num_options": 4,
            "avg_score": 0.65,
        }
        mock_evaluator.evaluate = AsyncMock(return_value=mock_results)
        mock_evaluator_class.return_value = mock_evaluator

        from med_reason_evals.verl.medbullets import main

        await main()

        mock_evaluator_class.assert_called_once()
        mock_evaluator.evaluate.assert_called_once_with(num_examples=100)
        mock_print.assert_called_once()
        assert "MedBullets Verl Results:" in mock_print.call_args[0][0]


class TestMedBulletsLoadDataset:
    """Tests for MedBullets _load_dataset method."""

    def test_load_dataset(self, mocker, medbullets_verl_dataset: Dataset):
        """Test _load_dataset method."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.medbullets.MedBulletsDataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = medbullets_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedBulletsEvaluator(num_options=5)
            result = evaluator._load_dataset()

        mock_dataset_class.assert_called_once_with(num_options=5, streaming=True)
        assert result == mock_verl_dataset

    def test_load_dataset_has_required_fields(
        self, mocker, medbullets_verl_dataset: Dataset
    ):
        """Test _load_dataset returns dataset with required Verl schema fields."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.medbullets.MedBulletsDataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = medbullets_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedBulletsEvaluator()
            result = evaluator._load_dataset()

        # Verify Verl schema: prompt, ground_truth, metadata
        assert "prompt" in result.column_names
        assert "ground_truth" in result.column_names
        assert "metadata" in result.column_names

    def test_load_dataset_different_num_options(
        self, mocker, medbullets_verl_dataset: Dataset
    ):
        """Test _load_dataset with different num_options values."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.medbullets.MedBulletsDataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = medbullets_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        for num_options in [4, 5]:
            with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
                evaluator = MedBulletsEvaluator(num_options=num_options)
                evaluator._load_dataset()

            mock_dataset_class.assert_called_with(
                num_options=num_options, streaming=True
            )


class TestMedBulletsBuildResult:
    """Tests for MedBullets _build_result method."""

    def test_build_result(self, assert_verl_result_shape: Callable):
        """Test _build_result method."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedBulletsEvaluator(num_options=5)
            avg_score = 2.5 / 3.0

            result = evaluator._build_result(avg_score, num_examples=3)

            assert_verl_result_shape(
                result,
                required_keys=["dataset", "num_examples", "avg_score", "num_options"],
                expected_dataset="medbullets",
            )
            assert result["num_examples"] == 3
            assert result["num_options"] == 5
            assert result["avg_score"] == pytest.approx(0.833, rel=1e-3)

    def test_build_result_zero_examples(self):
        """Test _build_result with zero examples."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = MedBulletsEvaluator()

            result = evaluator._build_result(0.0, num_examples=0)

            assert result["num_examples"] == 0
            assert result["avg_score"] == 0.0
