"""Tests for MedBullets Verifiers evaluator.

Tests cover dataset loading, environment construction, parser configuration,
rubric wiring, and MedBullets-specific num_options parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from med_reason_evals.verifiers.medbullets import MedBulletsEvaluator
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestMedBulletsDatasetLoading:
    """Tests for MedBullets dataset loading."""

    def test_load_datasets_no_train_split(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medbullets_mock_dataset: Dataset,
    ) -> None:
        """Test that MedBullets has no train split (train_ds is None)."""
        with patch(
            "med_reason_evals.data.medbullets.load_dataset",
            mock_load_dataset_factory(medbullets_mock_dataset),
        ):
            evaluator = MedBulletsEvaluator(streaming=False)
            train_ds, eval_ds = evaluator._load_datasets()

        assert train_ds is None, "MedBullets should not have a train split"
        assert eval_ds is not None

    def test_dataset_has_required_fields(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medbullets_mock_dataset: Dataset,
        assert_env_has_basic_fields: Callable,
    ) -> None:
        """Test that mapped dataset has required verifiers fields."""
        with patch(
            "med_reason_evals.data.medbullets.load_dataset",
            mock_load_dataset_factory(medbullets_mock_dataset),
        ):
            evaluator = MedBulletsEvaluator(streaming=False)
            env = evaluator.environment()

            eval_ds = env.eval_dataset
            assert "question" in eval_ds.column_names
            assert "answer" in eval_ds.column_names
            assert "info" in eval_ds.column_names


class TestMedBulletsEnvironment:
    """Tests for MedBullets environment construction."""

    def test_environment_construction(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medbullets_mock_dataset: Dataset,
        assert_env_has_basic_fields: Callable,
    ) -> None:
        """Test environment is constructed correctly."""
        with patch(
            "med_reason_evals.data.medbullets.load_dataset",
            mock_load_dataset_factory(medbullets_mock_dataset),
        ):
            evaluator = MedBulletsEvaluator(streaming=False)
            env = evaluator.environment()

            assert_env_has_basic_fields(env)


class TestMedBulletsParserConfiguration:
    """Tests for MedBullets parser configuration."""

    def test_default_boxed_format(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medbullets_mock_dataset: Dataset,
    ) -> None:
        """Verify default format is BOXED."""
        with patch(
            "med_reason_evals.data.medbullets.load_dataset",
            mock_load_dataset_factory(medbullets_mock_dataset),
        ):
            evaluator = MedBulletsEvaluator(streaming=False)

            assert evaluator.answer_format == AnswerFormat.BOXED
            assert evaluator.use_think is False

    def test_boxed_parser(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medbullets_mock_dataset: Dataset,
        assert_parser_is_boxed: Callable,
    ) -> None:
        """Verify BOXED format uses vf.Parser."""
        with patch(
            "med_reason_evals.data.medbullets.load_dataset",
            mock_load_dataset_factory(medbullets_mock_dataset),
        ):
            evaluator = MedBulletsEvaluator(streaming=False)
            env = evaluator.environment()

            assert_parser_is_boxed(env.parser, has_think=False)


class TestMedBulletsRubricConfiguration:
    """Tests for MedBullets rubric configuration."""

    def test_rubric_has_accuracy_reward(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medbullets_mock_dataset: Dataset,
        assert_rubric_has_one_func_weight_one: Callable,
    ) -> None:
        """Verify rubric uses accuracy_reward."""
        with patch(
            "med_reason_evals.data.medbullets.load_dataset",
            mock_load_dataset_factory(medbullets_mock_dataset),
        ):
            evaluator = MedBulletsEvaluator(streaming=False)
            env = evaluator.environment()

            assert_rubric_has_one_func_weight_one(
                env.rubric, func_name="accuracy_reward"
            )


class TestMedBulletsNumOptions:
    """Tests for MedBullets num_options parameter."""

    def test_num_options_parameter_4(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medbullets_mock_dataset: Dataset,
    ) -> None:
        """Verify num_options=4 is accepted."""
        with patch(
            "med_reason_evals.data.medbullets.load_dataset",
            mock_load_dataset_factory(medbullets_mock_dataset),
        ):
            evaluator = MedBulletsEvaluator(num_options=4, streaming=False)

            assert evaluator.num_options == 4

    def test_num_options_parameter_5(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medbullets_mock_dataset: Dataset,
    ) -> None:
        """Verify num_options=5 is accepted."""
        with patch(
            "med_reason_evals.data.medbullets.load_dataset",
            mock_load_dataset_factory(medbullets_mock_dataset),
        ):
            evaluator = MedBulletsEvaluator(num_options=5, streaming=False)

            assert evaluator.num_options == 5

    def test_invalid_num_options_raises(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medbullets_mock_dataset: Dataset,
    ) -> None:
        """Verify invalid num_options raises ValueError."""
        with patch(
            "med_reason_evals.data.medbullets.load_dataset",
            mock_load_dataset_factory(medbullets_mock_dataset),
        ):
            evaluator = MedBulletsEvaluator(num_options=3, streaming=False)

            with pytest.raises(ValueError, match="num_options must be 4 or 5"):
                evaluator.environment()
