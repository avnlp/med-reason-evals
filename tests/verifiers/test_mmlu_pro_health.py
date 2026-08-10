"""Tests for MMLUProHealth Verifiers evaluator.

Tests cover dataset loading, environment construction, parser configuration,
rubric wiring, and default use_think=True behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from med_reason_evals.verifiers.mmlu_pro_health import MMLUProHealthEvaluator
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestMMLUProHealthDatasetLoading:
    """Tests for MMLUProHealth dataset loading."""

    def test_load_datasets_no_train_split(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        mmlu_pro_health_mock_dataset: Dataset,
    ) -> None:
        """Test that MMLUProHealth has no train split."""
        with patch(
            "med_reason_evals.data.mmlu_pro_health.load_dataset",
            mock_load_dataset_factory(mmlu_pro_health_mock_dataset),
        ):
            evaluator = MMLUProHealthEvaluator(streaming=False)
            train_ds, eval_ds = evaluator._load_datasets()

        assert train_ds is None
        assert eval_ds is not None


class TestMMLUProHealthEnvironment:
    """Tests for MMLUProHealth environment construction."""

    def test_environment_construction(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        mmlu_pro_health_mock_dataset: Dataset,
        assert_env_has_basic_fields: Callable,
    ) -> None:
        """Test environment is constructed correctly."""
        with patch(
            "med_reason_evals.data.mmlu_pro_health.load_dataset",
            mock_load_dataset_factory(mmlu_pro_health_mock_dataset),
        ):
            evaluator = MMLUProHealthEvaluator(streaming=False)
            env = evaluator.environment()

            assert_env_has_basic_fields(env)


class TestMMLUProHealthParserConfiguration:
    """Tests for MMLUProHealth parser configuration."""

    def test_default_boxed_format_with_think(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        mmlu_pro_health_mock_dataset: Dataset,
    ) -> None:
        """Verify default format is BOXED with use_think=True."""
        with patch(
            "med_reason_evals.data.mmlu_pro_health.load_dataset",
            mock_load_dataset_factory(mmlu_pro_health_mock_dataset),
        ):
            evaluator = MMLUProHealthEvaluator(streaming=False)

            # MMLUProHealth defaults to use_think=True and BOXED
            assert evaluator.use_think is True
            assert evaluator.answer_format == AnswerFormat.BOXED

    def test_default_uses_thinkparser(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        mmlu_pro_health_mock_dataset: Dataset,
        assert_parser_is_boxed: Callable,
    ) -> None:
        """Verify default uses ThinkParser."""
        with patch(
            "med_reason_evals.data.mmlu_pro_health.load_dataset",
            mock_load_dataset_factory(mmlu_pro_health_mock_dataset),
        ):
            evaluator = MMLUProHealthEvaluator(streaming=False)
            env = evaluator.environment()

            assert_parser_is_boxed(env.parser, has_think=True)


class TestMMLUProHealthRubricConfiguration:
    """Tests for MMLUProHealth rubric configuration."""

    def test_rubric_has_accuracy_reward(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        mmlu_pro_health_mock_dataset: Dataset,
        assert_rubric_has_one_func_weight_one: Callable,
    ) -> None:
        """Verify rubric uses accuracy_reward."""
        with patch(
            "med_reason_evals.data.mmlu_pro_health.load_dataset",
            mock_load_dataset_factory(mmlu_pro_health_mock_dataset),
        ):
            evaluator = MMLUProHealthEvaluator(streaming=False)
            env = evaluator.environment()

            assert_rubric_has_one_func_weight_one(
                env.rubric, func_name="accuracy_reward"
            )
