"""Tests for MedXpertQA Verifiers evaluator.

Tests cover dataset loading, environment construction, parser configuration,
rubric wiring, and question_type parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from med_reason_evals.verifiers.medxpertqa import MedXpertQAEvaluator
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestMedXpertQADatasetLoading:
    """Tests for MedXpertQA dataset loading."""

    def test_load_datasets_no_train_split(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medxpertqa_mock_dataset: Dataset,
    ) -> None:
        """Test that MedXpertQA has no train split."""
        with patch(
            "med_reason_evals.data.medxpertqa.load_dataset",
            mock_load_dataset_factory(medxpertqa_mock_dataset),
        ):
            evaluator = MedXpertQAEvaluator(streaming=False)
            train_ds, eval_ds = evaluator._load_datasets()

        assert train_ds is None
        assert eval_ds is not None


class TestMedXpertQAEnvironment:
    """Tests for MedXpertQA environment construction."""

    def test_environment_construction(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medxpertqa_mock_dataset: Dataset,
        assert_env_has_basic_fields: Callable,
    ) -> None:
        """Test environment is constructed correctly."""
        with patch(
            "med_reason_evals.data.medxpertqa.load_dataset",
            mock_load_dataset_factory(medxpertqa_mock_dataset),
        ):
            evaluator = MedXpertQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert_env_has_basic_fields(env)


class TestMedXpertQAParserConfiguration:
    """Tests for MedXpertQA parser configuration."""

    def test_default_xml_format(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medxpertqa_mock_dataset: Dataset,
        assert_parser_is_xml: Callable,
    ) -> None:
        """Verify default format is XML."""
        with patch(
            "med_reason_evals.data.medxpertqa.load_dataset",
            mock_load_dataset_factory(medxpertqa_mock_dataset),
        ):
            evaluator = MedXpertQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert evaluator.answer_format == AnswerFormat.XML
            assert_parser_is_xml(env.parser, has_think=False)


class TestMedXpertQARubricConfiguration:
    """Tests for MedXpertQA rubric configuration."""

    def test_rubric_has_accuracy_reward(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medxpertqa_mock_dataset: Dataset,
        assert_rubric_has_one_func_weight_one: Callable,
    ) -> None:
        """Verify rubric uses accuracy_reward."""
        with patch(
            "med_reason_evals.data.medxpertqa.load_dataset",
            mock_load_dataset_factory(medxpertqa_mock_dataset),
        ):
            evaluator = MedXpertQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert_rubric_has_one_func_weight_one(
                env.rubric, func_name="accuracy_reward"
            )


class TestMedXpertQAQuestionType:
    """Tests for MedXpertQA question_type parameter."""

    @pytest.mark.parametrize("question_type", ["all", "reasoning", "understanding"])
    def test_question_type_forwarded_to_dataset(self, question_type: str) -> None:
        """Verify question_type is stored and forwarded to MedXpertQADataset.

        Patching the adapter class asserts that the evaluator actually passes
        the parameter through to the dataset loader, so the value is not just
        stashed on an attribute.
        """
        with patch(
            "med_reason_evals.verifiers.medxpertqa.MedXpertQADataset"
        ) as mock_dataset_cls:
            evaluator = MedXpertQAEvaluator(
                question_type=question_type, streaming=False
            )
            train_ds, eval_ds = evaluator._load_datasets()

        assert evaluator.question_type == question_type
        assert train_ds is None
        assert eval_ds is not None
        mock_dataset_cls.assert_called_once_with(
            split="test",
            streaming=False,
            question_type=question_type,
        )
