"""Tests for PubMedQA Verifiers evaluator.

Tests cover dataset loading, environment construction, parser configuration,
rubric wiring, and subset parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from med_reason_evals.verifiers.pubmedqa import PubMedQAEvaluator
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestPubMedQADatasetLoading:
    """Tests for PubMedQA dataset loading."""

    def test_load_datasets_returns_eval_only(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubmedqa_mock_dataset: Dataset,
    ) -> None:
        """Test that PubMedQA returns eval dataset only (no training split)."""
        with patch(
            "med_reason_evals.data.pubmedqa.load_dataset",
            mock_load_dataset_factory(pubmedqa_mock_dataset),
        ):
            evaluator = PubMedQAEvaluator(streaming=False)
            train_ds, eval_ds = evaluator._load_datasets()

        assert train_ds is None
        assert eval_ds is not None


class TestPubMedQAEnvironment:
    """Tests for PubMedQA environment construction."""

    def test_environment_construction(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubmedqa_mock_dataset: Dataset,
        assert_env_has_basic_fields: Callable,
    ) -> None:
        """Test environment is constructed correctly."""
        with patch(
            "med_reason_evals.data.pubmedqa.load_dataset",
            mock_load_dataset_factory(pubmedqa_mock_dataset),
        ):
            evaluator = PubMedQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert_env_has_basic_fields(env)


class TestPubMedQAParserConfiguration:
    """Tests for PubMedQA parser configuration."""

    def test_default_xml_format(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubmedqa_mock_dataset: Dataset,
        assert_parser_is_xml: Callable,
    ) -> None:
        """Verify default format is XML."""
        with patch(
            "med_reason_evals.data.pubmedqa.load_dataset",
            mock_load_dataset_factory(pubmedqa_mock_dataset),
        ):
            evaluator = PubMedQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert evaluator.answer_format == AnswerFormat.XML
            assert_parser_is_xml(env.parser, has_think=False)


class TestPubMedQARubricConfiguration:
    """Tests for PubMedQA rubric configuration."""

    def test_rubric_has_accuracy_reward(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubmedqa_mock_dataset: Dataset,
        assert_rubric_has_one_func_weight_one: Callable,
    ) -> None:
        """Verify rubric uses accuracy_reward."""
        with patch(
            "med_reason_evals.data.pubmedqa.load_dataset",
            mock_load_dataset_factory(pubmedqa_mock_dataset),
        ):
            evaluator = PubMedQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert_rubric_has_one_func_weight_one(
                env.rubric, func_name="accuracy_reward"
            )


class TestPubMedQASubset:
    """Tests for PubMedQA subset parameter."""

    def test_evaluator_creation(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubmedqa_mock_dataset: Dataset,
    ) -> None:
        """Verify evaluator can be created."""
        with patch(
            "med_reason_evals.data.pubmedqa.load_dataset",
            mock_load_dataset_factory(pubmedqa_mock_dataset),
        ):
            # Verifiers evaluator doesn't accept subset,
            # uses hardcoded values internally
            evaluator = PubMedQAEvaluator(streaming=False)
            assert evaluator is not None
