"""Tests for MedCaseReasoning Verifiers evaluator.

Tests cover dataset loading, environment construction, parser configuration,
rubric wiring, and judge-specific behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import verifiers as vf
from verifiers import JudgeRubric

from med_reason_evals.verifiers.medcasereasoning import MedCaseReasoningEvaluator


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestMedCaseReasoningDatasetLoading:
    """Tests for MedCaseReasoning dataset loading."""

    def test_load_datasets_returns_both_splits(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medcasereasoning_mock_dataset: Dataset,
    ) -> None:
        """Test that MedCaseReasoning returns both train and eval splits."""
        with patch(
            "med_reason_evals.data.medcasereasoning.load_dataset",
            mock_load_dataset_factory(medcasereasoning_mock_dataset),
        ) as mock_load:
            evaluator = MedCaseReasoningEvaluator()
            train_ds, eval_ds = evaluator._load_datasets()

        assert train_ds is not None
        assert eval_ds is not None
        assert train_ds is not eval_ds
        assert "question" in eval_ds.column_names
        assert "answer" in eval_ds.column_names
        # Each split must be requested with its own name so train/eval
        # separation is real rather than both loading the same split.
        assert mock_load.call_args_list[0].kwargs["split"] == "train"
        assert mock_load.call_args_list[1].kwargs["split"] == "val"

    def test_case_prompt_mapped_to_question(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medcasereasoning_mock_dataset: Dataset,
    ) -> None:
        """Test that case_prompt field is mapped to question."""
        with patch(
            "med_reason_evals.data.medcasereasoning.load_dataset",
            mock_load_dataset_factory(medcasereasoning_mock_dataset),
        ):
            evaluator = MedCaseReasoningEvaluator()
            _, eval_ds = evaluator._load_datasets()

            assert "question" in eval_ds.column_names


class TestMedCaseReasoningEnvironment:
    """Tests for MedCaseReasoning environment construction."""

    def test_environment_construction(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medcasereasoning_mock_dataset: Dataset,
        assert_env_has_basic_fields: Callable,
    ) -> None:
        """Test environment is constructed correctly."""
        with patch(
            "med_reason_evals.data.medcasereasoning.load_dataset",
            mock_load_dataset_factory(medcasereasoning_mock_dataset),
        ):
            evaluator = MedCaseReasoningEvaluator(judge_api_key="test-key")
            env = evaluator.environment()

            assert_env_has_basic_fields(env)


class TestMedCaseReasoningParserConfiguration:
    """Tests for MedCaseReasoning parser configuration."""

    def test_xml_parser_with_think(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medcasereasoning_mock_dataset: Dataset,
    ) -> None:
        """Test XML parser has think and answer fields."""
        with patch(
            "med_reason_evals.data.medcasereasoning.load_dataset",
            mock_load_dataset_factory(medcasereasoning_mock_dataset),
        ):
            evaluator = MedCaseReasoningEvaluator()
            parser, _ = evaluator._build_parser_and_prompt()

            assert isinstance(parser, vf.XMLParser)
            assert parser.answer_field == "answer"
            field_names = [f[0] for f in parser._fields]
            assert "think" in field_names
            assert "answer" in field_names

    def test_system_prompt_mentions_diagnosis(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medcasereasoning_mock_dataset: Dataset,
    ) -> None:
        """Test system prompt mentions diagnosis."""
        with patch(
            "med_reason_evals.data.medcasereasoning.load_dataset",
            mock_load_dataset_factory(medcasereasoning_mock_dataset),
        ):
            evaluator = MedCaseReasoningEvaluator()
            _, system_prompt = evaluator._build_parser_and_prompt()

            assert "diagnosis" in system_prompt.lower()


class TestMedCaseReasoningRubricConfiguration:
    """Tests for MedCaseReasoning rubric configuration."""

    def test_rubric_is_judge_rubric(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medcasereasoning_mock_dataset: Dataset,
    ) -> None:
        """Test rubric is JudgeRubric."""
        with patch(
            "med_reason_evals.data.medcasereasoning.load_dataset",
            mock_load_dataset_factory(medcasereasoning_mock_dataset),
        ):
            evaluator = MedCaseReasoningEvaluator(judge_api_key="test-key")
            parser, _ = evaluator._build_parser_and_prompt()
            rubric = evaluator._build_rubric(parser)

            assert isinstance(rubric, JudgeRubric)

    def test_judge_prompt_is_question_placeholder(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medcasereasoning_mock_dataset: Dataset,
        assert_judge_rubric_prompt_is_question: Callable,
    ) -> None:
        """Test judge_prompt is '{question}'."""
        with patch(
            "med_reason_evals.data.medcasereasoning.load_dataset",
            mock_load_dataset_factory(medcasereasoning_mock_dataset),
        ):
            evaluator = MedCaseReasoningEvaluator(judge_api_key="test-key")
            parser, _ = evaluator._build_parser_and_prompt()
            rubric = evaluator._build_rubric(parser)

            assert_judge_rubric_prompt_is_question(rubric)

    def test_reward_func_is_binary_judge(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medcasereasoning_mock_dataset: Dataset,
    ) -> None:
        """Test reward function is binary judge."""
        with patch(
            "med_reason_evals.data.medcasereasoning.load_dataset",
            mock_load_dataset_factory(medcasereasoning_mock_dataset),
        ):
            evaluator = MedCaseReasoningEvaluator(judge_api_key="test-key")
            parser, _ = evaluator._build_parser_and_prompt()
            rubric = evaluator._build_rubric(parser)

            assert len(rubric.funcs) == 1
            assert rubric.weights[0] == 1.0
            assert "binary_judge_reward" in rubric.funcs[0].__name__
