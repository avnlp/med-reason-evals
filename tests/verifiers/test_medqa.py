"""Tests for MedQA Verifiers evaluator.

Tests cover dataset loading, environment construction, parser configuration,
rubric wiring, and configuration options.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from med_reason_evals.verifiers.medqa import MedQAEvaluator
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestMedQADatasetLoading:
    """Tests for MedQA dataset loading."""

    def test_load_datasets_returns_train_and_eval(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
    ) -> None:
        """Test that _load_datasets returns both train and eval datasets."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(streaming=False)
            train_ds, eval_ds = evaluator._load_datasets()

        assert train_ds is not None, "Train dataset should not be None"
        assert eval_ds is not None, "Eval dataset should not be None"
        assert "question" in eval_ds.column_names
        assert "answer" in eval_ds.column_names
        assert "info" in eval_ds.column_names

    def test_load_datasets_returns_correct_mapping(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
    ) -> None:
        """Test that _load_datasets correctly maps dataset fields."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(streaming=False)
            train_ds, eval_ds = evaluator._load_datasets()

        # Both should be returned with mapped fields
        assert train_ds is not None
        assert eval_ds is not None


class TestMedQAEnvironment:
    """Tests for MedQA environment construction."""

    def test_environment_returns_singleturn_env(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
        assert_env_has_basic_fields: Callable,
    ) -> None:
        """Verify that evaluator.environment() returns a vf.SingleTurnEnv."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert_env_has_basic_fields(env)

    def test_environment_is_cached(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
    ) -> None:
        """Verify that multiple calls to environment() return the same instance."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(streaming=False)
            env1 = evaluator.environment()
            env2 = evaluator.environment()

            assert env1 is env2, "Environment should be cached"


class TestMedQAParserConfiguration:
    """Tests for MedQA parser configuration."""

    def test_default_xml_format_uses_xmlparser(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
        assert_parser_is_xml: Callable,
    ) -> None:
        """Verify default format is XML and uses XMLParser."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert_parser_is_xml(env.parser, has_think=False)

    def test_xml_format_with_think_has_think_field(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
        assert_parser_is_xml: Callable,
    ) -> None:
        """Verify XML format with use_think=True has think field."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(use_think=True, streaming=False)
            env = evaluator.environment()

            assert_parser_is_xml(env.parser, has_think=True)

    def test_boxed_format_uses_parser(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
        assert_parser_is_boxed: Callable,
    ) -> None:
        """Verify BOXED format uses vf.Parser."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(
                answer_format=AnswerFormat.BOXED, streaming=False
            )
            env = evaluator.environment()

            assert_parser_is_boxed(env.parser, has_think=False)

    def test_boxed_format_with_think_uses_thinkparser(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
        assert_parser_is_boxed: Callable,
    ) -> None:
        """Verify BOXED format with use_think=True uses ThinkParser."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(
                answer_format=AnswerFormat.BOXED, use_think=True, streaming=False
            )
            env = evaluator.environment()

            assert_parser_is_boxed(env.parser, has_think=True)

    def test_string_xml_format_works(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
        assert_parser_is_xml: Callable,
    ) -> None:
        """Verify string 'xml' works as answer_format."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(answer_format="xml", streaming=False)
            env = evaluator.environment()

            assert_parser_is_xml(env.parser)

    def test_string_boxed_format_works(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
        assert_parser_is_boxed: Callable,
    ) -> None:
        """Verify string 'boxed' works as answer_format."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(answer_format="boxed", streaming=False)
            env = evaluator.environment()

            assert_parser_is_boxed(env.parser)


class TestMedQARubricConfiguration:
    """Tests for MedQA rubric configuration."""

    def test_rubric_has_accuracy_reward(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
        assert_rubric_has_one_func_weight_one: Callable,
    ) -> None:
        """Verify that the rubric uses accuracy_reward function."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert_rubric_has_one_func_weight_one(
                env.rubric, func_name="accuracy_reward"
            )

    def test_rubric_has_parser_reference(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
    ) -> None:
        """Verify that the rubric has reference to the parser."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(streaming=False)
            env = evaluator.environment()

            evaluator_rubric = env.rubric.rubrics[0]
            assert evaluator_rubric.parser is env.parser


class TestMedQASystemPrompt:
    """Tests for MedQA system prompt configuration."""

    def test_default_system_prompt_mentions_xml(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
    ) -> None:
        """Verify default system prompt for XML format mentions answer tags."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert "answer" in env.system_prompt.lower()

    def test_boxed_system_prompt_mentions_boxed(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
    ) -> None:
        """Verify BOXED format prompt mentions boxed notation."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(
                answer_format=AnswerFormat.BOXED, streaming=False
            )
            env = evaluator.environment()

            assert "boxed" in env.system_prompt.lower()

    def test_custom_system_prompt_is_used(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
    ) -> None:
        """Verify that custom system prompt overrides default."""
        custom_prompt = "You are a medical expert. Answer carefully."

        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(system_prompt=custom_prompt, streaming=False)
            env = evaluator.environment()

            assert env.system_prompt == custom_prompt


class TestMedQAParameterStorage:
    """Tests for MedQA evaluator parameter storage."""

    def test_default_parameters(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
    ) -> None:
        """Verify default parameter values are stored correctly."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(streaming=False)

            assert evaluator.use_think is False
            assert evaluator.answer_format == AnswerFormat.XML
            assert evaluator.streaming is False

    def test_streaming_none_defaults_to_false(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
    ) -> None:
        """Verify streaming=None defaults to False for verifiers."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(streaming=None)

            assert evaluator.streaming is False

    def test_explicit_use_think_stored(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
    ) -> None:
        """Verify use_think parameter is stored."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(use_think=True, streaming=False)

            assert evaluator.use_think is True


class TestMedQAInheritance:
    """Tests for MedQA inheritance and base class behavior."""

    def test_inherits_from_basemcqevaluator(self) -> None:
        """Verify MedQAEvaluator inherits from BaseMCQEvaluator."""
        from med_reason_evals.verifiers.base import BaseMCQEvaluator

        assert issubclass(MedQAEvaluator, BaseMCQEvaluator)

    def test_has_load_datasets_method(self) -> None:
        """Verify MedQAEvaluator implements _load_datasets method."""
        assert hasattr(MedQAEvaluator, "_load_datasets")
        assert callable(MedQAEvaluator._load_datasets)


class TestMedQAInvalidConfiguration:
    """Tests for MedQA invalid configuration handling."""

    def test_invalid_answer_format_raises_error(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medqa_mock_dataset: Dataset,
    ) -> None:
        """Verify that invalid answer_format raises ValueError."""
        with patch(
            "med_reason_evals.data.medqa.load_dataset",
            mock_load_dataset_factory(medqa_mock_dataset),
        ):
            evaluator = MedQAEvaluator(answer_format="invalid", streaming=False)

            with pytest.raises(ValueError, match="is not a valid AnswerFormat"):
                evaluator.environment()
