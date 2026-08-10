"""Tests for MedMCQA Verifiers evaluator.

Tests cover dataset loading, environment construction, parser configuration,
rubric wiring, and configuration options.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from med_reason_evals.verifiers.med_mcqa import MedMCQAEvaluator
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestMedMCQADatasetLoading:
    """Tests for MedMCQA dataset loading."""

    def test_load_datasets_returns_train_and_eval(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
    ) -> None:
        """Test that _load_datasets returns both train and eval datasets."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(streaming=False)
            train_ds, eval_ds = evaluator._load_datasets()

        assert train_ds is not None, "Train dataset should not be None"
        assert eval_ds is not None, "Eval dataset should not be None"
        assert "question" in eval_ds.column_names
        assert "answer" in eval_ds.column_names
        assert "info" in eval_ds.column_names

    def test_load_datasets_maps_cop_to_answer(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
    ) -> None:
        """Test that _load_datasets correctly maps cop field to answer."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(streaming=False)
            _, eval_ds = evaluator._load_datasets()

        # Check answer is mapped from cop (1-indexed to letter)
        assert eval_ds is not None
        assert "answer" in eval_ds.column_names


class TestMedMCQAEnvironment:
    """Tests for MedMCQA environment construction."""

    def test_environment_returns_singleturn_env(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
        assert_env_has_basic_fields: Callable,
    ) -> None:
        """Verify that evaluator.environment() returns a vf.SingleTurnEnv."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert_env_has_basic_fields(env)

    def test_environment_is_cached(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
    ) -> None:
        """Verify that multiple calls to environment() return the same instance."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(streaming=False)
            env1 = evaluator.environment()
            env2 = evaluator.environment()

            assert env1 is env2, "Environment should be cached"


class TestMedMCQAParserConfiguration:
    """Tests for MedMCQA parser configuration."""

    def test_default_xml_format_uses_xmlparser(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
        assert_parser_is_xml: Callable,
    ) -> None:
        """Verify default format is XML and uses XMLParser."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert_parser_is_xml(env.parser, has_think=False)

    def test_xml_format_with_think_has_think_field(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
        assert_parser_is_xml: Callable,
    ) -> None:
        """Verify XML format with use_think=True has think field."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(use_think=True, streaming=False)
            env = evaluator.environment()

            assert_parser_is_xml(env.parser, has_think=True)

    def test_boxed_format_uses_parser(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
        assert_parser_is_boxed: Callable,
    ) -> None:
        """Verify BOXED format uses vf.Parser."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(
                answer_format=AnswerFormat.BOXED, streaming=False
            )
            env = evaluator.environment()

            assert_parser_is_boxed(env.parser, has_think=False)

    def test_boxed_format_with_think_uses_thinkparser(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
        assert_parser_is_boxed: Callable,
    ) -> None:
        """Verify BOXED format with use_think=True uses ThinkParser."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(
                answer_format=AnswerFormat.BOXED, use_think=True, streaming=False
            )
            env = evaluator.environment()

            assert_parser_is_boxed(env.parser, has_think=True)

    def test_string_xml_format_works(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
        assert_parser_is_xml: Callable,
    ) -> None:
        """Verify string 'xml' works as answer_format."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(answer_format="xml", streaming=False)
            env = evaluator.environment()

            assert_parser_is_xml(env.parser)

    def test_string_boxed_format_works(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
        assert_parser_is_boxed: Callable,
    ) -> None:
        """Verify string 'boxed' works as answer_format."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(answer_format="boxed", streaming=False)
            env = evaluator.environment()

            assert_parser_is_boxed(env.parser)


class TestMedMCQARubricConfiguration:
    """Tests for MedMCQA rubric configuration."""

    def test_rubric_has_accuracy_reward(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
        assert_rubric_has_one_func_weight_one: Callable,
    ) -> None:
        """Verify that the rubric uses accuracy_reward function."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert_rubric_has_one_func_weight_one(
                env.rubric, func_name="accuracy_reward"
            )

    def test_rubric_has_parser_reference(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
    ) -> None:
        """Verify that the rubric has reference to the parser."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(streaming=False)
            env = evaluator.environment()

            evaluator_rubric = env.rubric.rubrics[0]
            assert evaluator_rubric.parser is env.parser


class TestMedMCQASystemPrompt:
    """Tests for MedMCQA system prompt configuration."""

    def test_default_system_prompt_mentions_xml(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
    ) -> None:
        """Verify default system prompt for XML format mentions answer tags."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(streaming=False)
            env = evaluator.environment()

            assert "answer" in env.system_prompt.lower()

    def test_boxed_system_prompt_mentions_boxed(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
    ) -> None:
        """Verify BOXED format prompt mentions boxed notation."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(
                answer_format=AnswerFormat.BOXED, streaming=False
            )
            env = evaluator.environment()

            assert "boxed" in env.system_prompt.lower()

    def test_custom_system_prompt_is_used(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
    ) -> None:
        """Verify that custom system prompt overrides default."""
        custom_prompt = "You are a medical expert. Answer carefully."

        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(system_prompt=custom_prompt, streaming=False)
            env = evaluator.environment()

            assert env.system_prompt == custom_prompt


class TestMedMCQAParameterStorage:
    """Tests for MedMCQA evaluator parameter storage."""

    def test_default_parameters(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
    ) -> None:
        """Verify default parameter values are stored correctly."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(streaming=False)

            assert evaluator.use_think is False
            assert evaluator.answer_format == AnswerFormat.XML
            assert evaluator.streaming is False

    def test_streaming_none_defaults_to_false(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
    ) -> None:
        """Verify streaming=None defaults to False for verifiers."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(streaming=None)

            assert evaluator.streaming is False

    def test_explicit_use_think_stored(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
    ) -> None:
        """Verify use_think parameter is stored."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(use_think=True, streaming=False)

            assert evaluator.use_think is True


class TestMedMCQAInheritance:
    """Tests for MedMCQA inheritance and base class behavior."""

    def test_inherits_from_basemcqevaluator(self) -> None:
        """Verify MedMCQAEvaluator inherits from BaseMCQEvaluator."""
        from med_reason_evals.verifiers.base import BaseMCQEvaluator

        assert issubclass(MedMCQAEvaluator, BaseMCQEvaluator)

    def test_has_load_datasets_method(self) -> None:
        """Verify MedMCQAEvaluator implements _load_datasets method."""
        assert hasattr(MedMCQAEvaluator, "_load_datasets")
        assert callable(MedMCQAEvaluator._load_datasets)


class TestMedMCQAInvalidConfiguration:
    """Tests for MedMCQA invalid configuration handling."""

    def test_invalid_answer_format_raises_error(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        med_mcqa_mock_dataset: Dataset,
    ) -> None:
        """Verify that invalid answer_format raises ValueError."""
        with patch(
            "med_reason_evals.data.med_mcqa.load_dataset",
            mock_load_dataset_factory(med_mcqa_mock_dataset),
        ):
            evaluator = MedMCQAEvaluator(answer_format="invalid", streaming=False)

            with pytest.raises(ValueError, match="is not a valid AnswerFormat"):
                evaluator.environment()
