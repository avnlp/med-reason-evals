"""Tests for PubHealthBench Verifiers evaluator.

Tests cover dataset loading, environment construction, parser configuration,
rubric wiring, and question_type parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import verifiers as vf
from verifiers import JudgeRubric

from med_reason_evals.verifiers.pubhealthbench import PubHealthBenchEvaluator
from med_reason_evals.verifiers.utils.prompts import (
    THINK_XML_SYSTEM_PROMPT,
    XML_SYSTEM_PROMPT,
    AnswerFormat,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestPubHealthBenchDatasetLoading:
    """Tests for PubHealthBench dataset loading."""

    def test_load_datasets_eval_only(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ) -> None:
        """Test that PubHealthBench loads only eval dataset."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(streaming=False)
            train_ds, eval_ds = evaluator._load_datasets()

        assert train_ds is None
        assert eval_ds is not None
        assert "question" in eval_ds.column_names
        assert "answer" in eval_ds.column_names
        assert "info" in eval_ds.column_names


class TestPubHealthBenchEnvironment:
    """Tests for PubHealthBench environment construction."""

    def test_environment_construction(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
        assert_env_has_basic_fields: Callable,
    ) -> None:
        """Test environment is constructed correctly."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(
                judge_api_key="test-key", streaming=False
            )
            env = evaluator.environment()

            assert_env_has_basic_fields(env)

    def test_environment_caching(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ) -> None:
        """Test environment is cached after first call."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(
                judge_api_key="test-key", streaming=False
            )
            env1 = evaluator.environment()
            env2 = evaluator.environment()

            assert env1 is env2


class TestPubHealthBenchParserConfiguration:
    """Tests for PubHealthBench parser configuration."""

    def test_xml_parser_without_think(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ) -> None:
        """Test XML parser without think field."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(streaming=False)
            parser, _ = evaluator._build_parser_and_prompt()

            assert isinstance(parser, vf.XMLParser)
            assert parser.answer_field == "answer"
            field_names = [f[0] for f in parser._fields]
            assert "think" not in field_names
            assert "answer" in field_names

    def test_xml_parser_with_think(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ) -> None:
        """Test XML parser with think field."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(use_think=True, streaming=False)
            parser, _ = evaluator._build_parser_and_prompt()

            assert isinstance(parser, vf.XMLParser)
            field_names = [f[0] for f in parser._fields]
            assert "think" in field_names
            assert "answer" in field_names

    def test_non_xml_format_raises(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ) -> None:
        """Test that non-XML answer format raises ValueError."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(
                answer_format=AnswerFormat.BOXED, streaming=False
            )

            with pytest.raises(ValueError, match="XML"):
                evaluator._build_parser_and_prompt()


class TestPubHealthBenchSystemPrompt:
    """Tests for PubHealthBench system prompt."""

    def test_default_system_prompt(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ) -> None:
        """Test default system prompt."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(streaming=False)
            _, system_prompt = evaluator._build_parser_and_prompt()

            assert system_prompt == XML_SYSTEM_PROMPT

    def test_think_system_prompt(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ) -> None:
        """Test system prompt with use_think=True."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(use_think=True, streaming=False)
            _, system_prompt = evaluator._build_parser_and_prompt()

            assert system_prompt == THINK_XML_SYSTEM_PROMPT

    def test_custom_system_prompt(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ) -> None:
        """Test custom system prompt."""
        custom = "Custom prompt for testing."

        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(system_prompt=custom, streaming=False)
            _, system_prompt = evaluator._build_parser_and_prompt()

            assert system_prompt == custom


class TestPubHealthBenchRubricConfiguration:
    """Tests for PubHealthBench rubric configuration."""

    def test_rubric_is_judge_rubric(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ) -> None:
        """Test rubric is JudgeRubric."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(
                judge_api_key="test-key", streaming=False
            )
            parser, _ = evaluator._build_parser_and_prompt()
            rubric = evaluator._build_rubric(parser)

            assert isinstance(rubric, JudgeRubric)

    def test_judge_prompt_is_question_placeholder(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
        assert_judge_rubric_prompt_is_question: Callable,
    ) -> None:
        """Test judge_prompt is '{question}'."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(
                judge_api_key="test-key", streaming=False
            )
            parser, _ = evaluator._build_parser_and_prompt()
            rubric = evaluator._build_rubric(parser)

            assert_judge_rubric_prompt_is_question(rubric)

    def test_reward_func_is_pubhealthbench_reward(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ) -> None:
        """Test reward function is pubhealthbench_reward."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(
                judge_api_key="test-key", streaming=False
            )
            parser, _ = evaluator._build_parser_and_prompt()
            rubric = evaluator._build_rubric(parser)

            assert len(rubric.funcs) == 1
            assert rubric.funcs[0].__name__ == "pubhealthbench_reward"
            assert rubric.weights[0] == 1.0


class TestPubHealthBenchQuestionType:
    """Tests for PubHealthBench question_type parameter."""

    def test_question_type_all(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ) -> None:
        """Test question_type='all'."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(question_type="all", streaming=False)

            assert evaluator.question_type == "all"

    def test_question_type_mcq(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ) -> None:
        """Test question_type='mcq'."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(question_type="mcq", streaming=False)

            assert evaluator.question_type == "mcq"

    def test_question_type_freeform(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ) -> None:
        """Test question_type='freeform'."""
        with patch(
            "med_reason_evals.data.pubhealthbench.load_dataset",
            mock_load_dataset_factory(pubhealthbench_mock_dataset),
        ):
            evaluator = PubHealthBenchEvaluator(
                question_type="freeform", streaming=False
            )

            assert evaluator.question_type == "freeform"


class TestPubHealthBenchMain:
    """Tests for PubHealthBench main() function."""

    @patch("med_reason_evals.verifiers.pubhealthbench.os.getenv")
    @patch("med_reason_evals.verifiers.pubhealthbench.AsyncOpenAI")
    @patch.object(PubHealthBenchEvaluator, "evaluate", new_callable=AsyncMock)
    @patch.object(PubHealthBenchEvaluator, "_load_datasets")
    async def test_main_function(
        self,
        mock_load_datasets,
        mock_evaluate,
        mock_async_openai,
        mock_getenv,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        pubhealthbench_mock_dataset: Dataset,
    ):
        """Test main() function runs successfully."""
        mock_getenv.return_value = "test-api-key"
        mock_client = MagicMock()
        mock_async_openai.return_value = mock_client
        mock_evaluate.return_value = {"accuracy": 0.85, "total": 100}
        mock_load_datasets.return_value = (None, pubhealthbench_mock_dataset)

        from med_reason_evals.verifiers.pubhealthbench import main

        await main()

        mock_evaluate.assert_awaited_once()
        call_kwargs = mock_evaluate.call_args.kwargs
        assert call_kwargs["client"] == mock_client
        assert call_kwargs["model"] == "openai/gpt-oss-120b"
