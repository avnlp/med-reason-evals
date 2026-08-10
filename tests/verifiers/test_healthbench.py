"""Tests for HealthBench Verifiers evaluator.

Tests cover dataset loading, environment construction, parser configuration,
rubric wiring, and judge-specific behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import verifiers as vf
from verifiers import JudgeRubric

from med_reason_evals.verifiers.healthbench import HealthBenchEvaluator


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestHealthBenchDatasetLoading:
    """Tests for HealthBench dataset loading."""

    def test_load_datasets_eval_only(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        healthbench_mock_dataset: Dataset,
    ) -> None:
        """Test that HealthBench loads only eval dataset."""
        with patch(
            "med_reason_evals.data.healthbench.load_dataset",
            mock_load_dataset_factory(healthbench_mock_dataset),
        ):
            evaluator = HealthBenchEvaluator(streaming=False)
            train_ds, eval_ds = evaluator._load_datasets()

        assert train_ds is None
        assert eval_ds is not None
        assert "question" in eval_ds.column_names
        assert "answer" in eval_ds.column_names

    def test_streaming_rejected(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        healthbench_mock_dataset: Dataset,
    ) -> None:
        """Test that streaming=True raises because verifiers need random access."""
        with (
            patch(
                "med_reason_evals.data.healthbench.load_dataset",
                mock_load_dataset_factory(healthbench_mock_dataset),
            ),
            pytest.raises(ValueError, match="do not support streaming"),
        ):
            HealthBenchEvaluator(streaming=True)


class TestHealthBenchEnvironment:
    """Tests for HealthBench environment construction."""

    def test_environment_construction(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        healthbench_mock_dataset: Dataset,
        assert_env_has_basic_fields: Callable,
    ) -> None:
        """Test environment is constructed correctly."""
        with patch(
            "med_reason_evals.data.healthbench.load_dataset",
            mock_load_dataset_factory(healthbench_mock_dataset),
        ):
            evaluator = HealthBenchEvaluator(streaming=False)
            with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
                env = evaluator.environment()

            assert_env_has_basic_fields(env)


class TestHealthBenchParserConfiguration:
    """Tests for HealthBench parser configuration."""

    def test_basic_parser(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        healthbench_mock_dataset: Dataset,
    ) -> None:
        """Test basic Parser (not XMLParser)."""
        with patch(
            "med_reason_evals.data.healthbench.load_dataset",
            mock_load_dataset_factory(healthbench_mock_dataset),
        ):
            evaluator = HealthBenchEvaluator(streaming=False)
            parser, _ = evaluator._build_parser_and_prompt()

            assert isinstance(parser, vf.Parser)
            assert not isinstance(parser, vf.XMLParser)

    def test_empty_system_prompt(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        healthbench_mock_dataset: Dataset,
    ) -> None:
        """Test system prompt is empty."""
        with patch(
            "med_reason_evals.data.healthbench.load_dataset",
            mock_load_dataset_factory(healthbench_mock_dataset),
        ):
            evaluator = HealthBenchEvaluator(streaming=False)
            _, system_prompt = evaluator._build_parser_and_prompt()

            assert system_prompt == ""


class TestHealthBenchRubricConfiguration:
    """Tests for HealthBench rubric configuration."""

    def test_rubric_is_judge_rubric(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        healthbench_mock_dataset: Dataset,
    ) -> None:
        """Test rubric is JudgeRubric."""
        with patch(
            "med_reason_evals.data.healthbench.load_dataset",
            mock_load_dataset_factory(healthbench_mock_dataset),
        ):
            evaluator = HealthBenchEvaluator(streaming=False)
            parser, _ = evaluator._build_parser_and_prompt()
            with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
                rubric = evaluator._build_rubric(parser)

            assert isinstance(rubric, JudgeRubric)

    def test_judge_prompt_is_question_placeholder(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        healthbench_mock_dataset: Dataset,
        assert_judge_rubric_prompt_is_question: Callable,
    ) -> None:
        """Test judge_prompt is '{question}'."""
        with patch(
            "med_reason_evals.data.healthbench.load_dataset",
            mock_load_dataset_factory(healthbench_mock_dataset),
        ):
            evaluator = HealthBenchEvaluator(streaming=False)
            parser, _ = evaluator._build_parser_and_prompt()
            with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
                rubric = evaluator._build_rubric(parser)

            assert_judge_rubric_prompt_is_question(rubric)

    def test_reward_func_is_async(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        healthbench_mock_dataset: Dataset,
        assert_async_reward_func: Callable,
    ) -> None:
        """Test reward function is async."""
        with patch(
            "med_reason_evals.data.healthbench.load_dataset",
            mock_load_dataset_factory(healthbench_mock_dataset),
        ):
            evaluator = HealthBenchEvaluator(streaming=False)
            parser, _ = evaluator._build_parser_and_prompt()
            with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
                rubric = evaluator._build_rubric(parser)

            assert len(rubric.funcs) == 1
            assert rubric.weights[0] == 1.0
            assert_async_reward_func(rubric)


class TestHealthBenchDifficulty:
    """Tests for HealthBench difficulty parameter."""

    def test_difficulty_regular(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        healthbench_mock_dataset: Dataset,
    ) -> None:
        """Test difficulty='regular'."""
        with patch(
            "med_reason_evals.data.healthbench.load_dataset",
            mock_load_dataset_factory(healthbench_mock_dataset),
        ):
            evaluator = HealthBenchEvaluator(difficulty="regular", streaming=False)

            assert evaluator.difficulty == "regular"

    def test_difficulty_hard(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        healthbench_mock_dataset: Dataset,
    ) -> None:
        """Test difficulty='hard'."""
        with patch(
            "med_reason_evals.data.healthbench.load_dataset",
            mock_load_dataset_factory(healthbench_mock_dataset),
        ):
            evaluator = HealthBenchEvaluator(difficulty="hard", streaming=False)

            assert evaluator.difficulty == "hard"

    def test_difficulty_consensus(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        healthbench_mock_dataset: Dataset,
    ) -> None:
        """Test difficulty='consensus'."""
        with patch(
            "med_reason_evals.data.healthbench.load_dataset",
            mock_load_dataset_factory(healthbench_mock_dataset),
        ):
            evaluator = HealthBenchEvaluator(difficulty="consensus", streaming=False)

            assert evaluator.difficulty == "consensus"

    def test_max_parallel_judges_default(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        healthbench_mock_dataset: Dataset,
    ) -> None:
        """Test default max_parallel_judges."""
        with patch(
            "med_reason_evals.data.healthbench.load_dataset",
            mock_load_dataset_factory(healthbench_mock_dataset),
        ):
            evaluator = HealthBenchEvaluator(streaming=False)

            assert evaluator.max_parallel_judges == 5

    def test_max_parallel_judges_custom(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        healthbench_mock_dataset: Dataset,
    ) -> None:
        """Test custom max_parallel_judges."""
        with patch(
            "med_reason_evals.data.healthbench.load_dataset",
            mock_load_dataset_factory(healthbench_mock_dataset),
        ):
            evaluator = HealthBenchEvaluator(max_parallel_judges=10, streaming=False)

            assert evaluator.max_parallel_judges == 10
