"""Tests for MedBullets Verifiers evaluator.

Tests cover dataset loading, environment construction, parser configuration,
rubric wiring, and MedBullets-specific num_options parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from med_reason_evals.verifiers.medbullets import MedBulletsEvaluator
from med_reason_evals.verifiers.utils.prompts import AnswerFormat


if TYPE_CHECKING:
    from collections.abc import Callable


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


@pytest.fixture
def medbullets_op4_mock_dataset() -> Dataset:
    """Mock rows shaped like the upstream ``op4_test`` split.

    The upstream ``options`` struct always carries keys A-E; on the
    four-option split the ``E`` entry is null.
    """
    return Dataset.from_dict(
        {
            "question": [
                "A 30-year-old woman presents with fatigue. Labs show low hemoglobin.",
                "What is the most common cause of community-acquired pneumonia?",
            ],
            "options": [
                {
                    "A": "Iron deficiency",
                    "B": "B12 deficiency",
                    "C": "Folate deficiency",
                    "D": "Anemia of chronic disease",
                    "E": None,
                },
                {
                    "A": "S. pneumoniae",
                    "B": "H. influenzae",
                    "C": "M. pneumoniae",
                    "D": "S. aureus",
                    "E": None,
                },
            ],
            "answer": ["A", "A"],
        }
    )


@pytest.fixture
def medbullets_op5_mock_dataset() -> Dataset:
    """Mock rows shaped like the upstream ``op5_test`` split (A-E populated).

    The second row's gold answer is ``E``, the choice that exists only in the
    five-option variant.
    """
    return Dataset.from_dict(
        {
            "question": [
                "A 30-year-old woman presents with fatigue. Labs show low hemoglobin.",
                "What is the most common cause of community-acquired pneumonia?",
            ],
            "options": [
                {
                    "A": "Iron deficiency",
                    "B": "B12 deficiency",
                    "C": "Folate deficiency",
                    "D": "Anemia of chronic disease",
                    "E": "Sideroblastic anemia",
                },
                {
                    "A": "H. influenzae",
                    "B": "M. pneumoniae",
                    "C": "S. aureus",
                    "D": "K. pneumoniae",
                    "E": "S. pneumoniae",
                },
            ],
            "answer": ["A", "E"],
        }
    )


class TestMedBulletsNumOptions:
    """Tests for the MedBullets num_options parameter.

    These drive ``environment()`` rather than only reading the attribute, so
    split selection, row validation, and prompt mapping are all exercised.
    """

    def test_num_options_4_selects_op4_split_and_drops_option_e(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medbullets_op4_mock_dataset: Dataset,
    ) -> None:
        """Verify num_options=4 loads op4_test and strips the null E choice."""
        mock_load_dataset = mock_load_dataset_factory(medbullets_op4_mock_dataset)
        with patch("med_reason_evals.data.medbullets.load_dataset", mock_load_dataset):
            evaluator = MedBulletsEvaluator(num_options=4, streaming=False)
            env = evaluator.environment()

        assert evaluator.num_options == 4
        assert mock_load_dataset.call_args.kwargs["split"] == "op4_test"

        eval_ds = env.eval_dataset
        assert len(eval_ds) == 2
        for row in eval_ds:
            assert "D. " in row["question"]
            assert "E. " not in row["question"]
            assert row["answer"] in {"A", "B", "C", "D"}

    def test_num_options_5_selects_op5_split_and_keeps_option_e(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medbullets_op5_mock_dataset: Dataset,
    ) -> None:
        """Verify num_options=5 loads op5_test and accepts E as a gold answer."""
        mock_load_dataset = mock_load_dataset_factory(medbullets_op5_mock_dataset)
        with patch("med_reason_evals.data.medbullets.load_dataset", mock_load_dataset):
            evaluator = MedBulletsEvaluator(num_options=5, streaming=False)
            env = evaluator.environment()

        assert evaluator.num_options == 5
        assert mock_load_dataset.call_args.kwargs["split"] == "op5_test"

        eval_ds = env.eval_dataset
        assert len(eval_ds) == 2
        for row in eval_ds:
            assert "E. " in row["question"]
        # The E-answered row survives only under the five-option variant.
        assert eval_ds[1]["answer"] == "E"
        assert eval_ds[1]["info"]["answer_text"] == "S. pneumoniae"

    def test_num_options_4_filters_rows_answered_e(
        self,
        mock_load_dataset_factory: Callable[[Dataset], MagicMock],
        medbullets_op5_mock_dataset: Dataset,
    ) -> None:
        """Verify a gold answer of E is dropped under the four-option variant.

        Feeding five-option rows to a four-option evaluator strips option E
        from the prompt, so the E-answered row would otherwise carry ground
        truth with no matching choice.
        """
        mock_load_dataset = mock_load_dataset_factory(medbullets_op5_mock_dataset)
        with patch("med_reason_evals.data.medbullets.load_dataset", mock_load_dataset):
            evaluator = MedBulletsEvaluator(num_options=4, streaming=False)
            env = evaluator.environment()

        eval_ds = env.eval_dataset
        assert len(eval_ds) == 1
        assert eval_ds[0]["answer"] == "A"

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
