"""Tests for PubHealthBench Verl evaluator.

Tests cover main() function, dataset loading, result building, and _evaluate_example.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from med_reason_evals.verl import PubHealthBenchEvaluator


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


class TestPubHealthBenchMain:
    """Tests for PubHealthBench main() function."""

    async def test_main_prints_results(self, mocker):
        """Test main() function prints results correctly."""
        mock_print = mocker.patch("builtins.print")

        mock_evaluator_class = mocker.patch(
            "med_reason_evals.verl.pubhealthbench.PubHealthBenchEvaluator"
        )
        mock_evaluator = MagicMock()
        mock_results = {
            "dataset": "pubhealthbench",
            "split": "test",
            "question_type": "all",
            "num_examples": 100,
            "avg_score": 0.68,
        }
        mock_evaluator.evaluate = AsyncMock(return_value=mock_results)
        mock_evaluator_class.return_value = mock_evaluator

        from med_reason_evals.verl.pubhealthbench import main

        await main()

        mock_evaluator_class.assert_called_once()
        mock_evaluator.evaluate.assert_called_once_with(num_examples=100)
        mock_print.assert_called_once()
        assert "PubHealthBench Verl Results:" in mock_print.call_args[0][0]


class TestPubHealthBenchLoadDataset:
    """Tests for PubHealthBench _load_dataset method."""

    def test_load_dataset(self, mocker, pubhealthbench_verl_dataset: Dataset):
        """Test _load_dataset method."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.pubhealthbench.PubHealthBenchDataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = pubhealthbench_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = PubHealthBenchEvaluator(split="train", question_type="mcq")
            result = evaluator._load_dataset()

        mock_dataset_class.assert_called_once_with(
            split="train", question_type="mcq", streaming=True
        )
        assert result == mock_verl_dataset

    def test_load_dataset_question_types(
        self, mocker, pubhealthbench_verl_dataset: Dataset
    ):
        """Test _load_dataset with different question types."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.pubhealthbench.PubHealthBenchDataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = pubhealthbench_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        question_types = ["all", "mcq", "freeform"]
        for qtype in question_types:
            with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
                evaluator = PubHealthBenchEvaluator(question_type=qtype)
                evaluator._load_dataset()

            mock_dataset_class.assert_called_with(
                split="test", question_type=qtype, streaming=True
            )

    def test_load_dataset_has_required_fields(
        self, mocker, pubhealthbench_verl_dataset: Dataset
    ):
        """Test _load_dataset returns dataset with required Verl schema fields."""
        mock_dataset_class = mocker.patch(
            "med_reason_evals.verl.pubhealthbench.PubHealthBenchDataset"
        )
        mock_dataset_instance = MagicMock()
        mock_verl_dataset = pubhealthbench_verl_dataset
        mock_dataset_instance.get_verl_dataset.return_value = mock_verl_dataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = PubHealthBenchEvaluator()
            result = evaluator._load_dataset()

        # Verify Verl schema: prompt, ground_truth, metadata
        assert "prompt" in result.column_names
        assert "ground_truth" in result.column_names
        assert "metadata" in result.column_names


class TestPubHealthBenchBuildResult:
    """Tests for PubHealthBench _build_result method."""

    def test_build_result(self, assert_verl_result_shape: Callable):
        """Test _build_result method."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = PubHealthBenchEvaluator(split="test", question_type="freeform")
            avg_score = 0.5

            result = evaluator._build_result(avg_score, num_examples=2)

            assert_verl_result_shape(
                result,
                required_keys=["dataset", "split", "question_type", "num_examples"],
                expected_dataset="pubhealthbench",
            )
            assert result["question_type"] == "freeform"
            assert result["num_examples"] == 2


class TestPubHealthBenchEvaluateExample:
    """Tests for PubHealthBench _evaluate_example method."""

    async def test_evaluate_example(self, mocker):
        """Test _evaluate_example method."""
        mock_hybrid_score = mocker.patch(
            "med_reason_evals.verl.pubhealthbench.hybrid_score"
        )
        mock_hybrid_score.return_value = 1.0

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = PubHealthBenchEvaluator()
            evaluator._rollouts = MagicMock()
            evaluator._rollouts.generate = AsyncMock(return_value="<answer>A</answer>")

            prompt = [{"role": "user", "content": "What is the cause?"}]
            ground_truth = {"answer": "A"}
            metadata = {"is_mcq": True}

            score = await evaluator._evaluate_example(prompt, ground_truth, metadata)

            assert score == 1.0
            mock_hybrid_score.assert_called_once()
            call_kwargs = mock_hybrid_score.call_args.kwargs
            assert call_kwargs["solution_str"] == "<answer>A</answer>"
            assert call_kwargs["metadata"] == metadata

    async def test_evaluate_example_no_metadata(self, mocker):
        """Test _evaluate_example without metadata."""
        mock_hybrid_score = mocker.patch(
            "med_reason_evals.verl.pubhealthbench.hybrid_score"
        )
        mock_hybrid_score.return_value = 1.0

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = PubHealthBenchEvaluator()
            evaluator._rollouts = MagicMock()
            evaluator._rollouts.generate = AsyncMock(return_value="Response")

            prompt = [{"role": "user", "content": "Question"}]
            ground_truth = {"answer": "A"}

            # Call without metadata - should default to empty dict
            score = await evaluator._evaluate_example(prompt, ground_truth)

            assert score == 1.0
            call_kwargs = mock_hybrid_score.call_args.kwargs
            assert call_kwargs["metadata"] == {}
