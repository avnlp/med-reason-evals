"""Tests for MetaMedQA dataset adapter.

Tests cover the MetaMedQADataset class which handles multilingual text
normalization and maps answer text back to option letters.
"""

from unittest.mock import patch

import pytest
from datasets import Dataset

from med_reason_evals.data.metamedqa import MetaMedQADataset


class TestMetaMedQADataset:
    """Tests for MetaMedQADataset adapter."""

    @pytest.fixture
    def mock_examples(self):
        """Return sample MetaMedQA examples."""
        return [
            {
                "question": "What is the first-line treatment for hypertension?",
                "options": {
                    "A": "ACE inhibitors",
                    "B": "Beta blockers",
                    "C": "Calcium channel blockers",
                    "D": "Diuretics",
                },
                "answer": "ACE inhibitors",
                "language": "en",
            },
            {
                "question": "Which vitamin prevents scurvy?",
                "options": {
                    "A": "Vitamin A",
                    "B": "Vitamin B",
                    "C": "Vitamin C",
                    "D": "Vitamin D",
                },
                "answer": "Vitamin C",
                "language": "en",
            },
        ]

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_initialization(self, mock_load_dataset, mock_examples):
        """Test dataset initialization with default parameters."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = MetaMedQADataset()

        assert dataset.split == "test"
        assert dataset.streaming is True
        mock_load_dataset.assert_called_once_with(
            "maximegmd/MetaMedQA",
            split="test",
            streaming=True,
        )

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_initialization_forwards_kwargs(self, mock_load_dataset, mock_examples):
        """Test that extra kwargs are forwarded to load_dataset."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = MetaMedQADataset(
            split="test",
            streaming=False,
            revision="abc123",
            cache_dir="/tmp/cache",
        )

        assert dataset.split == "test"
        assert dataset.streaming is False
        mock_load_dataset.assert_called_once_with(
            "maximegmd/MetaMedQA",
            split="test",
            streaming=False,
            revision="abc123",
            cache_dir="/tmp/cache",
        )

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_num_options(self, mock_load_dataset, mock_examples):
        """Test num_options returns 6 for MetaMedQA."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MetaMedQADataset()

        assert dataset.num_options == 6

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_map_example_valid(self, mock_load_dataset, mock_examples):
        """Test mapping a valid example with answer matching an option."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MetaMedQADataset()

        example = {
            "question": "Test?",
            "options": {"A": "First", "B": "Second"},
            "answer": "Second",
        }

        result = dataset._map_example(example)

        assert result is not None
        assert result["answer"] == "B"
        assert result["info"]["answer_text"] == "Second"

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_map_example_answer_no_match(self, mock_load_dataset, mock_examples):
        """Test mapping when answer doesn't match any option."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MetaMedQADataset()

        example = {
            "question": "Test?",
            "options": {"A": "First", "B": "Second"},
            "answer": "Third",
        }

        result = dataset._map_example(example)

        assert result is None

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_map_example_unicode_normalized(self, mock_load_dataset, mock_examples):
        """Test that NFKC normalization and case folding enable matching."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MetaMedQADataset()

        # Full-width characters that NFKC normalizes to ASCII, plus different case
        example = {
            "question": "Test?",
            "options": {"A": "First", "B": "ＳＥＣＯＮＤ"},
            "answer": "second",
        }

        result = dataset._map_example(example)

        assert result is not None
        assert result["answer"] == "B"

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_map_example_collapses_internal_whitespace(
        self, mock_load_dataset, mock_examples
    ):
        """Test that internal whitespace differences do not break matching."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MetaMedQADataset()

        examples = [
            {
                "question": "Test?",
                "options": {"A": "First", "B": "ACE  inhibitors"},
                "answer": "ACE inhibitors",  # double space in option
            },
            {
                "question": "Test?",
                "options": {"A": "First", "B": "Vitamin\nC"},
                "answer": "Vitamin C",  # line break in option
            },
        ]

        results = [dataset._map_example(ex) for ex in examples]

        assert all(r is not None for r in results)
        assert [r["answer"] for r in results] == ["B", "B"]

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_map_example_empty_question(self, mock_load_dataset, mock_examples):
        """Test mapping with empty question returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MetaMedQADataset()

        example = {
            "question": "",
            "options": {"A": "First"},
            "answer": "First",
        }

        result = dataset._map_example(example)

        assert result is None

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_map_example_empty_options(self, mock_load_dataset, mock_examples):
        """Test mapping with empty options returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MetaMedQADataset()

        example = {
            "question": "Test?",
            "options": {},
            "answer": "First",
        }

        result = dataset._map_example(example)

        assert result is None

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_map_example_verl_valid(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with valid example."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MetaMedQADataset()

        example = {
            "question": "Test?",
            "options": {"A": "First", "B": "Second"},
            "answer": "First",
            "language": "en",
        }

        result = dataset._map_example_verl(example)

        assert result is not None
        assert result["data_source"] == "metamedqa"
        assert result["ground_truth"]["answer"] == "A"
        assert result["metadata"]["language"] == "en"

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_map_example_verl_no_match(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with no matching option returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MetaMedQADataset()

        example = {
            "question": "Test?",
            "options": {"A": "First", "B": "Second"},
            "answer": "Nonexistent",
        }

        result = dataset._map_example_verl(example)

        assert result is None

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_get_verifiers_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end verifiers dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MetaMedQADataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert ex["answer"] in ["A", "B", "C", "D", "E", "F"]

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_get_verl_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end Verl dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MetaMedQADataset()

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert "prompt" in ex
            assert "ground_truth" in ex

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_get_verifiers_dataset_filters_invalid(
        self, mock_load_dataset, mock_examples
    ):
        """Test that rows with no matching answer letter are dropped."""
        invalid_example = {
            "question": "Test?",
            "options": {"A": "First", "B": "Second"},
            "answer": "Nonexistent",
        }
        mock_load_dataset.return_value = Dataset.from_list(
            mock_examples + [invalid_example]
        )
        dataset = MetaMedQADataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert ex["answer"] in ["A", "B", "C", "D", "E", "F"]

    @patch("med_reason_evals.data.metamedqa.load_dataset")
    def test_get_verl_dataset_filters_invalid(self, mock_load_dataset, mock_examples):
        """Test that rows with no matching answer letter are dropped from Verl."""
        invalid_example = {
            "question": "Test?",
            "options": {"A": "First", "B": "Second"},
            "answer": "Nonexistent",
        }
        mock_load_dataset.return_value = Dataset.from_list(
            mock_examples + [invalid_example]
        )
        dataset = MetaMedQADataset()

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert "prompt" in ex
            assert "ground_truth" in ex
