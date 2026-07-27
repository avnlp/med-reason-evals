"""Tests for MedQA dataset adapter.

Tests cover the MedQADataset class which handles USMLE-style
multiple-choice questions.
"""

from unittest.mock import patch

import pytest
from datasets import Dataset

from med_reason_evals.data.medqa import MedQADataset


class TestMedQADataset:
    """Tests for MedQADataset adapter."""

    @pytest.fixture
    def mock_examples(self):
        """Return sample MedQA examples."""
        return [
            {
                "question": "A patient presents with jaundice and RUQ pain. What is the most likely diagnosis?",
                "options": {
                    "A": "Cholecystitis",
                    "B": "Hepatitis",
                    "C": "Pancreatitis",
                    "D": "Appendicitis",
                },
                "answer_idx": "A",
            },
            {
                "question": "Which drug is first-line for type 2 diabetes?",
                "options": {
                    "A": "Insulin",
                    "B": "Metformin",
                    "C": "Glipizide",
                    "D": "Pioglitazone",
                },
                "answer_idx": "B",
            },
        ]

    @patch("med_reason_evals.data.medqa.load_dataset")
    def test_initialization(self, mock_load_dataset, mock_examples):
        """Test dataset initialization with default parameters."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = MedQADataset()

        assert dataset.split == "test"
        assert dataset.streaming is True
        mock_load_dataset.assert_called_once_with(
            "GBaker/MedQA-USMLE-4-options",
            split="test",
            streaming=True,
        )

    @patch("med_reason_evals.data.medqa.load_dataset")
    def test_initialization_with_kwargs(self, mock_load_dataset, mock_examples):
        """Test dataset initialization with forwardable kwargs."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = MedQADataset(split="train", revision="main")

        assert dataset.split == "train"
        assert dataset.streaming is True
        mock_load_dataset.assert_called_once_with(
            "GBaker/MedQA-USMLE-4-options",
            split="train",
            streaming=True,
            revision="main",
        )

    @patch("med_reason_evals.data.medqa.load_dataset")
    def test_num_options(self, mock_load_dataset, mock_examples):
        """Test num_options returns 4 for MedQA."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedQADataset()

        assert dataset.num_options == 4

    @patch("med_reason_evals.data.medqa.load_dataset")
    def test_map_example_valid(self, mock_load_dataset, mock_examples):
        """Test mapping a valid example."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedQADataset()

        result = dataset._map_example(mock_examples[0])

        assert result["answer"] == "A"
        assert result["info"]["answer_text"] == "Cholecystitis"
        assert "Question:" in result["question"]

    @patch("med_reason_evals.data.medqa.load_dataset")
    def test_map_example_verl_valid(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with valid example."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedQADataset()

        result = dataset._map_example_verl(mock_examples[1])

        assert result["data_source"] == "medqa"
        assert result["ground_truth"]["answer"] == "B"
        assert result["ground_truth"]["answer_text"] == "Metformin"
        assert result["metadata"]["original_question"] == mock_examples[1]["question"]

    @patch("med_reason_evals.data.medqa.load_dataset")
    def test_get_verifiers_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end verifiers dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedQADataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert "question" in ex
            assert "answer" in ex
            assert "info" in ex

    @patch("med_reason_evals.data.medqa.load_dataset")
    def test_get_verl_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end Verl dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedQADataset()

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert "prompt" in ex
            assert "ground_truth" in ex
            assert "data_source" in ex

    @patch("med_reason_evals.data.medqa.load_dataset")
    def test_build_prompt_format(self, mock_load_dataset, mock_examples):
        """Test prompt formatting includes question and options."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedQADataset()

        options = {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"}
        prompt = dataset._build_prompt("Test question?", options)

        assert "Question: Test question?" in prompt
        assert "A. First" in prompt
        assert "D. Fourth" in prompt
        assert "Answer:" in prompt


class TestMedQAIsValidExample:
    """Tests for _is_valid_example validation logic."""

    def test_returns_true_for_valid_example(self):
        """Test valid example passes all checks."""
        example = {
            "question": "What is the most likely diagnosis?",
            "options": {
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D",
            },
            "answer_idx": "C",
        }
        assert MedQADataset._is_valid_example(example) is True

    def test_returns_true_for_lowercase_answer_idx(self):
        """Test lowercase answer_idx is normalized to uppercase."""
        example = {
            "question": "What is the most likely diagnosis?",
            "options": {
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D",
            },
            "answer_idx": "c",
        }
        assert MedQADataset._is_valid_example(example) is True

    def test_returns_false_when_question_missing(self):
        """Test returns False when question field is missing."""
        example = {
            "options": {"A": "Option A", "B": "Option B"},
            "answer_idx": "A",
        }
        assert MedQADataset._is_valid_example(example) is False

    def test_returns_false_when_question_empty(self):
        """Test returns False when question is an empty string."""
        example = {
            "question": "",
            "options": {"A": "Option A", "B": "Option B"},
            "answer_idx": "A",
        }
        assert MedQADataset._is_valid_example(example) is False

    def test_returns_false_when_question_not_string(self):
        """Test returns False when question is not a string."""
        example = {
            "question": None,
            "options": {"A": "Option A", "B": "Option B"},
            "answer_idx": "A",
        }
        assert MedQADataset._is_valid_example(example) is False

    def test_returns_false_when_options_missing(self):
        """Test returns False when options field is missing."""
        example = {
            "question": "What is the most likely diagnosis?",
            "answer_idx": "A",
        }
        assert MedQADataset._is_valid_example(example) is False

    def test_returns_false_when_options_empty_dict(self):
        """Test returns False when options is an empty dict."""
        example = {
            "question": "What is the most likely diagnosis?",
            "options": {},
            "answer_idx": "A",
        }
        assert MedQADataset._is_valid_example(example) is False

    def test_returns_false_when_options_not_dict(self):
        """Test returns False when options is not a dict."""
        example = {
            "question": "What is the most likely diagnosis?",
            "options": "not a dict",
            "answer_idx": "A",
        }
        assert MedQADataset._is_valid_example(example) is False

    def test_returns_false_when_answer_idx_missing(self):
        """Test returns False when answer_idx field is missing."""
        example = {
            "question": "What is the most likely diagnosis?",
            "options": {"A": "Option A", "B": "Option B"},
        }
        assert MedQADataset._is_valid_example(example) is False

    def test_returns_false_when_answer_idx_empty(self):
        """Test returns False when answer_idx is an empty string."""
        example = {
            "question": "What is the most likely diagnosis?",
            "options": {"A": "Option A", "B": "Option B"},
            "answer_idx": "",
        }
        assert MedQADataset._is_valid_example(example) is False

    def test_returns_false_when_answer_idx_not_in_options(self):
        """Test returns False when answer_idx does not match an option key."""
        example = {
            "question": "What is the most likely diagnosis?",
            "options": {"A": "Option A", "B": "Option B", "C": "Option C"},
            "answer_idx": "Z",
        }
        assert MedQADataset._is_valid_example(example) is False

    @patch("med_reason_evals.data.medqa.load_dataset")
    def test_get_verifiers_dataset_filters_invalid(self, mock_load_dataset):
        """Test that get_verifiers_dataset filters out invalid examples."""
        valid = [
            {
                "question": "What is the most likely diagnosis?",
                "options": {"A": "Option A", "B": "Option B"},
                "answer_idx": "A",
            },
            {
                "question": "Which drug is first-line for type 2 diabetes?",
                "options": {"A": "Metformin", "B": "Insulin"},
                "answer_idx": "A",
            },
        ]
        invalid = {
            "question": "",
            "options": {"A": "Option A"},
            "answer_idx": "A",
        }
        mock_load_dataset.return_value = Dataset.from_list(valid + [invalid])
        dataset = MedQADataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 2

    @patch("med_reason_evals.data.medqa.load_dataset")
    def test_get_verl_dataset_filters_invalid(self, mock_load_dataset):
        """Test that get_verl_dataset filters out invalid examples."""
        valid = [
            {
                "question": "What is the most likely diagnosis?",
                "options": {"A": "Option A", "B": "Option B"},
                "answer_idx": "A",
            },
            {
                "question": "Which drug is first-line for type 2 diabetes?",
                "options": {"A": "Metformin", "B": "Insulin"},
                "answer_idx": "A",
            },
        ]
        invalid = {
            "question": None,
            "options": {"A": "Option A"},
            "answer_idx": "A",
        }
        mock_load_dataset.return_value = Dataset.from_list(valid + [invalid])
        dataset = MedQADataset()

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 2
