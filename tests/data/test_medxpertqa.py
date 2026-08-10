"""Tests for MedXpertQA dataset adapter.

Tests cover the MedXpertQADataset class which handles expert-level medical
questions with optional question type filtering.
"""

from unittest.mock import patch

import pytest
from datasets import Dataset

from med_reason_evals.data.medxpertqa import MedXpertQADataset


class TestMedXpertQADataset:
    """Tests for MedXpertQADataset adapter."""

    @pytest.fixture
    def mock_examples(self):
        """Return sample MedXpertQA examples."""
        return [
            {
                "question": "What is the mechanism of action of metformin?",
                "options": {
                    "A": "Insulin secretion",
                    "B": "Gluconeogenesis inhibition",
                },
                "label": "B",
                "question_type": "reasoning",
                "medical_task": "pharmacology",
                "body_system": "endocrine",
            },
            {
                "question": "Which imaging modality is best for soft tissue?",
                "options": {"A": "X-ray", "B": "CT", "C": "MRI"},
                "label": "C",
                "question_type": "understanding",
                "medical_task": "radiology",
                "body_system": "musculoskeletal",
            },
        ]

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_initialization_default(self, mock_load_dataset, mock_examples):
        """Test dataset initialization with default parameters."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = MedXpertQADataset()

        assert dataset.split == "test"
        assert dataset.streaming is True
        assert dataset.question_type == "all"
        mock_load_dataset.assert_called_once_with(
            "TsinghuaC3I/MedXpertQA",
            name="Text",
            split="test",
            streaming=True,
        )

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_initialization_with_question_type_filter(
        self, mock_load_dataset, mock_examples
    ):
        """Test dataset initialization with question_type filter."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = MedXpertQADataset(question_type="reasoning")

        assert dataset.question_type == "reasoning"
        result = list(dataset._dataset)
        assert len(result) == 1
        assert result[0]["question_type"] == "reasoning"

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_format_question_with_embedded_choices(
        self, mock_load_dataset, mock_examples
    ):
        """Test question formatting when choices are embedded in the question."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        question = "What is X? Answer Choices: (A) foo (B) bar"
        options = {"A": "foo", "B": "bar"}

        result = dataset._format_question_with_options(question, options)

        assert "What is X?" in result
        assert "Answer Choices:" in result
        # The embedded choices in the question stem should be stripped
        assert result.count("Answer Choices:") == 1

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_format_question_without_embedded_choices(
        self, mock_load_dataset, mock_examples
    ):
        """Test normal question formatting without embedded choices."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        question = "What is the mechanism?"
        options = {"A": "Option A", "B": "Option B"}

        result = dataset._format_question_with_options(question, options)

        assert "What is the mechanism?" in result
        assert "(A) Option A" in result
        assert "(B) Option B" in result

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_format_question_empty_options(self, mock_load_dataset, mock_examples):
        """Test formatting with empty options returns the question as-is."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        result = dataset._format_question_with_options("Question?", {})

        assert result == "Question?"

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_map_example_valid(self, mock_load_dataset, mock_examples):
        """Test mapping a valid example."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        example = {
            "question": "Test?",
            "options": {"A": "First", "B": "Second"},
            "label": "A",
            "question_type": "reasoning",
        }

        result = dataset._map_example(example)

        assert result["answer"] == "A"
        assert result["info"]["answer_text"] == "First"
        assert "Test?" in result["question"]

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_map_example_non_dict_options(self, mock_load_dataset, mock_examples):
        """Test mapping with non-dict options returns invalid."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        example = {
            "question": "Test?",
            "options": None,
            "label": "A",
        }

        result = dataset._map_example(example)

        assert result["answer"] is None

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_map_example_missing_label(self, mock_load_dataset, mock_examples):
        """Test mapping with empty label returns invalid."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        example = {
            "question": "Test?",
            "options": {"A": "First"},
            "label": "",
        }

        result = dataset._map_example(example)

        assert result["answer"] is None

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_map_example_label_not_in_options(self, mock_load_dataset, mock_examples):
        """Test mapping when label is not in options dict."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        example = {
            "question": "Test?",
            "options": {"A": "First", "B": "Second"},
            "label": "Z",
        }

        result = dataset._map_example(example)

        assert result["answer"] is None

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_map_example_verl_valid(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with valid example."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        example = {
            "question": "Test?",
            "options": {"A": "First", "B": "Second"},
            "label": "B",
            "question_type": "reasoning",
            "medical_task": "diagnosis",
            "body_system": "cardiac",
        }

        result = dataset._map_example_verl(example)

        assert result["data_source"] == "medxpertqa"
        assert result["ground_truth"]["answer"] == "B"
        assert result["ground_truth"]["answer_text"] == "Second"
        assert result["metadata"]["question_type"] == "reasoning"

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_map_example_verl_non_dict_options(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with non-dict options returns null ground_truth."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        example = {
            "question": "Test?",
            "options": None,
            "label": "A",
        }

        result = dataset._map_example_verl(example)

        assert result["ground_truth"] is None

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_map_example_verl_label_not_in_options(
        self, mock_load_dataset, mock_examples
    ):
        """Test Verl mapping with mislabeled row."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        example = {
            "question": "Test?",
            "options": {"A": "First"},
            "label": "Z",
        }

        result = dataset._map_example_verl(example)

        assert result["ground_truth"] is None

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_get_verifiers_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end verifiers dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert "question" in ex
            assert "answer" in ex
            assert "info" in ex

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_get_verl_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end Verl dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert "prompt" in ex
            assert "ground_truth" in ex
            assert "data_source" in ex

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_get_verifiers_dataset_filters_invalid(self, mock_load_dataset):
        """Test that verifiers dataset filters out invalid examples."""
        examples = [
            {
                "question": "Valid?",
                "options": {"A": "Yes", "B": "No"},
                "label": "A",
                "question_type": "reasoning",
            },
            {
                "question": "Invalid label",
                "options": {"A": "Yes"},
                "label": "Z",
                "question_type": "reasoning",
            },
        ]
        mock_load_dataset.return_value = Dataset.from_list(examples)
        dataset = MedXpertQADataset()

        result = dataset.get_verifiers_dataset()
        filtered = list(result)

        assert len(filtered) == 1
        assert filtered[0]["answer"] == "A"
