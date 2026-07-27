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
