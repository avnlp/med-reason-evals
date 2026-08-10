"""Tests for PubMedQA dataset adapter.

Tests cover the PubMedQADataset class which maps the canonical
openlifescienceai/pubmedqa test set (A/B/C answer scheme).
"""

from unittest.mock import patch

import pytest
from datasets import Dataset

from med_reason_evals.data.pubmedqa import PubMedQADataset


class TestPubMedQADataset:
    """Tests for PubMedQADataset adapter."""

    @pytest.fixture
    def mock_examples(self):
        """Return sample examples in openlifescienceai/pubmedqa schema."""
        return [
            {
                "data": {
                    "Question": "Does aspirin reduce cardiovascular mortality?",
                    "Correct Option": "A",
                    "Context": [
                        "Aspirin is widely used...",
                        "A meta-analysis was conducted...",
                    ],
                    "Options": {"A": "Yes", "B": "No", "C": "Maybe"},
                }
            },
            {
                "data": {
                    "Question": "Is homeopathy effective for treating asthma?",
                    "Correct Option": "B",
                    "Context": ["No significant difference was found..."],
                    "Options": {"A": "Yes", "B": "No", "C": "Maybe"},
                }
            },
            {
                "data": {
                    "Question": "Can meditation lower blood pressure?",
                    "Correct Option": "C",
                    "Context": ["Results were inconclusive..."],
                    "Options": {"A": "Yes", "B": "No", "C": "Maybe"},
                }
            },
        ]

    @patch("med_reason_evals.data.pubmedqa.load_dataset")
    def test_initialization(self, mock_load_dataset, mock_examples):
        """Test dataset initialization with default parameters."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = PubMedQADataset()

        assert dataset.split == "test"
        assert dataset.streaming is True
        mock_load_dataset.assert_called_once_with(
            "openlifescienceai/pubmedqa",
            split="test",
            streaming=True,
        )

    @patch("med_reason_evals.data.pubmedqa.load_dataset")
    def test_map_example_answer_a(self, mock_load_dataset, mock_examples):
        """Test mapping an example with Correct Option A."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = PubMedQADataset()

        result = dataset._map_example(mock_examples[0])

        assert result is not None
        assert result["answer"] == "A"
        assert result["info"]["answer_text"] == "Yes"
        assert "Select the best answer" in result["question"]

    @patch("med_reason_evals.data.pubmedqa.load_dataset")
    def test_map_example_answer_b(self, mock_load_dataset, mock_examples):
        """Test mapping an example with Correct Option B."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = PubMedQADataset()

        result = dataset._map_example(mock_examples[1])

        assert result is not None
        assert result["answer"] == "B"
        assert result["info"]["answer_text"] == "No"

    @patch("med_reason_evals.data.pubmedqa.load_dataset")
    def test_map_example_answer_c(self, mock_load_dataset, mock_examples):
        """Test mapping an example with Correct Option C."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = PubMedQADataset()

        result = dataset._map_example(mock_examples[2])

        assert result is not None
        assert result["answer"] == "C"
        assert result["info"]["answer_text"] == "Maybe"

    @patch("med_reason_evals.data.pubmedqa.load_dataset")
    def test_map_example_invalid_option(self, mock_load_dataset, mock_examples):
        """Test mapping with invalid Correct Option returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = PubMedQADataset()

        example = {
            "data": {
                "Question": "Test?",
                "Correct Option": "Z",
                "Context": [],
                "Options": {},
            }
        }

        result = dataset._map_example(example)

        assert result is None

    @patch("med_reason_evals.data.pubmedqa.load_dataset")
    def test_map_example_empty_question(self, mock_load_dataset, mock_examples):
        """Test mapping with empty question returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = PubMedQADataset()

        example = {
            "data": {
                "Question": "",
                "Correct Option": "A",
                "Context": [],
                "Options": {},
            }
        }

        result = dataset._map_example(example)

        assert result is None

    @patch("med_reason_evals.data.pubmedqa.load_dataset")
    def test_map_example_context_formatting(self, mock_load_dataset, mock_examples):
        """Test that context paragraphs are joined with single newline."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = PubMedQADataset()

        result = dataset._map_example(mock_examples[0])

        assert result is not None
        assert "Aspirin is widely used" in result["question"]
        assert "A meta-analysis was conducted" in result["question"]

    @patch("med_reason_evals.data.pubmedqa.load_dataset")
    def test_map_example_verl_valid(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with valid example."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = PubMedQADataset()

        result = dataset._map_example_verl(mock_examples[0])

        assert result is not None
        assert result["data_source"] == "pubmedqa"
        assert result["ground_truth"]["answer"] == "A"
        assert result["ground_truth"]["answer_text"] == "Yes"
        assert result["metadata"] == {}

    @patch("med_reason_evals.data.pubmedqa.load_dataset")
    def test_map_example_verl_invalid_option(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with invalid Correct Option returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = PubMedQADataset()

        example = {
            "data": {
                "Question": "Test?",
                "Correct Option": "invalid",
                "Context": [],
                "Options": {},
            }
        }

        result = dataset._map_example_verl(example)

        assert result is None

    @patch("med_reason_evals.data.pubmedqa.load_dataset")
    def test_get_verifiers_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end verifiers dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = PubMedQADataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 3
        answers = {ex["answer"] for ex in examples}
        assert answers == {"A", "B", "C"}

    @patch("med_reason_evals.data.pubmedqa.load_dataset")
    def test_get_verl_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end Verl dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = PubMedQADataset()

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 3
        for ex in examples:
            assert "prompt" in ex
            assert "ground_truth" in ex
