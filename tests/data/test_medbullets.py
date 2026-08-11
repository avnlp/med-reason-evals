"""Tests for MedBullets dataset adapter.

Tests cover the MedBulletsDataset class which handles 4- and 5-option
exam-prep multiple-choice questions.
"""

from unittest.mock import patch

import pytest
from datasets import Dataset

from med_reason_evals.data.medbullets import MedBulletsDataset


class TestMedBulletsDataset:
    """Tests for MedBulletsDataset adapter."""

    @pytest.fixture
    def mock_examples(self):
        """Return sample MedBullets examples."""
        return [
            {
                "question": "A 45-year-old presents with chest pain. What is the diagnosis?",
                "options": {
                    "A": "MI",
                    "B": "PE",
                    "C": "Pneumothorax",
                    "D": "Pericarditis",
                    "E": "Aortic dissection",
                },
                "answer": "A",
            },
            {
                "question": "Which drug is a beta blocker?",
                "options": {
                    "A": "Atenolol",
                    "B": "Amlodipine",
                    "C": "Lisinopril",
                    "D": "Losartan",
                    "E": "Hydrochlorothiazide",
                },
                "answer": "A",
            },
        ]

    @patch("med_reason_evals.data.medbullets.load_dataset")
    def test_initialization_default(self, mock_load_dataset, mock_examples):
        """Test dataset initialization with default 4 options."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = MedBulletsDataset()

        assert dataset.split == "op4_test"
        assert dataset.num_options == 4
        mock_load_dataset.assert_called_once_with(
            "mkieffer/MedBullets",
            split="op4_test",
            streaming=True,
        )

    @patch("med_reason_evals.data.medbullets.load_dataset")
    def test_initialization_5_options(self, mock_load_dataset, mock_examples):
        """Test initialization with 5 options selects correct split."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = MedBulletsDataset(num_options=5)

        assert dataset.split == "op5_test"
        assert dataset.num_options == 5

    def test_initialization_invalid_options_raises(self):
        """Test initialization with invalid num_options raises ValueError."""
        with pytest.raises(ValueError, match="num_options must be 4 or 5"):
            MedBulletsDataset(num_options=3)

    @patch("med_reason_evals.data.medbullets.load_dataset")
    def test_map_example_valid_4_options(self, mock_load_dataset, mock_examples):
        """Test mapping a valid 4-option example."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedBulletsDataset(num_options=4)

        example = {
            "question": "Test?",
            "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four", "E": "Five"},
            "answer": "B",
        }

        result = dataset._map_example(example)

        assert result["answer"] == "B"
        assert result["info"]["answer_text"] == "Two"
        assert result["info"]["original_question"] == "Test?"
        # E option should be filtered out for 4-option variant
        assert "E." not in result["question"]

    @patch("med_reason_evals.data.medbullets.load_dataset")
    def test_valid_example_accepts_well_formed_row(
        self, mock_load_dataset, mock_examples
    ):
        """Test a well-formed row passes validation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedBulletsDataset()

        assert dataset._is_valid_example(mock_examples[0]) is True

    @patch("med_reason_evals.data.medbullets.load_dataset")
    def test_valid_example_rejects_invalid_answer(
        self, mock_load_dataset, mock_examples
    ):
        """Test a row whose answer is not one of the options is rejected."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedBulletsDataset()

        example = {
            "question": "Test?",
            "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
            "answer": "Z",
        }

        assert dataset._is_valid_example(example) is False

    @patch("med_reason_evals.data.medbullets.load_dataset")
    def test_valid_example_rejects_answer_e_on_4_option_split(
        self, mock_load_dataset, mock_examples
    ):
        """Test answer "E" is rejected when only 4 options are offered.

        Option E is stripped for the 4-option variant, so a gold answer of E
        would have no corresponding choice in the prompt.
        """
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedBulletsDataset(num_options=4)

        example = {
            "question": "Test?",
            "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four", "E": "Five"},
            "answer": "E",
        }

        assert dataset._is_valid_example(example) is False

    @patch("med_reason_evals.data.medbullets.load_dataset")
    def test_valid_example_rejects_empty_question(
        self, mock_load_dataset, mock_examples
    ):
        """Test a row with an empty question is rejected."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedBulletsDataset()

        example = {
            "question": "   ",
            "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
            "answer": "A",
        }

        assert dataset._is_valid_example(example) is False

    @patch("med_reason_evals.data.medbullets.load_dataset")
    def test_valid_example_rejects_wrong_option_count(
        self, mock_load_dataset, mock_examples
    ):
        """Test a row with fewer options than requested is rejected."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedBulletsDataset(num_options=4)

        example = {
            "question": "Test?",
            "options": {"A": "One", "B": "Two"},
            "answer": "A",
        }

        assert dataset._is_valid_example(example) is False

    @patch("med_reason_evals.data.medbullets.load_dataset")
    def test_usable_options_drops_empty_choices(self, mock_load_dataset, mock_examples):
        """Test that empty option values are dropped from the prompt.

        The upstream struct always carries an "E" key; on the 4-option split
        it is null.
        """
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedBulletsDataset(num_options=5)

        example = {
            "question": "Test?",
            "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four", "E": None},
            "answer": "A",
        }

        assert dataset._usable_options(example) == {
            "A": "One",
            "B": "Two",
            "C": "Three",
            "D": "Four",
        }
        # 5 options were requested but only 4 are usable
        assert dataset._is_valid_example(example) is False

    @patch("med_reason_evals.data.medbullets.load_dataset")
    def test_map_example_verl_valid(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with valid example."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedBulletsDataset()

        example = {
            "question": "Test?",
            "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
            "answer": "C",
        }

        result = dataset._map_example_verl(example)

        assert result["data_source"] == "medbullets"
        assert result["ground_truth"]["answer"] == "C"
        assert result["ground_truth"]["answer_text"] == "Three"
        assert result["metadata"]["num_options"] == 4
        assert result["prompt"][0]["role"] == "user"

    @patch("med_reason_evals.data.medbullets.load_dataset")
    def test_get_verifiers_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end verifiers dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedBulletsDataset(num_options=5)

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert "question" in ex
            assert "answer" in ex
            assert "info" in ex

    @patch("med_reason_evals.data.medbullets.load_dataset")
    def test_get_verifiers_dataset_filters_invalid_rows(
        self, mock_load_dataset, mock_examples
    ):
        """Test that malformed rows are dropped instead of passed through."""
        rows = [
            *mock_examples,
            {
                "question": "",
                "options": {
                    "A": "One",
                    "B": "Two",
                    "C": "Three",
                    "D": "Four",
                    "E": "Five",
                },
                "answer": "A",
            },
            {
                "question": "Unanswerable?",
                "options": {
                    "A": "One",
                    "B": "Two",
                    "C": "Three",
                    "D": "Four",
                    "E": "Five",
                },
                "answer": "Z",
            },
        ]
        mock_load_dataset.return_value = Dataset.from_list(rows)
        dataset = MedBulletsDataset(num_options=5)

        examples = list(dataset.get_verifiers_dataset())

        assert len(examples) == 2

    @patch("med_reason_evals.data.medbullets.load_dataset")
    def test_get_verl_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end Verl dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedBulletsDataset(num_options=5)

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert "prompt" in ex
            assert "ground_truth" in ex
            assert ex["data_source"] == "medbullets"
