"""Tests for MedMCQA dataset adapter.

Tests cover the MedMCQADataset class which handles 1-indexed answer keys
and maps them to lettered multiple-choice prompts.
"""

from unittest.mock import patch

import pytest
from datasets import Dataset

from med_reason_evals.data.med_mcqa import MedMCQADataset


class TestMedMCQADataset:
    """Tests for MedMCQADataset adapter."""

    @pytest.fixture
    def mock_examples(self):
        """Return sample MedMCQA examples."""
        return [
            {
                "question": "What is the most common cause of nephrotic syndrome?",
                "opa": "Minimal change disease",
                "opb": "Membranous nephropathy",
                "opc": "Focal segmental glomerulosclerosis",
                "opd": "IgA nephropathy",
                "cop": 1,
                "subject_name": "Medicine",
                "topic_name": "Nephrology",
            },
            {
                "question": "Which enzyme is deficient in PKU?",
                "opa": "Tyrosinase",
                "opb": "Phenylalanine hydroxylase",
                "opc": "Homogentisic acid oxidase",
                "opd": "Cystathionine synthase",
                "cop": 2,
                "subject_name": "Biochemistry",
                "topic_name": "Amino acid metabolism",
            },
        ]

    @patch("med_reason_evals.data.med_mcqa.load_dataset")
    def test_initialization(self, mock_load_dataset, mock_examples):
        """Test dataset initialization with default parameters."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = MedMCQADataset()

        assert dataset.split == "validation"
        assert dataset.streaming is True
        mock_load_dataset.assert_called_once_with(
            "lighteval/med_mcqa",
            split="validation",
            streaming=True,
        )

    @patch("med_reason_evals.data.med_mcqa.load_dataset")
    def test_initialization_with_kwargs(self, mock_load_dataset, mock_examples):
        """Test dataset initialization with forwardable kwargs."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = MedMCQADataset(split="train", revision="main")

        assert dataset.split == "train"
        assert dataset.streaming is True
        mock_load_dataset.assert_called_once_with(
            "lighteval/med_mcqa",
            split="train",
            streaming=True,
            revision="main",
        )

    @patch("med_reason_evals.data.med_mcqa.load_dataset")
    def test_num_options(self, mock_load_dataset, mock_examples):
        """Test num_options returns 4 for MedMCQA."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedMCQADataset()

        assert dataset.num_options == 4

    @patch("med_reason_evals.data.med_mcqa.load_dataset")
    def test_map_example_valid(self, mock_load_dataset, mock_examples):
        """Test mapping a valid example with cop=2 mapping to B."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedMCQADataset()

        example = {
            "question": "Test question?",
            "opa": "Option A",
            "opb": "Option B",
            "opc": "Option C",
            "opd": "Option D",
            "cop": 2,
        }

        result = dataset._map_example(example)

        assert result["answer"] == "B"
        assert result["info"]["answer_text"] == "Option B"
        assert "Test question?" in result["question"]

    @patch("med_reason_evals.data.med_mcqa.load_dataset")
    def test_map_example_verl_valid(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with valid example."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedMCQADataset()

        example = {
            "question": "Test?",
            "opa": "Option A",
            "opb": "Option B",
            "opc": "Option C",
            "opd": "Option D",
            "cop": 3,
            "subject_name": "Surgery",
            "topic_name": "GI",
        }

        result = dataset._map_example_verl(example)

        assert result["data_source"] == "med_mcqa"
        assert result["ground_truth"]["answer"] == "C"
        assert result["ground_truth"]["answer_text"] == "Option C"
        assert result["metadata"]["subject"] == "Surgery"
        assert result["metadata"]["topic"] == "GI"

    @patch("med_reason_evals.data.med_mcqa.load_dataset")
    def test_get_verifiers_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end verifiers dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedMCQADataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert ex["answer"] in ["A", "B", "C", "D"]

    @patch("med_reason_evals.data.med_mcqa.load_dataset")
    def test_get_verl_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end Verl dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedMCQADataset()

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert "prompt" in ex
            assert "ground_truth" in ex

    @patch("med_reason_evals.data.med_mcqa.load_dataset")
    def test_build_prompt(self, mock_load_dataset, mock_examples):
        """Test prompt formatting includes letter answer instruction."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedMCQADataset()

        prompt = dataset._build_prompt("Test?", ["X", "Y", "Z", "W"])

        assert "Give a letter answer among A, B, C or D." in prompt
        assert "A. X" in prompt
        assert "D. W" in prompt
        assert "Answer:" in prompt


class TestMedMCQAIsValidExample:
    """Tests for _is_valid_example validation logic."""

    @staticmethod
    def _valid_example() -> dict:
        """Return a valid MedMCQA example."""
        return {
            "question": "What is the most likely diagnosis?",
            "opa": "Option A",
            "opb": "Option B",
            "opc": "Option C",
            "opd": "Option D",
            "cop": 1,
        }

    def test_returns_true_for_valid_example(self):
        """Test valid example passes all checks."""
        assert MedMCQADataset._is_valid_example(self._valid_example()) is True

    def test_returns_false_for_cop_string(self):
        """Test non-integer cop is rejected."""
        example = self._valid_example()
        example["cop"] = "1"
        assert MedMCQADataset._is_valid_example(example) is False

    def test_returns_false_for_cop_out_of_range(self):
        """Test cop outside [1-4] is rejected."""
        example = self._valid_example()
        example["cop"] = 5
        assert MedMCQADataset._is_valid_example(example) is False

    def test_returns_false_for_cop_zero(self):
        """Test cop of 0 is rejected."""
        example = self._valid_example()
        example["cop"] = 0
        assert MedMCQADataset._is_valid_example(example) is False

    def test_returns_false_when_question_missing(self):
        """Test returns False when question field is missing."""
        example = self._valid_example()
        del example["question"]
        assert MedMCQADataset._is_valid_example(example) is False

    def test_returns_false_when_question_empty(self):
        """Test returns False when question is an empty string."""
        example = self._valid_example()
        example["question"] = ""
        assert MedMCQADataset._is_valid_example(example) is False

    def test_returns_false_when_question_not_string(self):
        """Test returns False when question is not a string."""
        example = self._valid_example()
        example["question"] = None
        assert MedMCQADataset._is_valid_example(example) is False

    def test_returns_false_when_all_options_empty(self):
        """Test returns False when all options are empty."""
        example = self._valid_example()
        example["opa"] = ""
        example["opb"] = ""
        example["opc"] = ""
        example["opd"] = ""
        assert MedMCQADataset._is_valid_example(example) is False

    def test_returns_false_when_some_options_missing(self):
        """Test returns False when not all four options are present."""
        example = self._valid_example()
        example["opb"] = ""
        example["opc"] = ""
        example["opd"] = ""
        assert MedMCQADataset._is_valid_example(example) is False

    def test_returns_false_when_gold_option_blank(self):
        """Test returns False when the gold option (pointed to by cop) is blank."""
        example = self._valid_example()
        example["cop"] = 2
        example["opb"] = ""
        assert MedMCQADataset._is_valid_example(example) is False

    def test_extract_options_valid(self):
        """Test _extract_options returns cleaned fields for a valid example."""
        example = self._valid_example()
        example["cop"] = 2

        result = MedMCQADataset._extract_options(example)

        assert result == (
            "What is the most likely diagnosis?",
            ["Option A", "Option B", "Option C", "Option D"],
            1,  # zero-based index of cop=2
        )

    def test_extract_options_strips_whitespace(self):
        """Test _extract_options strips surrounding whitespace from fields."""
        example = self._valid_example()
        example["question"] = "  What is the diagnosis?  "
        example["opa"] = "  Option A  "

        result = MedMCQADataset._extract_options(example)

        assert result is not None
        assert result[0] == "What is the diagnosis?"
        assert result[1][0] == "Option A"

    def test_extract_options_returns_none_for_malformed(self):
        """Test _extract_options returns None for a malformed example."""
        example = self._valid_example()
        example["cop"] = "1"
        assert MedMCQADataset._extract_options(example) is None

    @patch("med_reason_evals.data.med_mcqa.load_dataset")
    def test_get_verifiers_dataset_filters_invalid(self, mock_load_dataset):
        """Test that get_verifiers_dataset filters out invalid examples."""
        valid = [
            {
                "question": "What is the most likely diagnosis?",
                "opa": "Option A",
                "opb": "Option B",
                "opc": "Option C",
                "opd": "Option D",
                "cop": 1,
            },
            {
                "question": "Which drug is first-line for type 2 diabetes?",
                "opa": "Metformin",
                "opb": "Insulin",
                "opc": "Sulfonylurea",
                "opd": "DPP-4 inhibitor",
                "cop": 1,
            },
        ]
        invalid = {
            "question": "",
            "opa": "Option A",
            "opb": "Option B",
            "opc": "Option C",
            "opd": "Option D",
            "cop": 1,
        }
        mock_load_dataset.return_value = Dataset.from_list(valid + [invalid])
        dataset = MedMCQADataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 2

    @patch("med_reason_evals.data.med_mcqa.load_dataset")
    def test_get_verifiers_dataset_filters_invalid_last_row(self, mock_load_dataset):
        """Test filtering when the invalid row is last (regression for map+None)."""
        valid = {
            "question": "What is the most likely diagnosis?",
            "opa": "Option A",
            "opb": "Option B",
            "opc": "Option C",
            "opd": "Option D",
            "cop": 1,
        }
        invalid = {
            "question": "",
            "opa": "Option A",
            "opb": "Option B",
            "opc": "Option C",
            "opd": "Option D",
            "cop": 1,
        }
        mock_load_dataset.return_value = Dataset.from_list([valid, invalid])
        dataset = MedMCQADataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 1
        assert examples[0]["answer"] == "A"

    @patch("med_reason_evals.data.med_mcqa.load_dataset")
    def test_get_verl_dataset_filters_invalid(self, mock_load_dataset):
        """Test that get_verl_dataset filters out invalid examples."""
        valid = [
            {
                "question": "What is the most likely diagnosis?",
                "opa": "Option A",
                "opb": "Option B",
                "opc": "Option C",
                "opd": "Option D",
                "cop": 1,
            },
            {
                "question": "Which drug is first-line for type 2 diabetes?",
                "opa": "Metformin",
                "opb": "Insulin",
                "opc": "Sulfonylurea",
                "opd": "DPP-4 inhibitor",
                "cop": 1,
            },
        ]
        invalid = {
            "question": None,
            "opa": "Option A",
            "opb": "Option B",
            "opc": "Option C",
            "opd": "Option D",
            "cop": 1,
        }
        mock_load_dataset.return_value = Dataset.from_list(valid + [invalid])
        dataset = MedMCQADataset()

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 2

    @patch("med_reason_evals.data.med_mcqa.load_dataset")
    def test_get_verifiers_dataset_filters_blank_option_row(self, mock_load_dataset):
        """Test that a row with a blank option (incl. the gold one) is dropped."""
        valid = {
            "question": "What is the most likely diagnosis?",
            "opa": "Option A",
            "opb": "Option B",
            "opc": "Option C",
            "opd": "Option D",
            "cop": 1,
        }
        blank_option = {
            "question": "Which drug is first-line for type 2 diabetes?",
            "opa": "Metformin",
            "opb": "Insulin",
            "opc": "",
            "opd": "DPP-4 inhibitor",
            "cop": 1,
        }
        blank_gold = {
            "question": "What is the treatment for gout?",
            "opa": "Allopurinol",
            "opb": "",
            "opc": "Colchicine",
            "opd": "NSAIDs",
            "cop": 2,  # gold option is opb, which is blank
        }
        mock_load_dataset.return_value = Dataset.from_list(
            [valid, blank_option, blank_gold]
        )
        dataset = MedMCQADataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 1
        assert examples[0]["answer"] == "A"
        assert examples[0]["info"]["answer_text"] == "Option A"

    @patch("med_reason_evals.data.med_mcqa.load_dataset")
    def test_get_verl_dataset_filters_blank_gold_option(self, mock_load_dataset):
        """Test that a blank gold option is dropped from the verl dataset."""
        valid = {
            "question": "What is the most likely diagnosis?",
            "opa": "Option A",
            "opb": "Option B",
            "opc": "Option C",
            "opd": "Option D",
            "cop": 1,
        }
        blank_gold = {
            "question": "What is the treatment for gout?",
            "opa": "Allopurinol",
            "opb": "",
            "opc": "Colchicine",
            "opd": "NSAIDs",
            "cop": 2,
        }
        mock_load_dataset.return_value = Dataset.from_list([valid, blank_gold])
        dataset = MedMCQADataset()

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 1
        assert examples[0]["ground_truth"]["answer"] == "A"

    def test_map_example_raises_on_malformed(self):
        """Test that _map_example raises when given a malformed example."""
        dataset = MedMCQADataset.__new__(MedMCQADataset)
        malformed = {
            "question": "Test?",
            "opa": "A",
            "opb": "",
            "opc": "C",
            "opd": "D",
            "cop": 1,
        }

        with pytest.raises(ValueError, match="Malformed MedMCQA example"):
            dataset._map_example(malformed)
