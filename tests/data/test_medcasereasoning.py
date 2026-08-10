"""Tests for MedCaseReasoning dataset adapter.

Tests cover the MedCaseReasoningDataset class which wraps narrative case
presentations for diagnosis prediction.
"""

from unittest.mock import patch

import pytest
from datasets import Dataset

from med_reason_evals.data.medcasereasoning import MedCaseReasoningDataset


class TestMedCaseReasoningDataset:
    """Tests for MedCaseReasoningDataset adapter."""

    @pytest.fixture
    def mock_examples(self):
        """Return sample MedCaseReasoning examples."""
        return [
            {
                "case_prompt": "A 55-year-old male presents with chest pain radiating to the left arm, diaphoresis, and shortness of breath.",
                "final_diagnosis": "Acute Myocardial Infarction",
                "differential_diagnosis": ["Unstable Angina", "PE"],
            },
            {
                "case_prompt": "A 30-year-old female presents with butterfly rash, joint pain, and fatigue.",
                "final_diagnosis": "Systemic Lupus Erythematosus",
                "differential_diagnosis": ["Rheumatoid Arthritis"],
            },
        ]

    @patch("med_reason_evals.data.medcasereasoning.load_dataset")
    def test_initialization(self, mock_load_dataset, mock_examples):
        """Test dataset initialization with default parameters."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = MedCaseReasoningDataset()

        assert dataset.split == "val"
        assert dataset.streaming is True
        mock_load_dataset.assert_called_once_with(
            "zou-lab/MedCaseReasoning",
            split="val",
            streaming=True,
        )

    @patch("med_reason_evals.data.medcasereasoning.load_dataset")
    def test_map_example_valid(self, mock_load_dataset, mock_examples):
        """Test mapping a valid case with diagnosis."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedCaseReasoningDataset()

        result = dataset._map_example(mock_examples[0])

        assert result is not None
        assert result["answer"] == "Acute Myocardial Infarction"
        assert "CASE PRESENTATION" in result["question"]
        assert "chest pain" in result["question"]
        assert result["info"]["case_prompt"] == mock_examples[0]["case_prompt"]

    @patch("med_reason_evals.data.medcasereasoning.load_dataset")
    def test_map_example_empty_case(self, mock_load_dataset, mock_examples):
        """Test mapping with empty case_prompt returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedCaseReasoningDataset()

        example = {"case_prompt": "", "final_diagnosis": "Something"}

        result = dataset._map_example(example)

        assert result is None

    @patch("med_reason_evals.data.medcasereasoning.load_dataset")
    def test_map_example_empty_diagnosis(self, mock_load_dataset, mock_examples):
        """Test mapping with empty final_diagnosis returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedCaseReasoningDataset()

        example = {
            "case_prompt": "Patient presents with symptoms.",
            "final_diagnosis": "",
        }

        result = dataset._map_example(example)

        assert result is None

    @patch("med_reason_evals.data.medcasereasoning.load_dataset")
    def test_map_example_verl_valid(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with valid example."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedCaseReasoningDataset()

        result = dataset._map_example_verl(mock_examples[0])

        assert result is not None
        assert result["data_source"] == "medcasereasoning"
        assert result["ground_truth"]["answer"] == "Acute Myocardial Infarction"
        assert result["ground_truth"]["target"] == "Acute Myocardial Infarction"
        assert result["metadata"]["differential_diagnosis"] == [
            "Unstable Angina",
            "PE",
        ]

    @patch("med_reason_evals.data.medcasereasoning.load_dataset")
    def test_map_example_verl_empty_fields(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with empty fields returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedCaseReasoningDataset()

        example = {"case_prompt": "  ", "final_diagnosis": "  "}

        result = dataset._map_example_verl(example)

        assert result is None

    @patch("med_reason_evals.data.medcasereasoning.load_dataset")
    def test_get_verifiers_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end verifiers dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedCaseReasoningDataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert "question" in ex
            assert "answer" in ex
            assert ex["answer"] != ""

    @patch("med_reason_evals.data.medcasereasoning.load_dataset")
    def test_get_verl_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end Verl dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedCaseReasoningDataset()

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert "prompt" in ex
            assert "ground_truth" in ex

    @patch("med_reason_evals.data.medcasereasoning.load_dataset")
    def test_build_prompt_includes_template(self, mock_load_dataset, mock_examples):
        """Test that build_prompt includes the expected template elements."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedCaseReasoningDataset()

        result = dataset._build_prompt("Patient has fever.")

        assert "CASE PRESENTATION" in result
        assert "Patient has fever." in result
        assert "<think>" in result
        assert "<answer>" in result
