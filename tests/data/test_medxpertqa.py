"""Tests for MedXpertQA dataset adapter.

Tests cover the MedXpertQADataset class which handles expert-level medical
questions with optional question type filtering, the fixed ten-choice A-J
schema, and the lazy streaming path.
"""

from unittest.mock import patch

import pytest
from datasets import Dataset, IterableDataset

from med_reason_evals.data.medxpertqa import MedXpertQADataset


def _ten_options() -> dict[str, str]:
    """Return a full A-J options block matching the Text config schema."""
    return {
        "A": "First",
        "B": "Second",
        "C": "Third",
        "D": "Fourth",
        "E": "Fifth",
        "F": "Sixth",
        "G": "Seventh",
        "H": "Eighth",
        "I": "Ninth",
        "J": "Tenth",
    }


class TestMedXpertQADataset:
    """Tests for MedXpertQADataset adapter."""

    @pytest.fixture
    def mock_examples(self):
        """Return sample MedXpertQA examples with the fixed A-J schema."""
        return [
            {
                "question": "What is the mechanism of action of metformin?",
                "options": {
                    "A": "Insulin secretion",
                    "B": "Gluconeogenesis inhibition",
                    "C": "Glycogen synthesis",
                    "D": "Lipolysis activation",
                    "E": "Protein kinase C activation",
                    "F": "GLUT4 translocation",
                    "G": "Glycogenolysis inhibition",
                    "H": "Beta-oxidation",
                    "I": "Gluconeogenesis activation",
                    "J": "Insulin receptor downregulation",
                },
                "label": "B",
                "question_type": "reasoning",
                "medical_task": "pharmacology",
                "body_system": "endocrine",
            },
            {
                "question": "Which imaging modality is best for soft tissue?",
                "options": {
                    "A": "X-ray",
                    "B": "CT",
                    "C": "MRI",
                    "D": "Ultrasound",
                    "E": "PET",
                    "F": "SPECT",
                    "G": "Fluoroscopy",
                    "H": "Mammography",
                    "I": "Angiography",
                    "J": "DEXA",
                },
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
    def test_initialization_with_forwarded_kwargs(
        self, mock_load_dataset, mock_examples
    ):
        """Test loader options such as revision/cache_dir are forwarded."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = MedXpertQADataset(revision="abc123", cache_dir="/tmp/cache")

        assert dataset.split == "test"
        mock_load_dataset.assert_called_once_with(
            "TsinghuaC3I/MedXpertQA",
            name="Text",
            split="test",
            streaming=True,
            revision="abc123",
            cache_dir="/tmp/cache",
        )

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_initialization_normalizes_question_type(
        self, mock_load_dataset, mock_examples
    ):
        """Test question_type is lowercased and stripped."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = MedXpertQADataset(question_type="  Reasoning  ")

        assert dataset.question_type == "reasoning"

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_initialization_rejects_unknown_question_type(
        self, mock_load_dataset, mock_examples
    ):
        """Test unsupported question_type values fail fast with ValueError."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        # "reasning" is a deliberate typo for "reasoning" (missing the "o"):
        # exactly the kind of misspelling that used to silently produce an
        # empty evaluation set, and must now raise instead.
        with pytest.raises(ValueError, match="Unsupported question_type.*reasning"):
            MedXpertQADataset(question_type="reasning")

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
    def test_num_options(self, mock_load_dataset, mock_examples):
        """Test num_options returns 10 for MedXpertQA."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        assert dataset.num_options == 10

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_initialization_streaming_is_lazy(self, mock_load_dataset, mock_examples):
        """Test the streaming path keeps filtering lazy (no materialization)."""
        mock_load_dataset.return_value = IterableDataset.from_list(mock_examples)

        dataset = MedXpertQADataset(streaming=True)

        assert isinstance(dataset._dataset, IterableDataset)
        # Nothing is materialized until iterated.
        result = list(dataset._dataset)
        assert len(result) == 2

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_question_type_filter_streaming(self, mock_load_dataset, mock_examples):
        """Test question_type filtering works on the lazy streaming path."""
        mock_load_dataset.return_value = IterableDataset.from_list(mock_examples)

        dataset = MedXpertQADataset(streaming=True, question_type="understanding")

        assert isinstance(dataset._dataset, IterableDataset)
        result = list(dataset._dataset)
        assert len(result) == 1
        assert result[0]["question_type"] == "understanding"

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
            "options": _ten_options(),
            "label": "A",
            "question_type": "reasoning",
        }

        result = dataset._map_example(example)

        assert result["answer"] == "A"
        assert result["info"]["answer_text"] == "First"
        assert "Test?" in result["question"]

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_map_example_verl_valid(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with valid example."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = MedXpertQADataset()

        example = {
            "question": "Test?",
            "options": _ten_options(),
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
    def test_get_verifiers_dataset_streaming(self, mock_load_dataset, mock_examples):
        """Test verifiers projection stays lazy and filters under streaming."""
        mock_load_dataset.return_value = IterableDataset.from_list(mock_examples)
        dataset = MedXpertQADataset(streaming=True)

        result = dataset.get_verifiers_dataset()

        assert isinstance(result, IterableDataset)
        examples = list(result)
        assert len(examples) == 2

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_get_verl_dataset_streaming(self, mock_load_dataset, mock_examples):
        """Test verl projection stays lazy and filters under streaming."""
        mock_load_dataset.return_value = IterableDataset.from_list(mock_examples)
        dataset = MedXpertQADataset(streaming=True)

        result = dataset.get_verl_dataset()

        assert isinstance(result, IterableDataset)
        examples = list(result)
        assert len(examples) == 2

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_get_verifiers_dataset_filters_invalid(self, mock_load_dataset):
        """Test that verifiers dataset filters out invalid examples."""
        examples = [
            {
                "question": "Valid?",
                "options": _ten_options(),
                "label": "A",
                "question_type": "reasoning",
            },
            {
                "question": "Invalid label",
                "options": _ten_options(),
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


class TestMedXpertQAIsValidExample:
    """Tests for _is_valid_example validation logic."""

    @staticmethod
    def _valid_example() -> dict:
        """Return a valid MedXpertQA example."""
        return {
            "question": "Which of the following is true?",
            "options": _ten_options(),
            "label": "A",
            "question_type": "reasoning",
        }

    def test_returns_true_for_valid_example(self):
        """Test valid example passes all checks."""
        assert MedXpertQADataset._is_valid_example(self._valid_example()) is True

    def test_returns_true_for_lowercase_label(self):
        """Test lowercase labels are normalized to uppercase."""
        example = self._valid_example()
        example["label"] = "b"
        assert MedXpertQADataset._is_valid_example(example) is True

    def test_returns_false_for_non_dict_options(self):
        """Test non-dict options are rejected."""
        example = self._valid_example()
        example["options"] = None
        assert MedXpertQADataset._is_valid_example(example) is False

    def test_returns_false_for_wrong_option_count(self):
        """Test rows without exactly ten options are rejected."""
        example = self._valid_example()
        example["options"] = {"A": "One", "B": "Two"}
        assert MedXpertQADataset._is_valid_example(example) is False

    def test_returns_false_when_question_missing(self):
        """Test returns False when question field is missing."""
        example = self._valid_example()
        del example["question"]
        assert MedXpertQADataset._is_valid_example(example) is False

    def test_returns_false_when_question_empty(self):
        """Test returns False when question is an empty string."""
        example = self._valid_example()
        example["question"] = ""
        assert MedXpertQADataset._is_valid_example(example) is False

    def test_returns_false_when_label_missing(self):
        """Test returns False when label is missing."""
        example = self._valid_example()
        del example["label"]
        assert MedXpertQADataset._is_valid_example(example) is False

    def test_returns_false_when_label_not_in_options(self):
        """Test returns False when the label is not among the options."""
        example = self._valid_example()
        example["label"] = "Z"
        assert MedXpertQADataset._is_valid_example(example) is False

    @patch("med_reason_evals.data.medxpertqa.load_dataset")
    def test_get_verl_dataset_filters_wrong_option_count(self, mock_load_dataset):
        """Test rows that violate the ten-option invariant are dropped.

        Uses ``from_generator`` because ``from_list`` infers a union schema
        across rows and pads short options dicts back to ten keys, masking
        the row's real shape. The generator path keeps raw dicts intact,
        matching what the lazy streaming path actually yields.
        """
        valid = self._valid_example()
        wrong_count = {
            "question": "Short options?",
            "options": {"A": "Only", "B": "Two"},
            "label": "A",
            "question_type": "understanding",
        }
        mock_load_dataset.return_value = IterableDataset.from_generator(
            lambda: iter([valid, wrong_count])
        )
        dataset = MedXpertQADataset(streaming=True)

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 1
        assert examples[0]["ground_truth"]["answer"] == "A"
