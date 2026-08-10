"""Tests for PubHealthBench dataset adapter.

Tests cover the PubHealthBenchDataset class which handles both multiple-choice
and freeform questions with configurable filtering by question type.
"""

from unittest.mock import patch

import pytest
from datasets import Dataset

from med_reason_evals.data.pubhealthbench import PubHealthBenchDataset


class TestPubHealthBenchDataset:
    """Tests for PubHealthBenchDataset adapter."""

    @pytest.fixture
    def mock_mcq_examples(self):
        """Return sample MCQ examples."""
        return [
            {
                "question": "What is the primary cause of type 2 diabetes?",
                "options": [
                    "Genetic factors only",
                    "Insulin resistance and lifestyle factors",
                    "Viral infections",
                    "Bacterial infections",
                ],
                "answer": "Insulin resistance and lifestyle factors",
                "source": "test_source_1",
            },
            {
                "question": "Which nutrient is essential for blood clotting?",
                "options": ["Vitamin A", "Vitamin K", "Vitamin C", "Vitamin D"],
                "answer": "B",  # Letter answer
                "source": "test_source_2",
            },
        ]

    @pytest.fixture
    def mock_freeform_examples(self):
        """Return sample freeform examples."""
        return [
            {
                "question": "Describe the symptoms of dehydration.",
                "options": ["dehydration"],  # Single item indicates freeform
                "answer": "Thirst, dry mouth, fatigue, dizziness, decreased urine output",
                "source": "test_source_3",
            },
            {
                "question": "Explain how vaccines work.",
                "options": [],  # Empty options indicates freeform
                "answer": "Vaccines stimulate the immune system to produce antibodies.",
                "source": "test_source_4",
            },
        ]

    @pytest.fixture
    def mock_mixed_examples(self, mock_mcq_examples, mock_freeform_examples):
        """Return mixed MCQ and freeform examples."""
        return mock_mcq_examples + mock_freeform_examples

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_initialization_default(self, mock_load_dataset, mock_mixed_examples):
        """Test dataset initialization with default parameters (all types)."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)

        dataset = PubHealthBenchDataset()

        assert dataset.split == "test"
        assert dataset.streaming is True
        assert dataset.question_type == "all"
        mock_load_dataset.assert_called_once_with(
            "Joshua-Harris/PubHealthBench",
            split="test",
            streaming=True,
        )

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_initialization_mcq_filter(self, mock_load_dataset, mock_mixed_examples):
        """Test dataset initialization with MCQ filter."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)

        dataset = PubHealthBenchDataset(question_type="mcq")

        assert dataset.question_type == "mcq"
        # Should filter to only MCQ examples (2 MCQs in mixed_examples)
        result = list(dataset._dataset)
        assert len(result) == 2
        for ex in result:
            assert len(ex.get("options", [])) > 1

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_initialization_freeform_filter(
        self, mock_load_dataset, mock_mixed_examples
    ):
        """Test dataset initialization with freeform filter."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)

        dataset = PubHealthBenchDataset(question_type="freeform")

        assert dataset.question_type == "freeform"
        # Should filter to only freeform examples (2 freeform in mixed_examples)
        result = list(dataset._dataset)
        assert len(result) == 2
        for ex in result:
            options = ex.get("options", [])
            # Freeform has 0 or 1 options
            assert len(options) <= 1

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_initialization_uppercase_type(
        self, mock_load_dataset, mock_mixed_examples
    ):
        """Test that question_type is normalized to lowercase."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)

        dataset = PubHealthBenchDataset(question_type="MCQ")

        assert dataset.question_type == "mcq"

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_initialization_whitespace_stripped(
        self, mock_load_dataset, mock_mixed_examples
    ):
        """Test that question_type whitespace is stripped."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)

        dataset = PubHealthBenchDataset(question_type="  mcq  ")

        assert dataset.question_type == "mcq"

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_is_mcq_with_multiple_options(self, mock_load_dataset, mock_mixed_examples):
        """Test _is_mcq with multiple options."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {"options": ["A", "B", "C", "D"]}
        assert dataset._is_mcq(example) is True

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_is_mcq_with_single_option(self, mock_load_dataset, mock_mixed_examples):
        """Test _is_mcq with single option (freeform indicator)."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {"options": ["answer"]}
        assert dataset._is_mcq(example) is False

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_is_mcq_with_empty_options(self, mock_load_dataset, mock_mixed_examples):
        """Test _is_mcq with empty options."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {"options": []}
        assert dataset._is_mcq(example) is False

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_is_mcq_with_none_options(self, mock_load_dataset, mock_mixed_examples):
        """Test _is_mcq with None options."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {"options": None}
        assert dataset._is_mcq(example) is False

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_is_mcq_with_missing_options(self, mock_load_dataset, mock_mixed_examples):
        """Test _is_mcq with missing options key."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {}
        assert dataset._is_mcq(example) is False

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_build_mcq_prompt(self, mock_load_dataset, mock_mixed_examples):
        """Test building MCQ prompt."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        question = "What is the cause?"
        options = ["Option A", "Option B", "Option C"]

        prompt = dataset._build_mcq_prompt(question, options)

        assert "Question: What is the cause?" in prompt
        assert "A. Option A" in prompt
        assert "B. Option B" in prompt
        assert "C. Option C" in prompt
        assert "Answer:" in prompt

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_build_freeform_prompt(self, mock_load_dataset, mock_mixed_examples):
        """Test building freeform prompt."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        question = "Describe the process."

        prompt = dataset._build_freeform_prompt(question)

        assert prompt == "Question: Describe the process.\n\nAnswer:"

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_mcq_with_text_answer(
        self, mock_load_dataset, mock_mixed_examples
    ):
        """Test mapping MCQ example with text answer matching option."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "What is the cause?",
            "options": ["Wrong answer", "Correct answer", "Another wrong"],
            "answer": "Correct answer",
        }

        result = dataset._map_example(example)

        assert result["answer"] == "B"
        assert result["info"]["is_mcq"] is True
        assert result["info"]["answer_text"] == "Correct answer"
        assert "What is the cause?" in result["question"]

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_mcq_with_letter_answer(
        self, mock_load_dataset, mock_mixed_examples
    ):
        """Test mapping MCQ example with letter answer."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "What is the cause?",
            "options": ["Option A", "Option B", "Option C"],
            "answer": "C",
        }

        result = dataset._map_example(example)

        assert result["answer"] == "C"
        assert result["info"]["answer_text"] == "Option C"

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_mcq_with_lowercase_letter(
        self, mock_load_dataset, mock_mixed_examples
    ):
        """Test mapping MCQ with lowercase letter answer."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "What is the cause?",
            "options": ["Option A", "Option B", "Option C"],
            "answer": "b",
        }

        result = dataset._map_example(example)

        assert result["answer"] == "B"

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_mcq_seven_options_with_answer_index(
        self, mock_load_dataset, mock_mixed_examples
    ):
        """Test mapping MCQ with seven options resolved via answer_index."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "What is the cause?",
            "options": ["o1", "o2", "o3", "o4", "o5", "o6", "o7"],
            "answer": "o7",
            "answer_index": 6,
        }

        result = dataset._map_example(example)

        assert result["answer"] == "G"
        assert result["info"]["answer_text"] == "o7"
        assert "G. o7" in result["question"]

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_verl_mcq_seven_options_with_answer_index(
        self, mock_load_dataset, mock_mixed_examples
    ):
        """Test Verl mapping with seven options resolved via answer_index."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "What is the cause?",
            "options": ["o1", "o2", "o3", "o4", "o5", "o6", "o7"],
            "answer": "o7",
            "answer_index": 6,
        }

        result = dataset._map_example_verl(example)

        assert result["ground_truth"]["answer"] == "G"
        assert result["ground_truth"]["answer_text"] == "o7"
        assert "G. o7" in result["prompt"][0]["content"]

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_mcq_out_of_range_answer_index(
        self, mock_load_dataset, mock_mixed_examples
    ):
        """Test that an out-of-range answer_index degrades to None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "What is the cause?",
            "options": ["o1", "o2", "o3"],
            "answer": "nonexistent",
            "answer_index": 99,
        }

        result = dataset._map_example(example)

        assert result["answer"] is None
        assert result["question"] == ""

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_mcq_no_match(self, mock_load_dataset, mock_mixed_examples):
        """Test mapping MCQ where answer doesn't match any option or letter."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "What is the cause?",
            "options": ["Option A", "Option B", "Option C"],
            "answer": "Completely different answer",
        }

        result = dataset._map_example(example)

        # Returns empty question with None answer for filtering
        assert result["answer"] is None
        assert result["question"] == ""

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_freeform(self, mock_load_dataset, mock_mixed_examples):
        """Test mapping freeform example."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "Describe the symptoms.",
            "options": [],
            "answer": "Various symptoms here",
        }

        result = dataset._map_example(example)

        assert result["answer"] == "Various symptoms here"
        assert result["info"]["is_mcq"] is False
        assert "Describe the symptoms." in result["question"]

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_empty_question(self, mock_load_dataset, mock_mixed_examples):
        """Test mapping example with empty question."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "   ",  # Whitespace only
            "options": ["A", "B", "C"],
            "answer": "A",
        }

        result = dataset._map_example(example)

        assert result["answer"] is None
        assert result["question"] == ""

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_empty_answer(self, mock_load_dataset, mock_mixed_examples):
        """Test mapping example with empty answer."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "What is the cause?",
            "options": ["A", "B", "C"],
            "answer": "   ",  # Whitespace only
        }

        result = dataset._map_example(example)

        assert result["answer"] is None
        assert result["question"] == ""

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_none_answer(self, mock_load_dataset, mock_mixed_examples):
        """Test mapping example with None answer."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "What is the cause?",
            "options": ["A", "B", "C"],
            "answer": None,
        }

        result = dataset._map_example(example)

        assert result["answer"] is None
        assert result["question"] == ""

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_verl_mcq(self, mock_load_dataset, mock_mixed_examples):
        """Test mapping MCQ example to Verl format."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "What is the cause?",
            "options": ["Wrong", "Correct", "Also wrong"],
            "answer": "Correct",
            "source": "test_source",
        }

        result = dataset._map_example_verl(example)

        assert result["data_source"] == "pubhealthbench"
        assert result["prompt"][0]["role"] == "user"
        assert "What is the cause?" in result["prompt"][0]["content"]
        assert result["ground_truth"]["answer"] == "B"
        assert result["ground_truth"]["answer_text"] == "Correct"
        assert result["metadata"]["is_mcq"] is True
        assert result["metadata"]["source"] == "test_source"

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_verl_freeform(self, mock_load_dataset, mock_mixed_examples):
        """Test mapping freeform example to Verl format."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "Describe the process.",
            "options": [],
            "answer": "Detailed explanation here",
            "source": "another_source",
        }

        result = dataset._map_example_verl(example)

        assert result["data_source"] == "pubhealthbench"
        assert result["ground_truth"]["answer"] == "Detailed explanation here"
        assert result["ground_truth"]["target"] == "Detailed explanation here"
        assert result["metadata"]["is_mcq"] is False

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_verl_empty_question(
        self, mock_load_dataset, mock_mixed_examples
    ):
        """Test Verl mapping with empty question returns null ground_truth."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "",
            "options": ["A", "B"],
            "answer": "A",
        }

        result = dataset._map_example_verl(example)

        assert result["ground_truth"] is None
        assert result["prompt"] == []

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_map_example_verl_mcq_no_match(
        self, mock_load_dataset, mock_mixed_examples
    ):
        """Test Verl MCQ mapping when answer doesn't match."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        example = {
            "question": "What is the cause?",
            "options": ["A", "B", "C"],
            "answer": "No match",
        }

        result = dataset._map_example_verl(example)

        assert result["ground_truth"] is None
        assert result["prompt"] == []

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_get_verifiers_dataset(self, mock_load_dataset, mock_mixed_examples):
        """Test getting verifiers-formatted dataset."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        result = dataset.get_verifiers_dataset()

        assert result is not None
        examples = list(result)
        # All 4 examples are valid
        assert len(examples) == 4

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_get_verifiers_dataset_filters_null_answers(self, mock_load_dataset):
        """Test that verifiers dataset filters out null answers."""
        examples = [
            {
                "question": "Valid?",
                "options": ["A", "B"],
                "answer": "A",
            },
            {
                "question": "",  # Invalid
                "options": ["A", "B"],
                "answer": "A",
            },
        ]
        mock_load_dataset.return_value = Dataset.from_list(examples)
        dataset = PubHealthBenchDataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 1
        assert examples[0]["answer"] == "A"

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_get_verl_dataset(self, mock_load_dataset, mock_mixed_examples):
        """Test getting Verl-formatted dataset."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        result = dataset.get_verl_dataset()

        assert result is not None
        examples = list(result)
        assert len(examples) == 4

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_get_verl_dataset_filters_null_ground_truth(self, mock_load_dataset):
        """Test that Verl dataset filters out null ground_truth."""
        examples = [
            {
                "question": "Valid?",
                "options": ["A", "B"],
                "answer": "A",
            },
            {
                "question": "",  # Invalid
                "options": ["A", "B"],
                "answer": "A",
            },
        ]
        mock_load_dataset.return_value = Dataset.from_list(examples)
        dataset = PubHealthBenchDataset()

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 1

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_mcq_filter_integration(self, mock_load_dataset, mock_mixed_examples):
        """Test MCQ filter with get_verifiers_dataset integration."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset(question_type="mcq")

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        # Should have 2 MCQ examples
        assert len(examples) == 2
        for ex in examples:
            assert ex["info"]["is_mcq"] is True

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_freeform_filter_integration(self, mock_load_dataset, mock_mixed_examples):
        """Test freeform filter with get_verl_dataset integration."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset(question_type="freeform")

        result = dataset.get_verl_dataset()
        examples = list(result)

        # Should have 2 freeform examples
        assert len(examples) == 2
        for ex in examples:
            assert ex["metadata"]["is_mcq"] is False

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_num_options(self, mock_load_dataset, mock_mixed_examples):
        """Test num_options returns 1 for the hybrid dataset."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        assert dataset.num_options == 1

    @patch("med_reason_evals.data.pubhealthbench.load_dataset")
    def test_dataset_path_constant(self, mock_load_dataset, mock_mixed_examples):
        """Test dataset path constant."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)
        dataset = PubHealthBenchDataset()

        assert dataset.DATASET_PATH == "Joshua-Harris/PubHealthBench"
