"""Tests for MMLU-Pro Health dataset adapter.

Tests cover the MMLUProHealthDataset class which filters MMLU-Pro to health
category and handles both letter-based and index-based answer formats.
"""

from unittest.mock import patch

import pytest
from datasets import Dataset

from med_reason_evals.data.mmlu_pro_health import MMLUProHealthDataset


class TestMMLUProHealthDataset:
    """Tests for MMLUProHealthDataset adapter."""

    @pytest.fixture
    def mock_health_examples(self):
        """Return sample health category examples."""
        return [
            {
                "question": "What is the primary function of insulin?",
                "options": [
                    "Regulate blood glucose",
                    "Aid digestion",
                    "Produce antibodies",
                    "Stimulate growth",
                ],
                "answer": "A",  # Letter format
                "category": "health",
                "cot_content": "Insulin helps regulate blood sugar levels.",
            },
            {
                "question": "Which vitamin is produced when skin is exposed to sunlight?",
                "options": ["Vitamin A", "Vitamin C", "Vitamin D", "Vitamin E"],
                "answer": "C",  # Letter format (index handling tested separately)
                "category": "health",
                "cot_content": "Sunlight triggers Vitamin D synthesis.",
            },
            {
                "question": "What is the normal resting heart rate range?",
                "options": ["30-50 bpm", "60-100 bpm", "110-140 bpm", "150-180 bpm"],
                "answer": "b",  # Lowercase letter
                "category": "health",
                "cot_content": "Normal resting heart rate is 60-100 bpm.",
            },
        ]

    @pytest.fixture
    def mock_mixed_examples(self):
        """Return examples with mixed categories including non-health."""
        return [
            {
                "question": "Health question?",
                "options": ["A", "B", "C", "D"],
                "answer": "A",
                "category": "health",
                "cot_content": "",
            },
            {
                "question": "Physics question?",
                "options": ["A", "B", "C", "D"],
                "answer": "B",
                "category": "physics",
                "cot_content": "",
            },
            {
                "question": "Another health question?",
                "options": ["A", "B", "C", "D"],
                "answer": "C",
                "category": "health",
                "cot_content": "",
            },
        ]

    @pytest.fixture
    def mock_dataset(self, mock_health_examples):
        """Return a mock Dataset with health examples."""
        return Dataset.from_list(mock_health_examples)

    def _create_mock_load_dataset(self, examples):
        """Create a mock load_dataset function that returns a filterable dataset."""

        def mock_load(*args, **kwargs):
            dataset = Dataset.from_list(examples)
            # Store original for filtering
            dataset._original_examples = examples
            return dataset

        return mock_load

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_initialization(self, mock_load_dataset, mock_health_examples):
        """Test dataset initialization with default parameters."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)

        dataset = MMLUProHealthDataset()

        assert dataset.split == "test"
        assert dataset.streaming is True
        mock_load_dataset.assert_called_once_with(
            "TIGER-Lab/MMLU-Pro",
            split="test",
            streaming=True,
        )

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_initialization_with_custom_split(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test dataset initialization with custom split."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)

        dataset = MMLUProHealthDataset(split="validation", streaming=False)

        assert dataset.split == "validation"
        assert dataset.streaming is False
        mock_load_dataset.assert_called_once_with(
            "TIGER-Lab/MMLU-Pro",
            split="validation",
            streaming=False,
        )

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_health_category_filtering(self, mock_load_dataset, mock_mixed_examples):
        """Test that only health category examples are retained."""
        mock_load_dataset.return_value = Dataset.from_list(mock_mixed_examples)

        dataset = MMLUProHealthDataset()

        # Convert to list to evaluate the filtered dataset
        result = list(dataset._dataset)

        # Should only have health category questions
        assert len(result) == 2
        for example in result:
            assert example["category"].lower() == "health"

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_build_prompt(self, mock_load_dataset, mock_health_examples):
        """Test prompt building with question and options."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        question = "What is the function?"
        options = ["Option A", "Option B", "Option C"]

        prompt = dataset._build_prompt(question, options)

        assert "Question: What is the function?" in prompt
        assert "A. Option A" in prompt
        assert "B. Option B" in prompt
        assert "C. Option C" in prompt
        assert "Answer:" in prompt

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_build_prompt_with_ten_options(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test prompt building with maximum 10 options."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        question = "Complex question?"
        options = [f"Option {i}" for i in range(10)]

        prompt = dataset._build_prompt(question, options)

        # Should include all 10 letter options
        for letter in "ABCDEFGHIJ":
            assert f"{letter}. Option" in prompt

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_with_letter_answer(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test mapping example with letter-based answer."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["First", "Second", "Third", "Fourth"],
            "answer": "C",
            "category": "health",
        }

        result = dataset._map_example(example)

        assert result is not None
        assert result["answer"] == "C"
        assert result["info"]["answer_text"] == "Third"
        assert result["info"]["category"] == "health"
        assert "Test question?" in result["question"]

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_with_index_answer(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test mapping example with index-based answer."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["First", "Second", "Third", "Fourth"],
            "answer": 1,  # Index format
            "category": "health",
        }

        result = dataset._map_example(example)

        assert result is not None
        assert result["answer"] == "B"  # Converted to letter
        assert result["info"]["answer_text"] == "Second"

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_with_lowercase_letter(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test mapping example with lowercase letter answer."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["First", "Second", "Third", "Fourth"],
            "answer": "b",  # Lowercase
            "category": "health",
        }

        result = dataset._map_example(example)

        assert result is not None
        assert result["answer"] == "B"  # Normalized to uppercase

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_with_empty_question(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test mapping example with empty question returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "   ",  # Whitespace only
            "options": ["A", "B", "C"],
            "answer": "A",
            "category": "health",
        }

        result = dataset._map_example(example)

        assert result["answer"] is None

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_with_empty_options(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test mapping example with empty options returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": [],
            "answer": "A",
            "category": "health",
        }

        result = dataset._map_example(example)

        assert result["answer"] is None

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_with_none_options(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test mapping example with None options returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": None,
            "answer": "A",
            "category": "health",
        }

        result = dataset._map_example(example)

        assert result["answer"] is None

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_with_invalid_answer_letter(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test mapping example with invalid letter answer returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["A", "B", "C", "D"],
            "answer": "Z",  # Invalid letter
            "category": "health",
        }

        result = dataset._map_example(example)

        assert result["answer"] is None

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_with_out_of_range_index(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test mapping example with out-of-range index returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["A", "B", "C", "D"],
            "answer": 10,  # Out of range
            "category": "health",
        }

        result = dataset._map_example(example)

        assert result["answer"] is None

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_with_negative_index(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test mapping example with negative index returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["A", "B", "C", "D"],
            "answer": -1,  # Negative index
            "category": "health",
        }

        result = dataset._map_example(example)

        assert result["answer"] is None

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_index_exceeds_option_count(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test mapping when valid index exceeds available options."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["A", "B"],  # Only 2 options
            "answer": 5,  # Valid as a letter index but exceeds options
            "category": "health",
        }

        result = dataset._map_example(example)

        assert result["answer"] is None

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_letter_exceeds_option_count(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test mapping when valid letter answer exceeds available options (line 77)."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["A", "B"],  # Only 2 options
            "answer": "F",  # Valid letter but exceeds options count
            "category": "health",
        }

        result = dataset._map_example(example)

        assert result["answer"] is None
        assert result["question"] == ""
        assert result["info"] == {}

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_verl_format(self, mock_load_dataset, mock_health_examples):
        """Test mapping example to Verl format."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["First", "Second", "Third"],
            "answer": "B",
            "category": "health",
            "cot_content": "Some reasoning here",
        }

        result = dataset._map_example_verl(example)

        assert result is not None
        assert result["data_source"] == "mmlu_pro_health"
        assert result["prompt"][0]["role"] == "user"
        assert "Test question?" in result["prompt"][0]["content"]
        assert result["ground_truth"]["answer"] == "B"
        assert result["ground_truth"]["answer_text"] == "Second"
        assert result["metadata"]["category"] == "health"
        assert result["metadata"]["cot_content"] == "Some reasoning here"

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_verl_with_index_answer(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test Verl mapping with index-based answer."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["First", "Second", "Third"],
            "answer": 0,  # Index format
            "category": "health",
            "cot_content": "",
        }

        result = dataset._map_example_verl(example)

        assert result is not None
        assert result["ground_truth"]["answer"] == "A"
        assert result["ground_truth"]["answer_text"] == "First"

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_get_verifiers_dataset(self, mock_load_dataset, mock_health_examples):
        """Test getting verifiers-formatted dataset."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        result = dataset.get_verifiers_dataset()

        # Should return a filtered mapped dataset
        assert result is not None
        # Convert to list to verify filtering works
        examples = list(result)
        assert len(examples) == 3
        for ex in examples:
            assert "question" in ex
            assert "answer" in ex
            assert "info" in ex

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_get_verifiers_dataset_filters_invalid(self, mock_load_dataset):
        """Test that verifiers dataset filters out invalid examples."""
        mixed_examples = [
            {
                "question": "Valid question?",
                "options": ["A", "B", "C"],
                "answer": "A",
                "category": "health",
            },
            {
                "question": "",  # Invalid - empty
                "options": ["A", "B", "C"],
                "answer": "A",
                "category": "health",
            },
            {
                "question": "Another valid?",
                "options": ["A", "B", "C"],
                "answer": "Z",  # Invalid - out of range
                "category": "health",
            },
        ]
        mock_load_dataset.return_value = Dataset.from_list(mixed_examples)
        dataset = MMLUProHealthDataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        # Only 1 valid example should remain
        assert len(examples) == 1
        assert examples[0]["answer"] == "A"

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_get_verl_dataset(self, mock_load_dataset, mock_health_examples):
        """Test getting Verl-formatted dataset."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        result = dataset.get_verl_dataset()

        assert result is not None
        examples = list(result)
        assert len(examples) == 3
        for ex in examples:
            assert "prompt" in ex
            assert "ground_truth" in ex
            assert "data_source" in ex
            assert "metadata" in ex

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_get_verl_dataset_filters_invalid(self, mock_load_dataset):
        """Test that Verl dataset filters out invalid examples."""
        mixed_examples = [
            {
                "question": "Valid question?",
                "options": ["A", "B", "C"],
                "answer": "A",
                "category": "health",
                "cot_content": "",
            },
            {
                "question": "",  # Invalid
                "options": ["A", "B", "C"],
                "answer": "A",
                "category": "health",
                "cot_content": "",
            },
        ]
        mock_load_dataset.return_value = Dataset.from_list(mixed_examples)
        dataset = MMLUProHealthDataset()

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 1

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_num_options(self, mock_load_dataset, mock_health_examples):
        """Test num_options returns 10 for MMLU-Pro Health."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        assert dataset.num_options == 10

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_class_constants(self, mock_load_dataset, mock_health_examples):
        """Test class-level constants."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        assert dataset.DATASET_PATH == "TIGER-Lab/MMLU-Pro"
        assert dataset.HEALTH_CATEGORY == "health"
        assert len(dataset.LETTER_INDICES) == 10
        assert dataset.LETTER_INDICES == [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
        ]

    # Tests for covering specific edge cases in _map_example_verl

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_verl_with_empty_options(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test Verl mapping with empty options returns invalid (line 111).

        Covers the case where options list is empty, triggering early return.
        """
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": [],  # Empty options - should trigger line 111
            "answer": "A",
            "category": "health",
            "cot_content": "",
        }

        result = dataset._map_example_verl(example)

        # Should return invalid structure
        assert result["ground_truth"] is None
        assert result["data_source"] == "mmlu_pro_health"
        assert result["prompt"] == []
        assert result["metadata"] == {}

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_verl_with_none_options(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test Verl mapping with None options returns invalid (line 111)."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": None,  # None options - should trigger line 111
            "answer": "A",
            "category": "health",
            "cot_content": "",
        }

        result = dataset._map_example_verl(example)

        assert result["ground_truth"] is None

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_verl_with_empty_question(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test Verl mapping with empty question returns invalid (line 111)."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "   ",  # Whitespace only - should trigger line 111
            "options": ["A", "B", "C"],
            "answer": "A",
            "category": "health",
            "cot_content": "",
        }

        result = dataset._map_example_verl(example)

        assert result["ground_truth"] is None

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_verl_letter_exceeds_options(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test Verl mapping when valid letter exceeds option count (line 114).

        Covers the case where answer is a valid letter but answer_idx >= len(options).
        """
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["A", "B"],  # Only 2 options
            "answer": "F",  # Valid letter but index 5 >= 2 options
            "category": "health",
            "cot_content": "",
        }

        result = dataset._map_example_verl(example)

        # Should return invalid structure - this covers line 114
        assert result["ground_truth"] is None
        assert result["prompt"] == []
        assert result["data_source"] == "mmlu_pro_health"

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_verl_invalid_answer_string(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test Verl mapping with invalid answer string returns invalid (line 111).

        Covers the case where answer is not a valid letter and not an int.
        """
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["A", "B", "C", "D"],
            "answer": "Z",  # Invalid letter - not in LETTER_INDICES
            "category": "health",
            "cot_content": "",
        }

        result = dataset._map_example_verl(example)

        # Should return invalid - this covers the else branch leading to line 111
        assert result["ground_truth"] is None

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_verl_out_of_range_index(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test Verl mapping with out-of-range index returns invalid (line 111)."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["A", "B", "C"],
            "answer": 10,  # Out of range - >= len(options)
            "category": "health",
            "cot_content": "",
        }

        result = dataset._map_example_verl(example)

        # Integer 10 is not < len(options)=3, so falls through to invalid
        assert result["ground_truth"] is None

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_verl_negative_index(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test Verl mapping with negative index returns invalid (line 111)."""
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        example = {
            "question": "Test question?",
            "options": ["A", "B", "C"],
            "answer": -1,  # Negative - fails 0 <= answer check
            "category": "health",
            "cot_content": "",
        }

        result = dataset._map_example_verl(example)

        # Negative index fails the condition 0 <= answer < len(options)
        assert result["ground_truth"] is None

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_verl_answer_idx_exceeds_options(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test Verl mapping where computed answer_idx exceeds options (line 114).

        This specifically tests the check after valid letter is converted to index.
        """
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        # Answer "C" is valid (index 2), but only 2 options means idx 2 >= 2
        example = {
            "question": "Test question?",
            "options": ["A", "B"],  # 2 options: indices 0, 1
            "answer": "C",  # Letter C -> index 2, which is >= 2
            "category": "health",
            "cot_content": "",
        }

        result = dataset._map_example_verl(example)

        # Line 114: answer_idx (2) >= len(options) (2), so return invalid
        assert result["ground_truth"] is None
        assert result["prompt"] == []

    @patch("med_reason_evals.data.mmlu_pro_health.load_dataset")
    def test_map_example_letter_idx_exceeds_options_edge(
        self, mock_load_dataset, mock_health_examples
    ):
        """Test _map_example when letter index exactly equals option count (line 77).

        Edge case where answer_idx == len(options) (boundary condition).
        """
        mock_load_dataset.return_value = Dataset.from_list(mock_health_examples)
        dataset = MMLUProHealthDataset()

        # Exactly at boundary: "C" -> index 2, options length is 2
        # So 2 >= 2 is True, should return invalid
        example = {
            "question": "Test question?",
            "options": ["A", "B"],  # len = 2
            "answer": "C",  # index = 2, so 2 >= 2 -> invalid
            "category": "health",
        }

        result = dataset._map_example(example)

        # Line 77: answer_idx (2) >= len(options) (2), so return empty/None
        assert result["answer"] is None
        assert result["question"] == ""
        assert result["info"] == {}
