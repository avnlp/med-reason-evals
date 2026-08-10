"""Tests for HealthBench dataset adapter.

Tests cover the HealthBenchDataset class which handles rubric-based
evaluation prompts with single and multi-turn conversations.
"""

from unittest.mock import patch

import pytest
from datasets import Dataset, IterableDataset

from med_reason_evals.data.healthbench import HealthBenchDataset


class TestHealthBenchDataset:
    """Tests for HealthBenchDataset adapter."""

    @pytest.fixture
    def mock_examples(self):
        """Return sample HealthBench examples."""
        return [
            {
                "prompt": "What are the symptoms of diabetes?",
                "prompt_id": "p1",
                "rubrics": [
                    {
                        "criterion": "Mentions polyuria",
                        "points": 1,
                        "tags": ["axis:completeness"],
                    },
                    {
                        "criterion": "Mentions polydipsia",
                        "points": 1,
                        "tags": ["axis:safety"],
                    },
                ],
                "ideal_completions_data": ["Symptoms include..."],
            },
            {
                "prompt": "Explain blood pressure ranges.",
                "prompt_id": "p2",
                "rubrics": [
                    {
                        "criterion": "Mentions normal range",
                        "points": 2,
                        "tags": [],
                    },
                ],
                "ideal_completions_data": [],
            },
        ]

    @pytest.fixture
    def mock_multi_turn_examples(self):
        """Return examples with multi-turn prompts."""
        return [
            {
                "prompt": [
                    {"role": "user", "content": "I have a headache."},
                    {"role": "assistant", "content": "How long have you had it?"},
                    {"role": "user", "content": "About 3 days."},
                ],
                "prompt_id": "p3",
                "rubrics": [
                    {
                        "criterion": "Asks relevant follow-up",
                        "points": 1,
                        "tags": ["axis:completeness"],
                    },
                ],
                "ideal_completions_data": [],
            },
        ]

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_initialization(self, mock_load_dataset, mock_examples):
        """Test dataset initialization with default parameters."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = HealthBenchDataset()

        assert dataset.split == "test"
        assert dataset.streaming is True
        assert dataset.difficulty == "regular"
        mock_load_dataset.assert_called_once_with(
            "neuralleap/healthbench-regular",
            split="test",
            streaming=True,
        )

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_initialization_hard_difficulty(self, mock_load_dataset, mock_examples):
        """Test initialization with hard difficulty uses train split."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = HealthBenchDataset(difficulty="hard")

        mock_load_dataset.assert_called_once_with(
            "neuralleap/healthbench-hard",
            split="train",
            streaming=True,
        )
        assert dataset.split == "train"

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_initialization_explicit_split(self, mock_load_dataset, mock_examples):
        """Test an explicitly requested split is honored over the difficulty default."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)

        dataset = HealthBenchDataset(split="validation")

        assert dataset.split == "validation"
        mock_load_dataset.assert_called_once_with(
            "neuralleap/healthbench-regular",
            split="validation",
            streaming=True,
        )

    def test_initialization_invalid_difficulty(self):
        """Test initialization with invalid difficulty raises ValueError."""
        with pytest.raises(ValueError, match="Invalid difficulty"):
            HealthBenchDataset(difficulty="extreme")

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_num_options(self, mock_load_dataset, mock_examples):
        """Test num_options returns 1 for non-MCQ rubric dataset."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = HealthBenchDataset()

        assert dataset.num_options == 1

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_extract_prompt_string(self, mock_load_dataset, mock_examples):
        """Test extracting prompt from a string."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = HealthBenchDataset()

        result = dataset._extract_prompt({"prompt": "Simple question"})

        assert result == "Simple question"

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_extract_prompt_multi_turn(self, mock_load_dataset, mock_examples):
        """Test extracting prompt from multi-turn message list."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = HealthBenchDataset()

        example = {
            "prompt": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "Help me"},
            ]
        }

        result = dataset._extract_prompt(example)

        assert "Hello" in result
        assert "Hi there" in result
        assert "Help me" in result

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_map_example_valid(self, mock_load_dataset, mock_examples):
        """Test mapping a valid example with rubrics."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = HealthBenchDataset()

        result = dataset._map_example(mock_examples[0])

        assert result is not None
        assert result["question"] == "What are the symptoms of diabetes?"
        assert result["answer"] == ""
        assert len(result["info"]["criteria"]) == 2
        assert result["info"]["points_list"] == [1, 1]
        assert result["info"]["axes"] == ["completeness", "safety"]

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_map_example_empty_prompt(self, mock_load_dataset, mock_examples):
        """Test mapping with empty prompt returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = HealthBenchDataset()

        result = dataset._map_example({"prompt": "", "rubrics": []})

        assert result is None

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_map_example_verl_string_prompt(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with a string prompt."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = HealthBenchDataset()

        result = dataset._map_example_verl(mock_examples[0])

        assert result is not None
        assert result["data_source"] == "healthbench"
        assert result["prompt"] == [
            {"role": "user", "content": "What are the symptoms of diabetes?"}
        ]
        assert len(result["ground_truth"]["criteria"]) == 2

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_map_example_verl_multi_turn(
        self, mock_load_dataset, mock_multi_turn_examples
    ):
        """Test Verl mapping preserves multi-turn structure."""
        mock_load_dataset.return_value = Dataset.from_list(mock_multi_turn_examples)
        dataset = HealthBenchDataset()

        result = dataset._map_example_verl(mock_multi_turn_examples[0])

        assert result is not None
        assert len(result["prompt"]) == 3
        assert result["prompt"][0]["role"] == "user"
        assert result["prompt"][1]["role"] == "assistant"

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_map_example_verl_empty_prompt(self, mock_load_dataset, mock_examples):
        """Test Verl mapping with empty prompt returns None."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = HealthBenchDataset()

        result = dataset._map_example_verl({"prompt": "", "rubrics": []})

        assert result is None

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_get_verifiers_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end verifiers dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = HealthBenchDataset()

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert "question" in ex
            assert "info" in ex

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_get_verl_dataset(self, mock_load_dataset, mock_examples):
        """Test end-to-end Verl dataset generation."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = HealthBenchDataset()

        result = dataset.get_verl_dataset()
        examples = list(result)

        assert len(examples) == 2
        for ex in examples:
            assert "prompt" in ex
            assert "ground_truth" in ex
        # Variant routing is recorded in metadata; guard against a regression
        # where a row gets tagged with the wrong difficulty.
        assert examples[0]["metadata"]["difficulty"] == "regular"

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_get_verifiers_dataset_streaming(self, mock_load_dataset, mock_examples):
        """Test verifiers projection stays lazy and filters under streaming."""
        mock_load_dataset.return_value = IterableDataset.from_list(mock_examples)
        dataset = HealthBenchDataset(streaming=True)

        result = dataset.get_verifiers_dataset()

        assert isinstance(result, IterableDataset)
        examples = list(result)
        assert len(examples) == 2

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_get_verl_dataset_streaming(self, mock_load_dataset, mock_examples):
        """Test verl projection stays lazy and filters under streaming."""
        mock_load_dataset.return_value = IterableDataset.from_list(mock_examples)
        dataset = HealthBenchDataset(streaming=True)

        result = dataset.get_verl_dataset()

        assert isinstance(result, IterableDataset)
        examples = list(result)
        assert len(examples) == 2

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_get_verifiers_dataset_filters_invalid(self, mock_load_dataset):
        """Test that rows without rubrics or prompts are dropped.

        Uses ``from_generator`` because ``from_list`` infers a union schema
        across rows and pads short dicts, masking the row's real shape.
        """
        valid = {
            "prompt": "What should I do?",
            "prompt_id": "p1",
            "rubrics": [{"criterion": "C1", "points": 1, "tags": []}],
        }
        malformed = {
            "prompt": "No rubric",
            "prompt_id": "p2",
            "rubrics": [],
        }
        mock_load_dataset.return_value = IterableDataset.from_generator(
            lambda: iter([valid, malformed])
        )
        dataset = HealthBenchDataset(streaming=True)

        result = dataset.get_verifiers_dataset()
        examples = list(result)

        assert len(examples) == 1
        assert examples[0]["info"]["prompt_id"] == "p1"

    @patch("med_reason_evals.data.healthbench.load_dataset")
    def test_rubric_axis_extraction(self, mock_load_dataset, mock_examples):
        """Test that axis tags are correctly extracted from rubrics."""
        mock_load_dataset.return_value = Dataset.from_list(mock_examples)
        dataset = HealthBenchDataset()

        example = {
            "prompt": "Test",
            "rubrics": [
                {"criterion": "C1", "points": 1, "tags": ["axis:accuracy", "other"]},
                {"criterion": "C2", "points": 2, "tags": []},
            ],
        }

        result = dataset._extract_rubric_info(example)

        assert result["axes"] == ["accuracy", ""]
        assert result["points_list"] == [1, 2]


class TestHealthBenchIsValidExample:
    """Tests for _is_valid_example validation logic."""

    @staticmethod
    def _valid_example() -> dict:
        """Return a valid HealthBench example."""
        return {
            "prompt": "What should I do for a headache?",
            "prompt_id": "p1",
            "rubrics": [{"criterion": "Mentions rest", "points": 1, "tags": []}],
        }

    def test_returns_true_for_valid_example(self):
        """Test valid example passes all checks."""
        assert HealthBenchDataset._is_valid_example(self._valid_example()) is True

    def test_returns_true_for_multi_turn_prompt(self):
        """Test multi-turn message-list prompts are accepted."""
        example = self._valid_example()
        example["prompt"] = [{"role": "user", "content": "Help me"}]
        assert HealthBenchDataset._is_valid_example(example) is True

    def test_returns_false_when_prompt_missing(self):
        """Test returns False when prompt is missing."""
        example = self._valid_example()
        del example["prompt"]
        assert HealthBenchDataset._is_valid_example(example) is False

    def test_returns_false_when_prompt_empty(self):
        """Test returns False when prompt is empty."""
        example = self._valid_example()
        example["prompt"] = ""
        assert HealthBenchDataset._is_valid_example(example) is False

    def test_returns_false_when_rubrics_missing(self):
        """Test returns False when rubrics are missing."""
        example = self._valid_example()
        del example["rubrics"]
        assert HealthBenchDataset._is_valid_example(example) is False

    def test_returns_false_when_rubrics_empty(self):
        """Test returns False when rubrics have no criteria."""
        example = self._valid_example()
        example["rubrics"] = [{"points": 1, "tags": []}]
        assert HealthBenchDataset._is_valid_example(example) is False
