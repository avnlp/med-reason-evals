"""Dataset adapter for the MedQA USMLE-style multiple-choice benchmark.

The adapter reshapes MedQA questions into a fixed multiple-choice prompt for
consistent evaluation across Verifiers and Verl pipelines.
"""

from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from med_reason_evals.data.base import BaseDataset


class MedQADataset(BaseDataset):
    """MedQA-USMLE-4-options dataset for medical question answering.

    The dataset mirrors USMLE-style questions and expects lettered responses.
    """

    DATASET_PATH = "GBaker/MedQA-USMLE-4-options"

    NUM_OPTIONS = 4

    def __init__(
        self,
        split: str = "test",
        streaming: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the MedQA dataset adapter.

        Args:
            split: Dataset split to use ("train" or "test").
            streaming: Whether to stream the dataset.
            **kwargs: Additional keyword arguments forwarded to
                ``load_dataset()`` (e.g. ``revision``, ``cache_dir``).
        """
        super().__init__(split=split, streaming=streaming, **kwargs)
        self._dataset = load_dataset(
            self.DATASET_PATH,
            split=split,
            streaming=streaming,
            **kwargs,
        )

    @property
    def num_options(self) -> int:
        """Return the number of MCQ options (4 for MedQA)."""
        return self.NUM_OPTIONS

    @staticmethod
    def _is_valid_example(example: dict[str, Any]) -> bool:
        """Check whether a raw MedQA example has all required fields.

        Validates that the example contains a non-empty question, a dict of
        options with at least one entry, and a non-empty answer index that
        refers to a key in the options dict.

        Args:
            example: A raw dataset row.

        Returns:
            True if the example is well-formed and usable for evaluation.
        """
        question = example.get("question", "")
        if not isinstance(question, str) or not question.strip():
            return False

        options = example.get("options", {})
        if not isinstance(options, dict) or not options:
            return False

        answer_idx = example.get("answer_idx", "")
        if not isinstance(answer_idx, str) or not answer_idx.strip():
            return False

        return answer_idx.strip().upper() in options

    def _build_prompt(self, question: str, options: dict[str, str]) -> str:
        """Build a formatted prompt from question and options.

        The prompt is deliberately minimal to match the original dataset style
        while keeping option labeling explicit for downstream parsing.
        """
        opts = "\n".join(f"{k}. {v}" for k, v in options.items())
        return f"Question: {question}\n\n{opts}\n\nAnswer:"

    def _map_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to verifiers format.

        Unlike other datasets, MedQA examples are well-formed, so the mapping
        passes through without additional filtering.
        """
        question = example.get("question", "")
        options = example.get("options", {})
        answer_letter = example.get("answer_idx", "").strip().upper()

        return {
            "question": self._build_prompt(question, options),
            "answer": answer_letter,
            "info": {
                "answer_text": options.get(answer_letter, ""),
                "original_question": question,
            },
        }

    def _map_example_verl(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to Verl format.

        The raw options are preserved in metadata to support alternative reward
        strategies that consider answer text directly.
        """
        question = example.get("question", "")
        options = example.get("options", {})
        answer_letter = example.get("answer_idx", "").strip().upper()

        prompt = self._build_prompt(question, options)

        return {
            "prompt": [{"role": "user", "content": prompt}],
            "ground_truth": {
                "answer": answer_letter,
                "answer_text": options.get(answer_letter, ""),
            },
            "data_source": "medqa",
            "metadata": {
                "original_question": question,
                "options": options,
            },
        }

    def get_verifiers_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for verifiers evaluation."""
        return self._dataset.filter(self._is_valid_example).map(self._map_example)

    def get_verl_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for Verl training."""
        return self._dataset.filter(self._is_valid_example).map(self._map_example_verl)
