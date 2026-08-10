"""Dataset adapter for MMLU-Pro health category questions.

The adapter filters the MMLU-Pro dataset to the health category and handles
the variable-length option lists (up to 10 choices). Answer encoding can be
either letter-based or index-based, so the adapter normalizes both forms to
a consistent letter format for evaluation.
"""

from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from med_reason_evals.data.base import BaseDataset


class MMLUProHealthDataset(BaseDataset):
    """MMLUProHealth dataset for professional-level medical questions.

    This dataset contains challenging multiple-choice questions from the
    health category of MMLU-Pro, typically requiring chain-of-thought reasoning.
    """

    DATASET_PATH = "TIGER-Lab/MMLU-Pro"
    HEALTH_CATEGORY = "health"
    LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    def __init__(
        self,
        split: str = "test",
        streaming: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the MMLUProHealth dataset.

        Args:
            split: Dataset split to use.
            streaming: Whether to stream the dataset.
            **kwargs: Additional arguments.
        """
        super().__init__(split=split, streaming=streaming, **kwargs)
        dataset = load_dataset(
            self.DATASET_PATH,
            split=split,
            streaming=streaming,
        )
        # Filter to health category
        self._dataset = dataset.filter(
            lambda x: (x.get("category") or "").lower() == self.HEALTH_CATEGORY
        )

    @property
    def num_options(self) -> int:
        """Return the maximum number of MCQ options (10 for MMLU-Pro Health).

        MMLU-Pro questions have variable option counts (2-10), so this returns
        the maximum supported by the letter index table.

        Returns:
            The number of letters available for option labeling.
        """
        return len(self.LETTER_INDICES)

    def _build_prompt(self, question: str, options: list[str]) -> str:
        """Build a formatted prompt from question and options."""
        letters = self.LETTER_INDICES[: len(options)]
        opts = "\n".join(f"{ltr}. {opt}" for ltr, opt in zip(letters, options))
        return f"Question: {question}\n\n{opts}\n\nAnswer:"

    def _map_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to verifiers format."""
        question = (example.get("question") or "").strip()
        options = [
            o
            for o in (example.get("options", []) or [])
            if o and o.strip().upper() != "N/A"
        ]
        answer = example.get("answer", "")

        if not question or not options:
            return {"question": "", "answer": None, "info": {}}

        # Answer can be a letter or index
        if isinstance(answer, str) and answer.upper() in self.LETTER_INDICES:
            answer_letter = answer.upper()
            answer_idx = self.LETTER_INDICES.index(answer_letter)
        elif isinstance(answer, int) and 0 <= answer < len(options):
            answer_idx = answer
            answer_letter = self.LETTER_INDICES[answer_idx]
        else:
            return {"question": "", "answer": None, "info": {}}

        if answer_idx >= len(options):
            return {"question": "", "answer": None, "info": {}}

        return {
            "question": self._build_prompt(question, options),
            "answer": answer_letter,
            "info": {
                "answer_text": options[answer_idx],
                "category": self.HEALTH_CATEGORY,
            },
        }

    def _map_example_verl(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to Verl format."""
        question = (example.get("question") or "").strip()
        options = [
            o
            for o in (example.get("options", []) or [])
            if o and o.strip().upper() != "N/A"
        ]
        answer = example.get("answer", "")

        _invalid = {
            "prompt": [],
            "ground_truth": None,
            "data_source": "mmlu_pro_health",
            "metadata": {},
        }

        if not question or not options:
            return _invalid

        if isinstance(answer, str) and answer.upper() in self.LETTER_INDICES:
            answer_letter = answer.upper()
            answer_idx = self.LETTER_INDICES.index(answer_letter)
        elif isinstance(answer, int) and 0 <= answer < len(options):
            answer_idx = answer
            answer_letter = self.LETTER_INDICES[answer_idx]
        else:
            return _invalid

        if answer_idx >= len(options):
            return _invalid

        prompt = self._build_prompt(question, options)

        return {
            "prompt": [{"role": "user", "content": prompt}],
            "ground_truth": {
                "answer": answer_letter,
                "answer_text": options[answer_idx],
            },
            "data_source": "mmlu_pro_health",
            "metadata": {
                "category": self.HEALTH_CATEGORY,
                "cot_content": example.get("cot_content", ""),
            },
        }

    def get_verifiers_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for verifiers evaluation."""
        mapped = self._dataset.map(self._map_example)
        return mapped.filter(lambda x: x is not None and x.get("answer") is not None)

    def get_verl_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for Verl training."""
        mapped = self._dataset.map(self._map_example_verl)
        return mapped.filter(
            lambda x: x is not None and x.get("ground_truth") is not None
        )
