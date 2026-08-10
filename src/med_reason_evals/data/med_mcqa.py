"""Dataset adapter for the MedMCQA exam-style question bank.

The adapter reshapes MedMCQA into lettered multiple-choice prompts and handles
the dataset's 1-indexed answer keys so downstream evaluators remain consistent.
"""

from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from med_reason_evals.data.base import BaseDataset


LETTER_INDICES = ["A", "B", "C", "D"]


class MedMCQADataset(BaseDataset):
    """MedMCQA dataset for medical multiple-choice questions.

    The underlying dataset uses 1-indexed answer positions, which are mapped to
    A-D option letters for compatibility with shared evaluation tooling.
    """

    DATASET_PATH = "lighteval/med_mcqa"

    NUM_OPTIONS = 4

    def __init__(
        self,
        split: str = "validation",
        streaming: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the MedMCQA dataset adapter.

        Args:
            split: Dataset split to use ("train" or "validation").
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
        """Return the number of MCQ options (4 for MedMCQA)."""
        return self.NUM_OPTIONS

    @staticmethod
    def _is_valid_example(example: dict[str, Any]) -> bool:
        """Check whether a raw MedMCQA example has all required fields.

        Validates that the example contains a non-empty question, at least one
        non-empty option, and a ``cop`` answer index within the valid 1-4 range.

        Args:
            example: A raw dataset row.

        Returns:
            True if the example is well-formed and usable for evaluation.
        """
        cop = example.get("cop", -1)
        if not isinstance(cop, int) or cop not in [1, 2, 3, 4]:
            return False

        question = example.get("question", "")
        if not isinstance(question, str) or not question.strip():
            return False

        options = [
            (example.get("opa") or "").strip(),
            (example.get("opb") or "").strip(),
            (example.get("opc") or "").strip(),
            (example.get("opd") or "").strip(),
        ]

        return any(options)

    def _build_prompt(
        self,
        question: str,
        options: list[str],
    ) -> str:
        """Build a formatted prompt from question and options.

        The preface explicitly requests a lettered answer to keep model outputs
        aligned with the evaluator's extraction logic.
        """
        query = f"Give a letter answer among A, B, C or D.\n\nQuestion: {question}\n\n"
        query += "".join(
            f"{letter}. {option}\n" for letter, option in zip(LETTER_INDICES, options)
        )
        query += "\nAnswer:"
        return query

    def _map_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to verifiers format.

        Assumes the example has already passed ``_is_valid_example``.
        """
        cop = example["cop"]
        question = (example.get("question") or "").strip()
        options = [
            (example.get("opa") or "").strip(),
            (example.get("opb") or "").strip(),
            (example.get("opc") or "").strip(),
            (example.get("opd") or "").strip(),
        ]

        # Convert 1-indexed labels used by the dataset to zero-based offsets.
        answer_idx = cop - 1
        answer_letter = LETTER_INDICES[answer_idx]

        return {
            "question": self._build_prompt(question, options),
            "answer": answer_letter,
            "info": {
                "answer_text": options[answer_idx],
            },
        }

    def _map_example_verl(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to Verl format.

        Assumes the example has already passed ``_is_valid_example``.
        """
        cop = example["cop"]
        question = (example.get("question") or "").strip()
        options = [
            (example.get("opa") or "").strip(),
            (example.get("opb") or "").strip(),
            (example.get("opc") or "").strip(),
            (example.get("opd") or "").strip(),
        ]

        answer_idx = cop - 1
        answer_letter = LETTER_INDICES[answer_idx]

        prompt = self._build_prompt(question, options)

        return {
            "prompt": [{"role": "user", "content": prompt}],
            "ground_truth": {
                "answer": answer_letter,
                "answer_text": options[answer_idx],
            },
            "data_source": "med_mcqa",
            "metadata": {
                "original_question": question,
                "subject": example.get("subject_name", ""),
                "topic": example.get("topic_name", ""),
            },
        }

    def get_verifiers_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for verifiers evaluation."""
        return self._dataset.filter(self._is_valid_example).map(self._map_example)

    def get_verl_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for Verl training."""
        return self._dataset.filter(self._is_valid_example).map(self._map_example_verl)
