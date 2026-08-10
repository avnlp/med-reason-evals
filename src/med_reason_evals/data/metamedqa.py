"""Dataset adapter for the MetaMedQA multilingual medical QA benchmark.

The adapter handles normalization of multilingual text and maps answer text
back to option letters. This is necessary because MetaMedQA stores the full
answer text rather than option indices, requiring fuzzy matching after
text normalization to handle Unicode variations across languages.

Each raw row carries six lettered choices (A-F), where E and F are the fixed
"None of the above" and "I don't know or cannot answer" responses.
"""

from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from med_reason_evals.data.base import BaseDataset
from med_reason_evals.utils.text import nfkc_casefold, normalize_spaces


class MetaMedQADataset(BaseDataset):
    """MetaMedQA dataset for multilingual medical QA.

    This dataset contains medical questions with answer text
    that needs to be matched to the correct option letter.
    """

    DATASET_PATH = "maximegmd/MetaMedQA"
    NUM_OPTIONS = 6

    def __init__(
        self,
        split: str = "test",
        streaming: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the MetaMedQA dataset.

        Args:
            split: Dataset split to use.
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
        """Return the number of MCQ options (6 for MetaMedQA)."""
        return self.NUM_OPTIONS

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison across languages.

        NFKC normalization handles compatibility characters (e.g., full-width
        digits used in some locales), casefold enables case-insensitive
        matching that works across Unicode scripts, not just ASCII, and
        whitespace runs are collapsed so formatting differences (double spaces,
        tabs, line wraps) do not break exact-string matching.
        """
        return normalize_spaces(nfkc_casefold(text))

    def _find_answer_letter(
        self,
        answer_text: str,
        options: dict[str, str],
    ) -> str | None:
        """Map answer text to its option letter via normalized comparison.

        Since the dataset provides answer text but evaluators expect letters,
        we compare normalized forms to handle minor formatting differences
        (internal whitespace, Unicode variants, casing) that occur in
        multilingual data.
        """
        norm_answer = self._normalize_text(answer_text)
        for letter, text in options.items():
            if self._normalize_text(text) == norm_answer:
                return letter
        return None

    def _build_prompt(self, question: str, options: dict[str, str]) -> str:
        """Build a formatted prompt from question and options."""
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        return f"Question: {question}\n\n{opts}\n\nAnswer:"

    def _is_valid_example(self, example: dict[str, Any]) -> bool:
        """Check whether a raw example has a mappable answer letter.

        Validates that the example contains a non-empty question, a non-empty
        options dict, and an answer text that matches one of the options after
        normalization.
        """
        question = (example.get("question") or "").strip()
        options = example.get("options", {}) or {}
        answer_text = (example.get("answer") or "").strip()

        if not question or not options or not answer_text:
            return False

        return self._find_answer_letter(answer_text, options) is not None

    def _map_example(self, example: dict[str, Any]) -> dict[str, Any] | None:
        """Map a raw example to verifiers format.

        Returns None for rows without a matching answer letter; such rows are
        filtered out before mapping via ``_is_valid_example``.
        """
        question = (example.get("question") or "").strip()
        options = example.get("options", {}) or {}
        answer_text = (example.get("answer") or "").strip()

        if not question or not options or not answer_text:
            return None

        answer_letter = self._find_answer_letter(answer_text, options)
        if answer_letter is None:
            return None

        return {
            "question": self._build_prompt(question, options),
            "answer": answer_letter,
            "info": {
                "answer_text": answer_text,
            },
        }

    def _map_example_verl(self, example: dict[str, Any]) -> dict[str, Any] | None:
        """Map a raw example to Verl format.

        Returns None for rows without a matching answer letter; such rows are
        filtered out before mapping via ``_is_valid_example``.
        """
        question = (example.get("question") or "").strip()
        options = example.get("options", {}) or {}
        answer_text = (example.get("answer") or "").strip()

        if not question or not options or not answer_text:
            return None

        answer_letter = self._find_answer_letter(answer_text, options)
        if answer_letter is None:
            return None

        prompt = self._build_prompt(question, options)

        return {
            "prompt": [{"role": "user", "content": prompt}],
            "ground_truth": {
                "answer": answer_letter,
                "answer_text": answer_text,
            },
            "data_source": "metamedqa",
            "metadata": {
                "original_question": question,
                "language": example.get("language", ""),
            },
        }

    def get_verifiers_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for verifiers evaluation."""
        return self._dataset.filter(self._is_valid_example).map(self._map_example)

    def get_verl_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for Verl training."""
        return self._dataset.filter(self._is_valid_example).map(self._map_example_verl)
