"""Dataset adapter for MMLU-Pro health category questions.

The adapter filters the MMLU-Pro dataset to the health category and handles
the variable-length option lists (up to 10 choices). Answer encoding can be
either letter-based or index-based, so the adapter normalizes both forms to
a consistent letter format for evaluation.
"""

from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from med_reason_evals.data.base import BaseDataset


# Columns attached to every row by ``_normalize_row`` so the normalized form
# is computed exactly once per row and reused by the validity filter and the
# mappers, instead of being re-derived during both the filter and map passes.
_NORMALIZED_COLUMNS = [
    "_valid",
    "_question",
    "_options",
    "_answer_letter",
    "_answer_idx",
]


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
            **kwargs: Additional keyword arguments forwarded to
                ``load_dataset()`` (e.g. ``revision``, ``cache_dir``).
        """
        super().__init__(split=split, streaming=streaming, **kwargs)
        dataset = load_dataset(
            self.DATASET_PATH,
            split=split,
            streaming=streaming,
            **kwargs,
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

    def _normalize_example(
        self, example: dict[str, Any]
    ) -> tuple[str, list[str], str, int] | None:
        """Normalize a raw example's options and answer to letter form.

        Applies the shared validation rules used by both output schemas:
        strips the question, drops empty/``N/A`` options, and resolves the
        answer (letter or index) to a letter plus its index into the filtered
        options list.

        Returns:
            A ``(question, options, answer_letter, answer_idx)`` tuple, or
            ``None`` if the example cannot be mapped (blank question, no
            usable options, or an answer that cannot be resolved to a valid
            letter within range).
        """
        question = (example.get("question") or "").strip()
        options = [
            o
            for o in (example.get("options", []) or [])
            if o and o.strip().upper() != "N/A"
        ]
        answer = example.get("answer", "")

        if not question or not options:
            return None

        # Answer can be a letter or index
        if isinstance(answer, str) and answer.upper() in self.LETTER_INDICES:
            answer_letter = answer.upper()
            answer_idx = self.LETTER_INDICES.index(answer_letter)
        elif isinstance(answer, int) and 0 <= answer < len(options):
            answer_idx = answer
            answer_letter = self.LETTER_INDICES[answer_idx]
        else:
            return None

        # Bound the letter branch: an int answer is already checked against
        # len(options) above, but a letter can resolve to an index past the
        # (filtered) option list.
        if answer_idx >= len(options):
            return None

        return question, options, answer_letter, answer_idx

    def _normalize_row(self, example: dict[str, Any]) -> dict[str, Any]:
        """Normalize an example once and attach the result as extra columns.

        Stores the normalized fields on the row so the cheap validity filter
        and the mappers can reuse them instead of re-running
        ``_normalize_example``. Invalid rows carry sentinel values and a
        ``_valid`` flag of False; they are dropped by the pipeline filter.

        Args:
            example: A raw dataset row.

        Returns:
            The row dict augmented with the ``_NORMALIZED_COLUMNS`` keys.
        """
        normalized = self._normalize_example(example)
        if normalized is None:
            return {
                "_valid": False,
                "_question": "",
                # A list-of-string sentinel (not []) keeps the ``_options``
                # column's Arrow type consistent across map batches: an empty
                # list infers as ``list<null>``, which cannot later store real
                # option strings when the first batch of rows is all invalid.
                "_options": [""],
                "_answer_letter": "",
                "_answer_idx": -1,
            }

        question, options, answer_letter, answer_idx = normalized
        return {
            "_valid": True,
            "_question": question,
            "_options": options,
            "_answer_letter": answer_letter,
            "_answer_idx": answer_idx,
        }

    def _normalized(
        self, example: dict[str, Any]
    ) -> tuple[str, list[str], str, int] | None:
        """Return the normalized tuple for a row, reusing pre-attached fields.

        Rows that flowed through ``_normalize_row`` carry the ``_valid`` flag
        and the normalized fields as columns, so the mappers reuse them
        instead of normalizing a second time. Rows normalized as invalid
        (``_valid`` False) return ``None`` here, routing them through the
        mappers' placeholder path exactly as a raw row rejected by
        ``_normalize_example`` would. Direct calls with a raw row (e.g. in
        tests) fall back to ``_normalize_example`` so the cache path and the
        validation path cannot disagree about what constitutes a usable row.

        Args:
            example: A dataset row (raw or pre-normalized).

        Returns:
            A ``(question, options, answer_letter, answer_idx)`` tuple, or
            ``None`` if the example cannot be mapped.
        """
        if "_valid" in example:
            if not example["_valid"]:
                return None
            return (
                example["_question"],
                example["_options"],
                example["_answer_letter"],
                example["_answer_idx"],
            )
        return self._normalize_example(example)

    def _map_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to verifiers format."""
        normalized = self._normalized(example)
        if normalized is None:
            return {"question": "", "answer": None, "info": {}}

        question, options, answer_letter, answer_idx = normalized
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
        normalized = self._normalized(example)
        if normalized is None:
            return {
                "prompt": [],
                "ground_truth": None,
                "data_source": "mmlu_pro_health",
                "metadata": {},
            }

        question, options, answer_letter, answer_idx = normalized
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
        """Return dataset formatted for verifiers evaluation.

        Each row is normalized exactly once (``_normalize_row``); invalid rows
        are then dropped by the cheap ``_valid`` filter before mapping, so the
        mapper never emits placeholder rows that would otherwise shape the
        mapped dataset's inferred features.
        """
        return (
            self._dataset.map(self._normalize_row)
            .filter(lambda row: row["_valid"])
            .map(self._map_example, remove_columns=_NORMALIZED_COLUMNS)
        )

    def get_verl_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for Verl training.

        Invalid rows are filtered out before mapping (see
        ``get_verifiers_dataset``).
        """
        return (
            self._dataset.map(self._normalize_row)
            .filter(lambda row: row["_valid"])
            .map(self._map_example_verl, remove_columns=_NORMALIZED_COLUMNS)
        )
