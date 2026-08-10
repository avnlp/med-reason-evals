"""Dataset adapter for PubHealthBench mixed-format public health questions.

PubHealthBench includes both multiple-choice and freeform answers, so this
adapter normalizes prompts and ground-truth fields for each question type.
"""

import string
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from med_reason_evals.data.base import BaseDataset


class PubHealthBenchDataset(BaseDataset):
    """PubHealthBench dataset for public health information evaluation.

    The adapter supports filtering by question type and maps MCQ answers to
    lettered choices while keeping freeform answers intact.
    """

    DATASET_PATH = "Joshua-Harris/PubHealthBench"
    #: Letter labels for MCQ options (A-Z, enough for any realistic option list).
    LETTERS = string.ascii_uppercase
    #: Supported question_type filter values.
    QUESTION_TYPES = ("mcq", "freeform", "all")

    def __init__(
        self,
        split: str = "test",
        streaming: bool = True,
        question_type: str = "all",
        **kwargs: Any,
    ) -> None:
        """Initialize the PubHealthBench dataset adapter.

        Args:
            split: Dataset split to use.
            streaming: Whether to stream the dataset.
            question_type: Filter by type (``"mcq"``, ``"freeform"``, ``"all"``).
            **kwargs: Additional arguments forwarded to ``load_dataset()``
                (e.g. ``revision``, ``cache_dir``).

        Raises:
            ValueError: If ``question_type`` is not one of ``QUESTION_TYPES``.
        """
        super().__init__(split=split, streaming=streaming, **kwargs)
        self.question_type = (question_type or "").lower().strip()
        if self.question_type not in self.QUESTION_TYPES:
            raise ValueError(
                f"Invalid question_type {question_type!r}; expected one of "
                f"{', '.join(self.QUESTION_TYPES)}"
            )

        dataset = load_dataset(
            self.DATASET_PATH,
            split=split,
            streaming=streaming,
            **kwargs,
        )

        # Pre-filter by question type to avoid mapping large unused splits.
        if self.question_type == "mcq":
            dataset = dataset.filter(self._is_mcq)
        elif self.question_type == "freeform":
            dataset = dataset.filter(lambda x: not self._is_mcq(x))

        self._dataset = dataset

    @property
    def num_options(self) -> int:
        """Return the number of MCQ options.

        PubHealthBench mixes multiple-choice and freeform questions with
        per-item option counts, so there is no single fixed option count.
        Question-type routing is handled per item via the ``is_mcq`` flag.
        """
        return 1

    def _is_mcq(self, example: dict[str, Any]) -> bool:
        """Check if the example is a multiple-choice question.

        PubHealthBench encodes MCQs with an options array of length > 1.
        """
        options = example.get("options", [])
        return bool(options and len(options) > 1)

    def _build_mcq_prompt(self, question: str, options: list[str]) -> str:
        """Build a formatted MCQ prompt with lettered choices."""
        letters = self.LETTERS[: len(options)]
        opts = "\n".join(f"{ltr}. {opt}" for ltr, opt in zip(letters, options))
        return f"Question: {question}\n\n{opts}\n\nAnswer:"

    def _build_freeform_prompt(self, question: str) -> str:
        """Build a freeform question prompt."""
        return f"Question: {question}\n\nAnswer:"

    def _resolve_mcq_answer(self, example: dict[str, Any]) -> tuple[str, str] | None:
        """Resolve an MCQ example's answer to ``(letter, answer_text)``.

        The answer may be encoded as an ``answer_index``, as the option text
        itself, or as a single letter.  Only indices within the letter table
        are accepted, so oversized option lists degrade to ``None`` (filtered
        by the pipeline) instead of raising ``IndexError``.

        Returns:
            A ``(answer_letter, answer_text)`` tuple, or ``None`` if the
            answer cannot be mapped to a valid letter.
        """
        options = example.get("options", []) or []
        answer = (example.get("answer") or "").strip()
        letters = self.LETTERS[: len(options)]

        answer_letter: str | None = None
        answer_idx_field = example.get("answer_index")
        if isinstance(answer_idx_field, int) and 0 <= answer_idx_field < len(letters):
            answer_letter = letters[answer_idx_field]
        else:
            for i, opt in enumerate(options):
                if i >= len(letters):
                    break
                if opt.strip().lower() == answer.lower():
                    answer_letter = letters[i]
                    break
            if answer_letter is None and answer.upper() in letters:
                answer_letter = answer.upper()

        if answer_letter is None:
            return None

        answer_idx = letters.index(answer_letter)
        return answer_letter, options[answer_idx]

    def _map_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to verifiers format.

        Returns a dict with ``answer=None`` for invalid examples so streaming
        datasets can be filtered lazily.
        """
        question = (example.get("question") or "").strip()
        answer = (example.get("answer") or "").strip()

        if not question or not answer:
            return {"question": "", "answer": None, "info": {}}

        is_mcq = self._is_mcq(example)
        options = example.get("options", []) or []

        if is_mcq:
            resolved = self._resolve_mcq_answer(example)
            if resolved is None:
                return {"question": "", "answer": None, "info": {}}
            answer_letter, answer_text = resolved
            return {
                "question": self._build_mcq_prompt(question, options),
                "answer": answer_letter,
                "info": {
                    "answer_text": answer_text,
                    "is_mcq": True,
                },
            }
        return {
            "question": self._build_freeform_prompt(question),
            "answer": answer,
            "info": {
                "is_mcq": False,
            },
        }

    def _map_example_verl(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to Verl format.

        Returns a dict with ``ground_truth=None`` for invalid examples so the
        dataset stream can be filtered lazily.
        """
        question = (example.get("question") or "").strip()
        answer = (example.get("answer") or "").strip()

        if not question or not answer:
            return {
                "prompt": [],
                "ground_truth": None,
                "data_source": "pubhealthbench",
                "metadata": {},
            }

        is_mcq = self._is_mcq(example)
        options = example.get("options", []) or []

        if is_mcq:
            resolved = self._resolve_mcq_answer(example)
            if resolved is None:
                return {
                    "prompt": [],
                    "ground_truth": None,
                    "data_source": "pubhealthbench",
                    "metadata": {},
                }
            answer_letter, answer_text = resolved
            prompt = self._build_mcq_prompt(question, options)
            ground_truth = {
                "answer": answer_letter,
                "answer_text": answer_text,
            }
        else:
            prompt = self._build_freeform_prompt(question)
            ground_truth = {
                "answer": answer,
                "target": answer,
            }

        return {
            "prompt": [{"role": "user", "content": prompt}],
            "ground_truth": ground_truth,
            "data_source": "pubhealthbench",
            "metadata": {
                "is_mcq": is_mcq,
                "source": example.get("source", ""),
            },
        }

    def get_verifiers_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for verifiers evaluation."""
        mapped = self._dataset.map(self._map_example)
        return mapped.filter(lambda x: x.get("answer") is not None)

    def get_verl_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for Verl training."""
        mapped = self._dataset.map(self._map_example_verl)
        return mapped.filter(lambda x: x.get("ground_truth") is not None)
