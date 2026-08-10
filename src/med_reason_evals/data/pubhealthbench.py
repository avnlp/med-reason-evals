"""Dataset adapter for PubHealthBench mixed-format public health questions.

PubHealthBench includes both multiple-choice and freeform answers, so this
adapter normalizes prompts and ground-truth fields for each question type.
"""

from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from med_reason_evals.data.base import BaseDataset


class PubHealthBenchDataset(BaseDataset):
    """PubHealthBench dataset for public health information evaluation.

    The adapter supports filtering by question type and maps MCQ answers to
    lettered choices while keeping freeform answers intact.
    """

    DATASET_PATH = "Joshua-Harris/PubHealthBench"

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
            question_type: Filter by type ("mcq", "freeform", "all").
            **kwargs: Additional arguments.
        """
        super().__init__(split=split, streaming=streaming, **kwargs)
        self.question_type = question_type.lower().strip()

        dataset = load_dataset(
            self.DATASET_PATH,
            split=split,
            streaming=streaming,
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
        letters = ["A", "B", "C", "D", "E", "F"][: len(options)]
        opts = "\n".join(f"{ltr}. {opt}" for ltr, opt in zip(letters, options))
        return f"Question: {question}\n\n{opts}\n\nAnswer:"

    def _build_freeform_prompt(self, question: str) -> str:
        """Build a freeform question prompt."""
        return f"Question: {question}\n\nAnswer:"

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
            letters = ["A", "B", "C", "D", "E", "F"][: len(options)]
            answer_letter = None
            answer_idx_field = example.get("answer_index")
            if isinstance(answer_idx_field, int) and 0 <= answer_idx_field < len(
                options
            ):
                answer_letter = letters[answer_idx_field]
            else:
                for i, opt in enumerate(options):
                    if opt.strip().lower() == answer.lower():
                        answer_letter = letters[i]
                        break
                if answer_letter is None and answer.upper() in letters:
                    answer_letter = answer.upper()

            if answer_letter is None:
                return {"question": "", "answer": None, "info": {}}

            answer_idx = letters.index(answer_letter)
            return {
                "question": self._build_mcq_prompt(question, options),
                "answer": answer_letter,
                "info": {
                    "answer_text": (
                        options[answer_idx] if answer_idx < len(options) else answer
                    ),
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
            letters = ["A", "B", "C", "D", "E", "F"][: len(options)]
            answer_letter = None
            answer_idx_field = example.get("answer_index")
            if isinstance(answer_idx_field, int) and 0 <= answer_idx_field < len(
                options
            ):
                answer_letter = letters[answer_idx_field]
            else:
                for i, opt in enumerate(options):
                    if opt.strip().lower() == answer.lower():
                        answer_letter = letters[i]
                        break
                if answer_letter is None and answer.upper() in letters:
                    answer_letter = answer.upper()

            if answer_letter is None:
                return {
                    "prompt": [],
                    "ground_truth": None,
                    "data_source": "pubhealthbench",
                    "metadata": {},
                }

            answer_idx = letters.index(answer_letter)
            prompt = self._build_mcq_prompt(question, options)
            ground_truth = {
                "answer": answer_letter,
                "answer_text": (
                    options[answer_idx] if answer_idx < len(options) else answer
                ),
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
