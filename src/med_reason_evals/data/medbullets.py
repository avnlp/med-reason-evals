"""Dataset adapter for MedBullets USMLE-style multiple-choice questions.

MedBullets ships the same questions in a four-option (A-D) and a five-option
(A-E) variant, exposed as the ``op4_test`` and ``op5_test`` splits. The adapter
makes that choice explicit through ``num_options`` and formats prompts with the
lettered layout shared by the other MCQ adapters.
"""

from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from med_reason_evals.data.base import BaseDataset


class MedBulletsDataset(BaseDataset):
    """MedBullets dataset for medical board exam preparation.

    The upstream ``options`` struct always carries keys A-E; on the four-option
    split the ``E`` entry is empty, so it is dropped before prompts are built.
    """

    DATASET_PATH = "mkieffer/MedBullets"
    #: Option counts with a corresponding upstream split.
    VALID_NUM_OPTIONS = (4, 5)

    def __init__(
        self,
        streaming: bool = True,
        num_options: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize the MedBullets dataset adapter.

        The split is derived from ``num_options`` (``op4_test`` or
        ``op5_test``), since the option count *is* the split for this dataset.

        Args:
            streaming: Whether to stream the dataset.
            num_options: Number of options per question (4 or 5).
            **kwargs: Additional keyword arguments forwarded to
                ``load_dataset()`` (e.g. ``revision``, ``cache_dir``).

        Raises:
            ValueError: If ``num_options`` is not 4 or 5.
        """
        if num_options not in self.VALID_NUM_OPTIONS:
            raise ValueError("num_options must be 4 or 5")

        super().__init__(split=f"op{num_options}_test", streaming=streaming, **kwargs)
        self._num_options = num_options
        self._dataset = load_dataset(
            self.DATASET_PATH,
            split=self.split,
            streaming=streaming,
            **kwargs,
        )

    @property
    def num_options(self) -> int:
        """Return the number of MCQ options (4 or 5, chosen at construction)."""
        return self._num_options

    def _usable_options(self, example: dict[str, Any]) -> dict[str, str]:
        """Return the option map actually offered to the model.

        Drops the ``E`` entry on the four-option variant and any option left
        empty upstream, so the rendered prompt never shows a blank choice.
        """
        options = example.get("options") or {}
        if not isinstance(options, dict):
            return {}
        if self._num_options == 4:
            options = {k: v for k, v in options.items() if k != "E"}
        return {k: v for k, v in options.items() if v is not None and v != ""}

    def _is_valid_example(self, example: dict[str, Any]) -> bool:
        """Check whether a raw MedBullets example is usable for evaluation.

        Requires the option count to match ``num_options`` and the gold letter
        to name one of the remaining options. This rejects four-option rows
        whose answer is ``E``, which would otherwise yield ground truth with no
        matching choice in the prompt.

        Args:
            example: A raw dataset row.

        Returns:
            True if the example is well-formed and usable for evaluation.
        """
        question = example.get("question", "")
        if not isinstance(question, str) or not question.strip():
            return False

        options = self._usable_options(example)
        if len(options) != self._num_options:
            return False

        answer = example.get("answer", "")
        return isinstance(answer, str) and answer.strip().upper() in options

    def _build_prompt(self, question: str, options: dict[str, str]) -> str:
        """Build a formatted prompt from question and options.

        The prompt uses the same lettered option format as the other MCQ
        adapters so answer extraction stays consistent across datasets.
        """
        opts = "\n".join(f"{k}. {v}" for k, v in options.items())
        return f"Question: {question}\n\n{opts}\n\nAnswer:"

    def _map_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to verifiers format."""
        question = example["question"].strip()
        options = self._usable_options(example)
        answer_letter = example["answer"].strip().upper()

        return {
            "question": self._build_prompt(question, options),
            "answer": answer_letter,
            "info": {
                "answer_text": options[answer_letter],
                "original_question": question,
            },
        }

    def _map_example_verl(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to Verl format."""
        question = example["question"].strip()
        options = self._usable_options(example)
        answer_letter = example["answer"].strip().upper()

        return {
            "prompt": [
                {"role": "user", "content": self._build_prompt(question, options)}
            ],
            "ground_truth": {
                "answer": answer_letter,
                "answer_text": options[answer_letter],
            },
            "data_source": "medbullets",
            "metadata": {
                "original_question": question,
                "num_options": self._num_options,
            },
        }

    def get_verifiers_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for verifiers evaluation.

        Invalid rows are filtered before mapping so the mapper can assume
        well-formed input and the lazy streaming path never sees a bad row.
        """
        return self._dataset.filter(self._is_valid_example).map(self._map_example)

    def get_verl_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for Verl training.

        Invalid rows are filtered before mapping (see ``get_verifiers_dataset``
        for why this matters under streaming).
        """
        return self._dataset.filter(self._is_valid_example).map(self._map_example_verl)
