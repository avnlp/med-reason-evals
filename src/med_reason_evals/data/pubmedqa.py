"""Dataset adapter for the PubMedQA yes/no/maybe benchmark.

Uses the canonical 500-example human-annotated test split from
``openlifescienceai/pubmedqa``, which pre-splits the labeled set so results
are directly comparable to the published PubMedQA leaderboard.
"""

from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from med_reason_evals.data.base import BaseDataset


class PubMedQADataset(BaseDataset):
    """PubMedQA dataset for biomedical research question answering."""

    DATASET_PATH = "openlifescienceai/pubmedqa"
    OPTIONS = {"A": "Yes", "B": "No", "C": "Maybe"}

    PROMPT_TEMPLATE = """Select the best answer.

Context: {context}

Question: {question}

{options_block}

Answer:"""

    def __init__(
        self,
        split: str = "test",
        streaming: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the PubMedQA dataset adapter.

        Args:
            split: Dataset split to use (``"test"`` for the canonical
                500-example evaluation split).
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
        """Return the number of answer options (3 for yes/no/maybe)."""
        return len(self.OPTIONS)

    @staticmethod
    def _is_valid_example(example: dict[str, Any]) -> bool:
        """Check whether a raw example has a valid question and answer letter.

        Args:
            example: A raw dataset row.

        Returns:
            True if the example is usable for evaluation.
        """
        data = example.get("data", {}) or {}
        question = (data.get("Question") or "").strip()
        answer_letter = (data.get("Correct Option") or "").strip()
        return bool(question) and answer_letter in PubMedQADataset.OPTIONS

    def _build_prompt(self, question: str, context: str) -> str:
        """Build a formatted prompt with a fixed A/B/C answer block."""
        options_block = "\n".join(f"{k}. {v}" for k, v in self.OPTIONS.items())
        return self.PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            options_block=options_block,
        )

    def _map_example(self, example: dict[str, Any]) -> dict[str, Any] | None:
        """Map a raw example to verifiers format.

        Returns None for rows missing a valid question or answer letter.
        """
        data = example.get("data", {}) or {}
        question = (data.get("Question") or "").strip()
        answer_letter = (data.get("Correct Option") or "").strip()
        context_list = data.get("Context", []) or []

        if not question or answer_letter not in self.OPTIONS:
            return None

        context = "\n".join(context_list)
        return {
            "question": self._build_prompt(question, context),
            "answer": answer_letter,
            "info": {
                "answer_text": self.OPTIONS[answer_letter],
            },
        }

    def _map_example_verl(self, example: dict[str, Any]) -> dict[str, Any] | None:
        """Map a raw example to Verl format.

        Returns None for rows missing a valid question or answer letter.
        """
        data = example.get("data", {}) or {}
        question = (data.get("Question") or "").strip()
        answer_letter = (data.get("Correct Option") or "").strip()
        context_list = data.get("Context", []) or []

        if not question or answer_letter not in self.OPTIONS:
            return None

        context = "\n".join(context_list)
        prompt = self._build_prompt(question, context)
        return {
            "prompt": [{"role": "user", "content": prompt}],
            "ground_truth": {
                "answer": answer_letter,
                "answer_text": self.OPTIONS[answer_letter],
            },
            "data_source": "pubmedqa",
            "metadata": {},
        }

    def get_verifiers_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for verifiers evaluation.

        Invalid rows are filtered out before mapping, so the mapper never
        sees (or returns ``None`` for) malformed examples. This is required
        for the lazy ``IterableDataset`` streaming path, where a ``None``
        mapping result raises ``TypeError`` instead of being dropped.
        """
        return self._dataset.filter(self._is_valid_example).map(self._map_example)

    def get_verl_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for Verl training.

        Invalid rows are filtered out before mapping (see
        ``get_verifiers_dataset`` for why this matters under streaming).
        """
        return self._dataset.filter(self._is_valid_example).map(self._map_example_verl)
