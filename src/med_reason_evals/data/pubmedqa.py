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
            **kwargs: Additional arguments.
        """
        super().__init__(split=split, streaming=streaming, **kwargs)
        self._dataset = load_dataset(
            self.DATASET_PATH,
            split=split,
            streaming=streaming,
        )

    @property
    def num_options(self) -> int:
        """Return the number of answer options (3 for yes/no/maybe)."""
        return len(self.OPTIONS)

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
        """Return dataset formatted for verifiers evaluation."""
        mapped = self._dataset.map(self._map_example)
        return mapped.filter(lambda x: x is not None and x.get("answer") is not None)

    def get_verl_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for Verl training."""
        mapped = self._dataset.map(self._map_example_verl)
        return mapped.filter(
            lambda x: x is not None and x.get("ground_truth") is not None
        )
