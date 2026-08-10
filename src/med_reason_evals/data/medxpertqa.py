"""Dataset adapter for MedXpertQA multiple-choice medical reasoning tasks.

The adapter loads the MedXpertQA split, optionally filters by question type,
and produces both Verifiers and Verl projections with consistently formatted
multiple-choice prompts.
"""

from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from med_reason_evals.data.base import BaseDataset


class MedXpertQADataset(BaseDataset):
    """MedXpertQA dataset for expert-level medical QA.

    The dataset distinguishes between reasoning-heavy and understanding-focused
    questions, which the adapter can filter at initialization time.
    """

    DATASET_PATH = "TsinghuaC3I/MedXpertQA"
    DATASET_CONFIG = "Text"

    NUM_OPTIONS = 4

    def __init__(
        self,
        split: str = "test",
        streaming: bool = True,
        question_type: str = "all",
        **kwargs: Any,
    ) -> None:
        """Initialize the MedXpertQA dataset adapter.

        Args:
            split: Dataset split to use.
            streaming: Whether to stream the dataset.
            question_type: Filter by type ("reasoning", "understanding", "all").
            **kwargs: Additional arguments.
        """
        super().__init__(split=split, streaming=streaming, **kwargs)
        self.question_type = question_type.lower().strip()

        dataset = load_dataset(
            self.DATASET_PATH,
            name=self.DATASET_CONFIG,
            split=split,
            streaming=streaming,
        )

        # Pre-filter by question type to avoid mapping large unused splits.
        if self.question_type != "all":
            dataset = dataset.filter(
                lambda x: (
                    (x.get("question_type") or "").lower().strip() == self.question_type
                )
            )

        self._dataset = dataset

    @property
    def num_options(self) -> int:
        """Return the number of MCQ options (4 for MedXpertQA)."""
        return self.NUM_OPTIONS

    def _format_question_with_options(
        self, question: str, options: dict[str, str]
    ) -> str:
        """Format question with answer choices.

        MedXpertQA sometimes embeds the choices in the question stem, so we
        strip any embedded choice block and rebuild a consistent prompt.
        """
        if not options:
            return question

        # Render options in a compact inline layout to match dataset examples.
        formatted_options = " ".join(
            f"({letter}) {text}" for letter, text in sorted(options.items())
        )

        # Extract the stem when the dataset already injected choices.
        if "Answer Choices:" in question:
            stem, _, _ = question.partition("Answer Choices:")
            stem = stem.strip()
        else:
            stem = question.strip()

        return f"{stem}\nAnswer Choices: {formatted_options}"

    def _map_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to verifiers format.

        Returns a dict with ``answer=None`` for invalid examples so the caller
        can filter them without losing streaming semantics.
        """
        question = (example.get("question") or "").strip()
        raw_options = example.get("options")
        # Cast labels to strings because some splits encode them numerically.
        label = str(example.get("label", "")).strip().upper()

        # Options arrive as a mapping of choice letters to text.
        if not isinstance(raw_options, dict):
            return {"question": "", "answer": None, "info": {}}
        options = dict(raw_options)

        if not question or not options or not label:
            return {"question": "", "answer": None, "info": {}}

        # Guard against mislabeled rows to keep evaluation stable.
        if label not in options:
            return {"question": "", "answer": None, "info": {}}

        answer_text = options.get(label, "")

        return {
            "question": self._format_question_with_options(question, options),
            "answer": label,
            "info": {
                "answer_text": answer_text,
                "question_type": example.get("question_type", ""),
            },
        }

    def _map_example_verl(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to Verl format.

        Returns a dict with ``ground_truth=None`` for invalid examples so the
        dataset stream can be filtered lazily.
        """
        question = (example.get("question") or "").strip()
        raw_options = example.get("options")
        # Cast labels to strings because some splits encode them numerically.
        label = str(example.get("label", "")).strip().upper()

        # Options arrive as a mapping of choice letters to text.
        if not isinstance(raw_options, dict):
            return {"prompt": [], "ground_truth": None, "data_source": "medxpertqa"}
        options = dict(raw_options)

        if not question or not options or not label:
            return {"prompt": [], "ground_truth": None, "data_source": "medxpertqa"}

        if label not in options:
            return {"prompt": [], "ground_truth": None, "data_source": "medxpertqa"}

        answer_text = options.get(label, "")
        prompt = self._format_question_with_options(question, options)

        return {
            "prompt": [{"role": "user", "content": prompt}],
            "ground_truth": {
                "answer": label,
                "answer_text": answer_text,
            },
            "data_source": "medxpertqa",
            "metadata": {
                "question_type": example.get("question_type", ""),
                "medical_task": example.get("medical_task", ""),
                "body_system": example.get("body_system", ""),
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
