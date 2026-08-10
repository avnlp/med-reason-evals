"""Dataset adapter for MedXpertQA multiple-choice medical reasoning tasks.

The adapter loads the MedXpertQA ``Text`` split (a fixed ten-choice A-J
schema), optionally filters by question type, and produces both Verifiers
and Verl projections with consistently formatted multiple-choice prompts.
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

    NUM_OPTIONS = 10

    ALLOWED_QUESTION_TYPES = ("all", "reasoning", "understanding")

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
            question_type: Filter by type (``"reasoning"``, ``"understanding"``,
                ``"all"``). Case-insensitive. Unsupported values raise
                ``ValueError`` so a typo fails fast instead of silently
                producing an empty evaluation set.
            **kwargs: Additional keyword arguments forwarded to
                ``load_dataset()`` (e.g. ``revision``, ``cache_dir``).

        Raises:
            ValueError: If ``question_type`` is not one of the allowed values.
        """
        super().__init__(split=split, streaming=streaming, **kwargs)
        self.question_type = question_type.lower().strip()
        if self.question_type not in self.ALLOWED_QUESTION_TYPES:
            raise ValueError(
                f"Unsupported question_type {question_type!r}. "
                f"Expected one of {self.ALLOWED_QUESTION_TYPES}."
            )

        dataset = load_dataset(
            self.DATASET_PATH,
            name=self.DATASET_CONFIG,
            split=split,
            streaming=streaming,
            **kwargs,
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
        """Return the number of MCQ options (10 for MedXpertQA).

        MedXpertQA Text questions always present ten choices (A-J), enforced
        by ``_is_valid_example``, so the count reported here always matches
        the rows that are actually evaluated.
        """
        return self.NUM_OPTIONS

    @staticmethod
    def _is_valid_example(example: dict[str, Any]) -> bool:
        """Check whether a raw MedXpertQA example is usable for evaluation.

        The Text config schema fixes ``options`` to exactly ``NUM_OPTIONS``
        (A-J) entries, so rows with any other option count are rejected rather
        than evaluated against a reported ``num_options`` that would not match
        the prompt.

        Args:
            example: A raw dataset row.

        Returns:
            True if the example is well-formed and usable for evaluation.
        """
        question = example.get("question", "")
        if not isinstance(question, str) or not question.strip():
            return False

        options = example.get("options")
        if (
            not isinstance(options, dict)
            or len(options) != MedXpertQADataset.NUM_OPTIONS
        ):
            return False

        label = str(example.get("label", "")).strip().upper()
        return bool(label) and label in options

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

        Assumes the example has already passed ``_is_valid_example``.
        """
        question = example["question"].strip()
        options = dict(example["options"])
        label = str(example["label"]).strip().upper()

        return {
            "question": self._format_question_with_options(question, options),
            "answer": label,
            "info": {
                "answer_text": options.get(label, ""),
                "question_type": example.get("question_type", ""),
            },
        }

    def _map_example_verl(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to Verl format.

        Assumes the example has already passed ``_is_valid_example``.
        """
        question = example["question"].strip()
        options = dict(example["options"])
        label = str(example["label"]).strip().upper()

        prompt = self._format_question_with_options(question, options)

        return {
            "prompt": [{"role": "user", "content": prompt}],
            "ground_truth": {
                "answer": label,
                "answer_text": options.get(label, ""),
            },
            "data_source": "medxpertqa",
            "metadata": {
                "question_type": example.get("question_type", ""),
                "medical_task": example.get("medical_task", ""),
                "body_system": example.get("body_system", ""),
            },
        }

    def get_verifiers_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for verifiers evaluation.

        Invalid rows are filtered out before mapping so the mapper can assume
        well-formed input and the lazy ``IterableDataset`` streaming path
        never sees a malformed row.
        """
        return self._dataset.filter(self._is_valid_example).map(self._map_example)

    def get_verl_dataset(self) -> Dataset | IterableDataset:
        """Return dataset formatted for Verl training.

        Invalid rows are filtered out before mapping (see
        ``get_verifiers_dataset`` for why this matters under streaming).
        """
        return self._dataset.filter(self._is_valid_example).map(self._map_example_verl)
