"""Load and process the MedQA dataset.

Dataset: HuggingFace `GBaker/MedQA-USMLE-4-options` dataset.
Each example is normalized to the following fields:
{
    "question": "<question + formatted options>",  # string used as the user prompt
    "answer":   "<A|B|C|D>",                       # top-level gold letter
    "info":     { ...original example fields... }  # full source row for debugging
}
"""

from typing import Any

from datasets import load_dataset


class MedQADataset:
    """Process the MedQA dataset."""

    def __init__(
        self,
        num_train_examples: int = -1,
        num_test_examples: int = -1,
    ):
        """Initialize the MedQA dataset processor.

        Args:
            num_train_examples: Number of training examples to use (-1 for all)
            num_test_examples: Number of test examples to use (-1 for all)
        """
        self.num_train_examples = num_train_examples
        self.num_test_examples = num_test_examples
        self.rng_seed = 12345

        # Load and process datasets on initialization
        self.train_ds, self.test_ds = self._load_and_process_datasets()

    def _load_and_process_datasets(self) -> tuple:
        """Load and process the MedQA datasets."""
        # Load the raw datasets
        ds = load_dataset("GBaker/MedQA-USMLE-4-options")
        train_raw = ds["train"]
        test_raw = ds["test"]

        # Limit number of examples if specified
        if self.num_train_examples != -1:
            train_raw = train_raw.select(
                range(min(self.num_train_examples, len(train_raw)))
            )
        if self.num_test_examples != -1:
            test_raw = test_raw.select(
                range(min(self.num_test_examples, len(test_raw)))
            )

        # Format datasets for verifiers
        train_formatted = self._format_for_verifiers(train_raw, "train")
        test_formatted = self._format_for_verifiers(test_raw, "test")

        # Shuffle datasets
        train_formatted = train_formatted.shuffle(seed=self.rng_seed)
        test_formatted = test_formatted.shuffle(seed=self.rng_seed)

        return train_formatted, test_formatted

    def _format_for_verifiers(self, dataset: Any, split: str) -> Any:
        """Format dataset for verifiers with question, answer, and info fields."""
        valid = {"A", "B", "C", "D"}

        def format_row(row: dict) -> dict:
            row = dict(row)

            # Build the user-visible question string (question + options)
            q = row.get("question", "") or ""
            opts = row.get("options", {}) or {}

            question_str = f"Question: {q}\n"
            for k, v in opts.items():
                # Skip null or empty values
                if v is not None and v != "":
                    question_str += f"\n{k}. {v}"

            # Lift the answer top-level, normalize to a single letter
            ans = (row.get("answer_idx") or "").strip().upper()
            if ans not in valid:
                # Final guard: set to empty if unexpected
                ans = ""

            # Keep full original example under 'info'
            info = dict(row)

            return {
                "question": question_str,
                "answer": ans,
                "info": info,
            }

        return dataset.map(format_row)
