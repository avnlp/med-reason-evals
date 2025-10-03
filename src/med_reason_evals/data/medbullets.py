"""Load and process the Medbullets dataset.

Dataset: HuggingFace `mkieffer/Medbullets` dataset.
Each example is normalized to the fields expected by `vf.Verifiers`:
{
    "question": "<stem + formatted options>",      # string used as the user prompt
    "answer":   "<A|B|C|D|E>",                     # top-level gold letter
    "info":     { ...original example fields... }  # full source row for debugging
}

- num_options=4 : loads splits `op4_train` / `op4_eval` and drops option "E"
- num_options=5 : loads splits `op5_train` / `op5_eval`
"""

from typing import Any

from datasets import load_dataset


class MedBulletsDataset:
    """Process the MedBullets dataset."""

    def __init__(
        self,
        num_train_examples: int = -1,
        num_eval_examples: int = -1,
        num_options: int = 4,
    ):
        """Initialize the MedBullets dataset processor.

        Args:
            num_train_examples: Number of training examples to use (-1 for all)
            num_eval_examples: Number of evaluation examples to use (-1 for all)
            num_options: Number of options per question (4 or 5)
        """
        if num_options not in [4, 5]:
            raise ValueError("'num_options' must be 4 or 5")

        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples
        self.num_options = num_options
        self.rng_seed = 12345

        # Load and process datasets on initialization
        self.train_ds, self.eval_ds = self._load_and_process_datasets()

    def _load_and_process_datasets(self) -> tuple:
        """Load and process the MedBullets datasets."""
        # Load the raw datasets based on number of options
        if self.num_options == 4:
            train_raw, eval_raw = load_dataset(
                "mkieffer/Medbullets", split=["op4_train", "op4_eval"]
            )
            # Remove option E from 4-option datasets
            train_raw = self._remove_option_e(train_raw)
            eval_raw = self._remove_option_e(eval_raw)
        else:  # num_options == 5
            train_raw, eval_raw = load_dataset(
                "mkieffer/Medbullets", split=["op5_train", "op5_eval"]
            )

        # Limit number of examples if specified
        if self.num_train_examples != -1:
            train_raw = train_raw.select(
                range(min(self.num_train_examples, len(train_raw)))
            )
        if self.num_eval_examples != -1:
            eval_raw = eval_raw.select(
                range(min(self.num_eval_examples, len(eval_raw)))
            )

        # Format datasets for verifiers
        train_formatted = self._format_for_verifiers(train_raw, "train")
        eval_formatted = self._format_for_verifiers(eval_raw, "eval")

        # Shuffle datasets
        train_formatted = train_formatted.shuffle(seed=self.rng_seed)
        eval_formatted = eval_formatted.shuffle(seed=self.rng_seed)

        return train_formatted, eval_formatted

    def _remove_option_e(self, dataset: Any) -> Any:
        """Remove option E from the dataset."""

        def remove_e(ex: dict) -> dict:
            ex = dict(ex)
            ex["options"] = {k: v for k, v in ex["options"].items() if k != "E"}
            return ex

        return dataset.map(remove_e)

    def _format_for_verifiers(self, dataset: Any, split: str) -> Any:
        """Format dataset for verifiers with question, answer, and info fields."""
        valid = {"A", "B", "C", "D", "E"}

        def format_row(row: dict) -> dict:
            row = dict(row)

            # Build the user-visible question string (stem + options)
            q = row.get("question", "") or ""
            opts = row.get("options", {}) or {}

            question_str = f"Question: {q}\n"
            for k, v in opts.items():
                # Skip null values of v (for the combined dataset where E
                # opt for 4op is null)
                if v is not None and v != "":
                    question_str += f"\n{k}: {v}"

            # Lift the answer top-level, normalize to a single letter
            ans = (row.get("answer") or "").strip().upper()
            if ans not in valid:
                # If op4 split sometimes stores 'E' or empty, coerce safely
                if ans == "" and "answer_letter" in row:
                    ans = str(row["answer_letter"]).strip().upper()
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
