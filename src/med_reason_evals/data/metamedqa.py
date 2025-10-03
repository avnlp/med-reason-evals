"""Load and process the MetaMedQA dataset.

Dataset: HuggingFace `maximegmd/MetaMedQA` dataset.
Each example is normalized to the fields expected by `vf.Verifiers`:
{
    "question": "<formatted question + options>",      # string used as the user prompt
    "answer":   "<A|B|C|D|E>",                        # top-level gold letter
    "info":     { ...original example fields... }      # full source row for debugging
}
"""

from typing import Any

from datasets import load_dataset


class MetaMedQADataset:
    """Process the MetaMedQA dataset."""

    def __init__(
        self,
        split: str = "test",
        num_examples: int = -1,
    ):
        """Initialize the MetaMedQA dataset processor.

        Args:
            split: Dataset split to use (train, validation, test)
            num_examples: Number of examples to use (-1 for all)
        """
        self.split = split
        self.num_examples = num_examples
        self.rng_seed = 12345

        # Load and process datasets on initialization
        self.dataset = self._load_and_process_dataset()

    def _load_and_process_dataset(self) -> Any:
        """Load and process the MetaMedQA dataset."""
        # Load the raw dataset
        raw_ds = load_dataset("maximegmd/MetaMedQA", split=self.split)

        # Limit number of examples if specified
        if self.num_examples != -1:
            raw_ds = raw_ds.select(range(min(self.num_examples, len(raw_ds))))

        # Format dataset for verifiers
        formatted_ds = self._format_for_verifiers(raw_ds)

        # Shuffle dataset
        return formatted_ds.shuffle(seed=self.rng_seed)

    def _build_prompt(self, question: str, options: dict) -> str:
        """Build prompt with question and options."""
        opts = "\n".join(f"{k}. {v}" for k, v in options.items())
        letters = ", ".join(sorted(options.keys()))
        return (
            "You are a clinician. Choose exactly ONE option letter.\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{opts}\n\n"
            f"Answer with ONLY the letter ({letters})."
        )

    def _format_for_verifiers(self, dataset: Any) -> Any:
        """Format dataset for verifiers with question, answer, and info fields."""
        valid = {"A", "B", "C", "D", "E"}

        def format_row(row: dict) -> dict:
            row = dict(row)

            q: str = row["question"]
            options: dict = row["options"]
            gold_text: str = row["answer"]

            # Find the gold letter by matching the answer text with options
            gold_letter = None
            for k, v in options.items():
                if (v or "").strip().lower() == (gold_text or "").strip().lower():
                    gold_letter = k
                    break

            # If we can't find a matching letter, return None to filter out
            if gold_letter is None or gold_letter not in valid:
                # Default to first option if no match found
                gold_letter = next(iter(options.keys()))

            # Build the user-visible question string (question + options)
            question_str = self._build_prompt(q, options)

            # Keep full original example under 'info'
            info = dict(row)

            return {
                "question": question_str,
                "answer": gold_letter,
                "info": info,
            }

        return dataset.map(format_row)
