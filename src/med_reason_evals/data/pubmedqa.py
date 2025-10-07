"""Load and process the PubMedQA dataset.

Dataset: HuggingFace `qiaojin/PubMedQA` dataset.
Each example is normalized to the following fields:
{
    "question": "<formatted question with context>",  # complete prompt with abstract
    "answer":   "<A|B|C>",                           # A=yes, B=no, C=maybe
    "info":     { ...original example fields... }    # full source row for debugging
}
"""

import json
import os
from typing import Any

from datasets import load_dataset


class PubMedQADataset:
    """Process the PubMedQA dataset."""

    def __init__(
        self,
        num_train_examples: int = -1,
        num_test_examples: int = -1,
    ):
        """Initialize the PubMedQA dataset processor.

        Args:
            num_train_examples: Number of training examples to use (-1 for all)
            num_test_examples: Number of test examples to use (-1 for all)
        """
        self.num_train_examples = num_train_examples
        self.num_test_examples = num_test_examples
        self.rng_seed = 12345
        self.dataset_path = "qiaojin/PubMedQA"

        # Load and process datasets on initialization
        self.train_ds, self.test_ds = self._load_and_process_datasets()

    def _load_and_process_datasets(self) -> tuple:
        """Load and process the PubMedQA datasets."""
        # Load the raw datasets
        # pqa_artificial is the training set, pqa_labeled is the test set
        train_raw = load_dataset(
            self.dataset_path, name="pqa_artificial", split="train"
        )
        test_raw = load_dataset(self.dataset_path, name="pqa_labeled", split="train")

        # Filter test set to only include human-annotated samples
        test_raw = self._filter_test_set(test_raw)

        # Limit number of examples if specified
        if self.num_train_examples != -1:
            train_raw = train_raw.select(
                range(min(self.num_train_examples, len(train_raw)))
            )
        if self.num_test_examples != -1:
            test_raw = test_raw.select(
                range(min(self.num_test_examples, len(test_raw)))
            )

        # Format datasets
        train_formatted = self._format_dataset(train_raw, "train")
        test_formatted = self._format_dataset(test_raw, "test")

        # Shuffle datasets
        train_formatted = train_formatted.shuffle(seed=self.rng_seed)
        test_formatted = test_formatted.shuffle(seed=self.rng_seed)

        return train_formatted, test_formatted

    def _filter_test_set(self, dataset: Any) -> Any:
        """Filter test set to only include human-annotated samples (500 from 1000)."""
        # Load the predefined test IDs
        here = os.path.dirname(__file__)
        file_path = os.path.join(here, "data", "test_ground_truth.json")

        try:
            with open(file_path) as f:
                test_ids = json.load(f)

            # Filter to only the 500 human-annotated samples
            return dataset.filter(lambda sample: str(sample["pubid"]) in test_ids)
        except FileNotFoundError:
            # If the file doesn't exist, return the full test set
            print(f"Warning: {file_path} not found. Using full test set.")
            return dataset

    def _format_dataset(self, dataset: Any, split: str) -> Any:
        """Format dataset with question, answer, and info fields."""
        choices_map = {"yes": "A", "no": "B", "maybe": "C"}
        prompt_template = "Answer A for yes, B for no or C for maybe.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"

        def format_row(row: dict) -> dict:
            row = dict(row)

            # Extract question
            question_text = row.get("question", "") or ""

            # Extract and format context
            context_dict = row.get("context", {}) or {}
            labels = context_dict.get("labels", []) or []
            contexts = context_dict.get("contexts", []) or []

            # Format contexts with their labels
            formatted_contexts = []
            for label, context in zip(labels, contexts):
                formatted_contexts.append(f"{label}. {context}")
            context_text = "\n".join(formatted_contexts)

            # Build complete prompt
            complete_prompt = prompt_template.format(
                context=context_text, question=question_text
            )

            # Map final decision to letter (A/B/C)
            final_decision = (row.get("final_decision", "") or "").lower()
            answer = choices_map.get(final_decision, "")

            # Keep full original example under 'info'
            info = dict(row)

            return {
                "question": complete_prompt,
                "answer": answer,
                "info": info,
            }

        return dataset.map(format_row, load_from_cache_file=False)
