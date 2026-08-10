"""Dataset adapter for the MedCaseReasoning clinical diagnosis benchmark.

The dataset wraps narrative case presentations and asks models to predict a
final diagnosis. Prompts embed explicit XML tags to align downstream judge
parsing with the evaluation rubric.
"""

from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from med_reason_evals.data.base import BaseDataset


class MedCaseReasoningDataset(BaseDataset):
    """MedCaseReasoning dataset for medical diagnosis prediction.

    Each example is converted into a structured prompt that guides the model to
    emit its reasoning and final diagnosis in separate XML tags.
    """

    DATASET_PATH = "zou-lab/MedCaseReasoning"

    QUESTION_TEMPLATE = """\
----------------------------------------
CASE PRESENTATION
----------------------------------------
{case_prompt}
----------------------------------------
OUTPUT TEMPLATE
----------------------------------------
<think>
...your internal reasoning for the diagnosis...
</think>
<answer>
...the name of the disease/entity...
</answer>"""

    def __init__(
        self,
        split: str = "val",
        streaming: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the MedCaseReasoning dataset adapter.

        Args:
            split: Dataset split to use ("train" or "val").
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
        """Return the number of MCQ options.

        MedCaseReasoning is a free-form diagnosis task, so there are no
        predefined options.

        Returns:
            Always 1 for open-ended datasets.
        """
        return 1

    def _build_prompt(self, case_prompt: str) -> str:
        """Build a formatted prompt from the case presentation.

        The template deliberately includes explicit XML tags to ensure the
        judge pipeline can extract the final diagnosis reliably.
        """
        return self.QUESTION_TEMPLATE.format(case_prompt=case_prompt)

    def _map_example(self, example: dict[str, Any]) -> dict[str, Any] | None:
        """Map a raw example to verifiers format.

        Returns None for malformed rows to keep streaming datasets lazy.
        """
        case_prompt = (example.get("case_prompt") or "").strip()
        final_diagnosis = (example.get("final_diagnosis") or "").strip()

        if not case_prompt or not final_diagnosis:
            return None

        return {
            "question": self._build_prompt(case_prompt),
            "answer": final_diagnosis,
            "info": {
                "case_prompt": case_prompt,
            },
        }

    def _map_example_verl(self, example: dict[str, Any]) -> dict[str, Any] | None:
        """Map a raw example to Verl format.

        The ground truth includes both ``answer`` and ``target`` so reward
        functions can choose between exact matching and semantic scoring.
        """
        case_prompt = (example.get("case_prompt") or "").strip()
        final_diagnosis = (example.get("final_diagnosis") or "").strip()

        if not case_prompt or not final_diagnosis:
            return None

        prompt = self._build_prompt(case_prompt)

        return {
            "prompt": [{"role": "user", "content": prompt}],
            "ground_truth": {
                "answer": final_diagnosis,
                "target": final_diagnosis,
            },
            "data_source": "medcasereasoning",
            "metadata": {
                "case_prompt": case_prompt,
                "differential_diagnosis": example.get("differential_diagnosis", []),
            },
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
