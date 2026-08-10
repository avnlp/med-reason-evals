"""Dataset adapter for HealthBench rubric-based evaluation prompts.

HealthBench uses multi-criterion rubrics instead of single answers, so this
adapter surfaces rubric metadata alongside prompts for judge-based scoring.
"""

from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from med_reason_evals.data.base import BaseDataset


class HealthBenchDataset(BaseDataset):
    """HealthBench dataset for rubric-based health response evaluation.

    Each prompt includes a rubric with scored criteria, which evaluators use to
    judge response quality rather than perform exact answer matching.
    """

    DATASET_MAPPING = {
        "regular": "neuralleap/healthbench-regular",
        "consensus": "neuralleap/healthbench-consensus",
        "hard": "neuralleap/healthbench-hard",
    }

    def __init__(
        self,
        split: str | None = None,
        streaming: bool = True,
        difficulty: str = "regular",
        **kwargs: Any,
    ) -> None:
        """Initialize the HealthBench dataset adapter.

        Args:
            split: Dataset split to load. Defaults to ``"test"`` for
                ``difficulty="regular"`` and ``"train"`` otherwise, which are
                the partitions each difficulty variant ships with. An explicit
                value is honored as-is.
            streaming: Whether to stream the dataset.
            difficulty: Dataset difficulty (``"regular"``, ``"consensus"``,
                ``"hard"``). Unsupported values raise ``ValueError`` so a typo
                fails fast instead of silently producing an empty evaluation
                set.
            **kwargs: Additional arguments forwarded to ``load_dataset``
                (e.g. ``revision``, ``cache_dir``).

        Raises:
            ValueError: If ``difficulty`` is not one of the allowed values.
        """
        if difficulty not in self.DATASET_MAPPING:
            raise ValueError(f"Invalid difficulty: {difficulty}")
        self.difficulty = difficulty

        # The regular variant exposes a test split, while consensus and hard
        # only ship a train split. Derive the default partition from the
        # difficulty, but honor an explicitly requested split so ``self.split``
        # always reflects the partition that was actually loaded.
        actual_split = split or ("test" if difficulty == "regular" else "train")
        super().__init__(split=actual_split, streaming=streaming, **kwargs)
        self._dataset = load_dataset(
            self.DATASET_MAPPING[difficulty],
            split=actual_split,
            streaming=streaming,
            **kwargs,
        )

    @property
    def num_options(self) -> int:
        """Return the number of MCQ options.

        HealthBench is a rubric-based open-ended dataset, so there are no
        fixed multiple-choice options to report.
        """
        return 1

    @staticmethod
    def _is_valid_example(example: dict[str, Any]) -> bool:
        """Check whether a raw HealthBench example is usable for evaluation.

        A row is usable when it has a non-empty prompt and at least one rubric
        criterion to score against.

        Args:
            example: A raw dataset row.

        Returns:
            True if the example is well-formed and usable for evaluation.
        """
        prompt = example.get("prompt", "")
        if not prompt:
            return False

        rubrics = example.get("rubrics", []) or []
        return any(r.get("criterion") for r in rubrics if isinstance(r, dict))

    def _extract_rubric_info(self, example: dict[str, Any]) -> dict[str, Any]:
        """Extract rubric criteria and points from example.

        The rubric tags encode axes such as safety or completeness, which are
        stored alongside the criteria for downstream reporting.
        """
        rubrics = example.get("rubrics", []) or []

        criteria = []
        points_list = []
        axes = []

        for rubric in rubrics:
            criterion = rubric.get("criterion", "")
            points = rubric.get("points", 0)

            if criterion:
                criteria.append(criterion)
                points_list.append(points)

                # Extract the axis tag, if provided, for richer rubric metadata.
                tags = rubric.get("tags", []) or []
                axis = ""
                for tag in tags:
                    if tag.startswith("axis:"):
                        axis = tag.split(":", 1)[1]
                        break
                axes.append(axis)

        return {
            "criteria": criteria,
            "points_list": points_list,
            "axes": axes,
        }

    def _extract_prompt(self, example: dict[str, Any]) -> str:
        """Extract the prompt from the example.

        Prompts can be provided as a multi-turn message list; for evaluation we
        flatten the content into a single text prompt.
        """
        prompt = example.get("prompt", "")
        if isinstance(prompt, list):
            # Multi-turn format: retain only the textual content in order.
            texts = []
            for msg in prompt:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if content:
                        texts.append(content)
            return "\n\n".join(texts)
        return prompt

    def _map_example(self, example: dict[str, Any]) -> dict[str, Any] | None:
        """Map a raw example to verifiers format.

        Returns None for malformed rows so streaming evaluation can skip them.
        """
        prompt = self._extract_prompt(example)

        if not prompt:
            return None

        rubric_info = self._extract_rubric_info(example)

        return {
            "question": prompt,
            "answer": "",
            "info": {
                "prompt_id": example.get("prompt_id", ""),
                "criteria": rubric_info["criteria"],
                "points_list": rubric_info["points_list"],
                "axes": rubric_info["axes"],
                "ideal_completions": example.get("ideal_completions_data", []),
            },
        }

    def _map_example_verl(self, example: dict[str, Any]) -> dict[str, Any] | None:
        """Map a raw example to Verl format.

        The prompt is preserved as role-tagged messages when available to
        support chat-style reward model training.
        """
        prompt_content = self._extract_prompt(example)

        if not prompt_content:
            return None

        rubric_info = self._extract_rubric_info(example)

        # Build messages while preserving any multi-turn structure.
        raw_prompt = example.get("prompt", "")
        if isinstance(raw_prompt, list):
            messages = []
            for msg in raw_prompt:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if content:
                        messages.append({"role": role, "content": content})
        else:
            messages = [{"role": "user", "content": prompt_content}]

        return {
            "prompt": messages,
            "ground_truth": {
                "criteria": rubric_info["criteria"],
                "points_list": rubric_info["points_list"],
                "axes": rubric_info["axes"],
            },
            "data_source": "healthbench",
            "metadata": {
                "prompt_id": example.get("prompt_id", ""),
                "difficulty": self.difficulty,
                "ideal_completions": example.get("ideal_completions_data", []),
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
