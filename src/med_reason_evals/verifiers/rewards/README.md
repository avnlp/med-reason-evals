# Verifiers Reward Functions

This module provides scoring logic for Verifiers-based evaluation, including accuracy grading, judge-based evaluation, rubric scoring, and hybrid routing.

## What this module does

- **Scores multiple-choice answers** using layered extraction and normalization.
- **Judges open-ended responses** using model-based comparisons to ground truth.
- **Applies multi-criteria rubrics** and normalizes scores by total points.
- **Routes mixed-format datasets** to the right scoring path.

## Core flow

1. Extract the candidate answer from the model output.
2. Select the scoring path based on task type.
3. Compute a score and return it to the evaluation harness.

```python
from med_reason_evals.verifiers.rewards import multiple_choice_accuracy

score = await multiple_choice_accuracy(completion, answer, info)
```

## Inputs and outputs

- **Inputs** include model output text, expected answers, and metadata.
- **Outputs** are normalized scores in the range from 0.0 to 1.0.

## Key behaviors and edge handling

- **Extraction** uses XML and boxed formats, anchored phrases, and a tail-window fallback.
- **Negation handling** reduces false positives in multiple-choice grading.
- **Judge prompts** are template-driven to support task-specific grading.
- **Rubrics** evaluate criteria in parallel and aggregate with weights.
- **Hybrid routing** uses metadata to choose between multiple-choice and judge scoring.

## How it connects to other modules

- Used by the Verifiers-based evaluation path in [../README.md](../README.md).
- Uses parsing and normalization helpers from [../../utils/README.md](../../utils/README.md).

## Supported datasets

See [../../data/README.md](../../data/README.md) for the available dataset adapters.

## Directory map

```text
rewards/
├── __init__.py                 # Public exports
├── multiple_choice_accuracy.py # Multiple-choice grading
├── llm_as_judge.py             # Judge-based scoring
├── judge_rubric.py             # Multi-criteria rubric scoring
└── hybrid_pubhealthbench.py    # Mixed-format routing
```

## References

- Verifiers-based evaluation: [../README.md](../README.md)
