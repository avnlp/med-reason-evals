# Verl Reward Functions

This module provides scoring logic for the Verl evaluation path, including accuracy grading, judge-based evaluation, rubric scoring, semantic matching, and hybrid routing.

## What this module does

- **Scores multiple-choice answers** using extraction and normalization.
- **Judges open-ended responses** with model-based comparisons to ground truth.
- **Applies multi-criteria rubrics** and normalizes scores by total points.
- **Compares semantic equivalence** for free-form answers that have multiple valid phrasings.

## Core flow

1. Extract a candidate answer from the model output.
2. Select the scoring path based on dataset identity and question type.
3. Compute a normalized score.

```python
from med_reason_evals.verl.rewards import mcq_score

score = mcq_score(solution_text, ground_truth)
```

## Inputs and outputs

- **Inputs** include a solution text and a ground-truth object.
- **Outputs** are normalized scores in the range from 0.0 to 1.0.

## Key behaviors and edge handling

- **Extraction** uses structured tags, anchored phrases, and tail-window fallback.
- **Semantic matching** applies normalization and supports multiple valid answers.
- **Judge prompts** are template-driven to support task-specific grading.
- **Rubrics** evaluate criteria in parallel and aggregate with weights.
- **Hybrid routing** uses metadata to choose between multiple-choice and judge scoring.

## How it connects to other modules

- Used by the Verl evaluation path in [../README.md](../README.md).
- Uses parsing and normalization helpers from [../../utils/README.md](../../utils/README.md).

## Supported datasets

See [../../../README.md](../../../README.md) for the full dataset table and links.

## Directory map

```text
rewards/
├── __init__.py                 # Public exports
├── multiple_choice_accuracy.py # Multiple-choice grading
├── llm_as_judge.py             # Judge-based scoring
├── healthbench_rubric.py       # Multi-criteria rubric scoring
├── hybrid_pubhealthbench.py    # Mixed-format routing
└── semantic_equivalence.py     # Semantic matching
```

## References

- Verl evaluation: [../README.md](../README.md)
