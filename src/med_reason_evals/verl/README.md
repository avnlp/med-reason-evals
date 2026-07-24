# Verl Evaluation Pipeline

This module provides an async evaluation path that streams datasets, generates responses through a Groq-backed client, and computes scores using reward functions.

## What this module does

- **Streams datasets** to keep evaluation memory efficient.
- **Generates responses** with an OpenAI-compatible Groq endpoint.
- **Scores outputs** using rule-based, judge-based, or rubric-based reward logic.

## Core flow

1. Stream dataset examples from the adapter.
2. Generate a response for each prompt.
3. Score the response using the dataset identity and ground truth.
4. Aggregate scores and report summary metrics.

```python
import asyncio
from med_reason_evals.verl import MedQAEvaluator

evaluator = MedQAEvaluator(split="test")
results = asyncio.run(evaluator.evaluate(num_examples=100))
```

## Inputs and outputs

- **Inputs** include a message-based prompt shape, ground-truth objects, metadata, and dataset identity.
- **Outputs** include aggregated scores and example counts.

## Key behaviors and edge handling

- **Reward routing** uses dataset identity to select the correct scoring function.
- **Judging** uses a separate model call when exact matching is insufficient.
- **Rubric scoring** evaluates criteria independently and normalizes scores.
- **Streaming evaluation** avoids loading entire datasets in memory.

## How it connects to other modules

- Consumes dataset adapters from [../data/README.md](../data/README.md).
- Uses shared extraction and retry utilities from [../utils/README.md](../utils/README.md).
- Uses reward functions documented in [rewards/README.md](rewards/README.md).

## Supported datasets

See [../../README.md](../../README.md) for the full dataset table and links.

## Directory map

```text
verl/
├── __init__.py   # Evaluator exports
├── base.py       # Shared evaluator behavior
├── rollouts.py   # Groq-backed generation helpers
├── rewards/      # Scoring functions
└── *.py          # Dataset-specific evaluators
```

## References

- Verl framework: https://github.com/verl-project/verl
- Verl docs: https://verl.readthedocs.io/en/latest/
