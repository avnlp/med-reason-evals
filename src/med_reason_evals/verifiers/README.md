# Verifiers-Based Evaluation

This module provides a Verifiers-based evaluation path built on the Verifiers framework, using single-turn environments, structured prompts, and rubric-driven scoring.

## What this module does

- **Builds evaluation environments** that pair dataset examples with scoring logic.
- **Configures answer formats** to align prompts and parsers for reliable extraction.
- **Scores responses** using accuracy, judge-based evaluation, or multi-criteria rubrics.

## Core flow

1. Create an evaluator for a dataset.
2. Load a random-access dataset shape.
3. Select answer format and prompt guidance.
4. Run evaluation with an OpenAI-compatible client.
5. Aggregate scores and metrics.

```python
from openai import AsyncOpenAI
from med_reason_evals.verifiers import MedQAEvaluator

client = AsyncOpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

evaluator = MedQAEvaluator(use_think=True, answer_format="xml")
results = evaluator.evaluate(client=client, model="openai/gpt-oss-120b", num_examples=100)
```

## Inputs and outputs

- **Inputs** include a question-and-answer dataset shape, model credentials, and scoring configuration.
- **Outputs** include aggregated scores and dataset metadata suitable for reporting.

## Key behaviors and edge handling

- **Answer formats** support XML and boxed responses, with optional reasoning segments.
- **Judging** uses a separate model to compare predictions to ground truth when exact matching is insufficient.
- **Rubric scoring** evaluates multiple criteria and normalizes scores by total points.
- **Random access** is required for shuffling and subset selection during evaluation.

## How it connects to other modules

- Consumes dataset adapters from [../data/README.md](../data/README.md).
- Uses prompt and parsing helpers from [utils/README.md](utils/README.md).
- Uses shared extraction and retry utilities from [../utils/README.md](../utils/README.md).

## Supported datasets

See [../../README.md](../../README.md) for the full dataset table and links.

## Directory map

```text
verifiers/
├── __init__.py     # Evaluator exports
├── base.py         # Shared evaluator behavior
├── utils/          # Prompt templates and parsing helpers
├── rewards/        # Scoring functions and rubrics
└── *.py            # Dataset-specific evaluators
```

## References

- Verifiers framework: https://github.com/PrimeIntellect-ai/verifiers
- Verifiers docs: https://docs.primeintellect.ai/verifiers/overview
