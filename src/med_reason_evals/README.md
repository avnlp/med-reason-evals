# Medical Reasoning Evaluation Package

This package exposes dataset adapters and connects them to two evaluation paths, Verifiers and Verl, while keeping shared utilities for extraction, parsing, and retry behavior in a single place.

## Core flow

1. Select a dataset adapter.
2. Produce the dataset shape needed by the chosen evaluation path.
3. Run evaluation and aggregate results.

```python
import os
from openai import AsyncOpenAI
from med_reason_evals import MedQADataset
from med_reason_evals.verifiers import MedQAEvaluator

client = AsyncOpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

dataset = MedQADataset(split="test", streaming=False)
verifiers_dataset = dataset.get_verifiers_dataset()

evaluator = MedQAEvaluator(use_think=True, answer_format="xml")
results = evaluator.evaluate(
    client=client,
    model="openai/gpt-oss-120b",
    num_examples=100,
)
```

## Inputs and outputs

- **Inputs** include dataset configuration, evaluation settings, and model credentials.
- **Verifiers** use a question-and-answer dataset shape with metadata for scoring.
- **Verl** uses a message-based prompt shape with ground truth, metadata, and dataset identity for reward routing.
- **Outputs** are aggregated scores per dataset with counts and averages suitable for reporting.

## Key behaviors and edge handling

- **Streaming vs random access** is handled by dataset adapters to support both evaluation paths.
- **Answer extraction** uses layered heuristics to handle structured and free-form outputs.
- **Judging and rubric scoring** use separate model calls for open-ended tasks and multi-criteria grading.
- **Retry logic** wraps OpenAI-compatible API calls with exponential backoff and configurable limits.

## Architecture

- Dataset adapters and schema details live in [data/README.md](data/README.md).
- The Verifiers evaluation path is documented in [verifiers/README.md](verifiers/README.md).
- The Verl evaluation path is documented in [verl/README.md](verl/README.md).
- Shared parsing, extraction, and retry utilities are documented in [utils/README.md](utils/README.md).

## Package Structure

```text
med_reason_evals/
├── __init__.py      # Flat exports and environment loading
├── data/            # Dataset adapters and schema mapping
├── utils/           # Shared parsing, extraction, and retry helpers
├── verifiers/       # Verifiers evaluation harness
└── verl/            # Verl evaluation pipeline
```

## References

- Dataset adapters: [data/README.md](data/README.md)
- Verifiers: [verifiers/README.md](verifiers/README.md)
- Verl: [verl/README.md](verl/README.md)
