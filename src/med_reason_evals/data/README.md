# Dataset Adapters

This module loads medical benchmarks from Hugging Face and reshapes them into two standardized dataset shapes, one for Verifiers evaluation and one for Verl evaluation.

## What this module does

- **Loads raw datasets** from Hugging Face with streaming or in-memory access.
- **Normalizes records** into two consistent shapes for the two evaluation paths.
- **Applies dataset-specific rules** for options, labels, and rubric metadata.

## Core flow

1. Load raw examples from the dataset source.
2. Map each example into the target schema.
3. Filter malformed rows.
4. Return a streaming or in-memory dataset depending on configuration.

```python
from med_reason_evals import MedQADataset

dataset = MedQADataset(split="test", streaming=False)
verifiers_dataset = dataset.get_verifiers_dataset()
verl_dataset = dataset.get_verl_dataset()
```

## Inputs and outputs

- **Inputs** include a dataset name, split, and a streaming flag.
- **Environment-based shape** uses a question string, an expected answer, and metadata for scoring.
- **Async shape** uses a message-based prompt, a ground-truth object, metadata, and a dataset identity used for reward routing.
- **Outputs** are dataset objects suitable for shuffling or streaming, depending on the evaluation path.

## Key behaviors and edge handling

- **Streaming defaults** favor random access for environment-based evaluation and streaming for async evaluation.
- **Option filtering** supports datasets with variable choice counts.
- **Label mapping** converts numeric or yes/no labels into canonical forms.
- **Unicode normalization** ensures consistent matching across scripts and casing.
- **Hybrid routing metadata** marks mixed-format datasets so scoring follows the right path.

## How it connects to other modules

- The Verifiers evaluation path consumes the question-and-answer shape in [verifiers/README.md](verifiers/README.md).
- The Verl evaluation path consumes the message-based shape in [verl/README.md](verl/README.md).
- Shared extraction and normalization helpers live in [../utils/README.md](../utils/README.md).

## Supported datasets

See [../../README.md](../../README.md) for the full dataset table and links.

## Directory map

```text
data/
├── __init__.py         # Adapter exports
├── base.py             # Shared adapter behavior
├── medqa.py            # MedQA mapping
├── med_mcqa.py         # MedMCQA mapping
├── pubmedqa.py         # PubMedQA mapping
├── medbullets.py       # MedBullets mapping
├── metamedqa.py        # MetaMedQA mapping
├── mmlu_pro_health.py  # MMLU-Pro Health mapping
├── medxpertqa.py       # MedXpertQA mapping
├── healthbench.py      # HealthBench mapping
├── medcasereasoning.py # MedCaseReasoning mapping
└── pubhealthbench.py   # PubHealthBench mapping
```

## References

- Package overview: [../README.md](../README.md)
- Verifiers evaluation: [../verifiers/README.md](../verifiers/README.md)
- Verl evaluation: [../verl/README.md](../verl/README.md)
