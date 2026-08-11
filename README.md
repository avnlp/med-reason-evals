<h1 align="center">Medical Reasoning Evaluation</h1>

<div align="center">

[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/avnlp/med-reason-evals)
[![CI](https://img.shields.io/github/actions/workflow/status/avnlp/med-reason-evals/ci.yml?branch=main&label=CI&logo=githubactions)](https://github.com/avnlp/med-reason-evals/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/github/actions/workflow/status/avnlp/med-reason-evals/ci.yml?branch=main&label=Ruff&logo=ruff)](https://github.com/avnlp/med-reason-evals/actions/workflows/ci.yml)
[![MyPy](https://img.shields.io/github/actions/workflow/status/avnlp/med-reason-evals/ci.yml?branch=main&label=MyPy&logo=python)](https://github.com/avnlp/med-reason-evals/actions/workflows/ci.yml)
[![Bandit](https://img.shields.io/github/actions/workflow/status/avnlp/med-reason-evals/ci.yml?branch=main&label=Bandit&logo=owasp)](https://github.com/avnlp/med-reason-evals/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/avnlp/med-reason-evals/ci.yml?branch=main&label=Tests&logo=pytest)](https://github.com/avnlp/med-reason-evals/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/github/avnlp/med-reason-evals/graph/badge.svg)](https://codecov.io/github/avnlp/med-reason-evals)
[![License](https://img.shields.io/github/license/avnlp/med-reason-evals?color=green)](https://github.com/avnlp/med-reason-evals/blob/main/LICENSE)

</div>

Medical Reasoning Evaluation is a medical framework for evaluating language models across 10 standardized medical datasets and scoring strategies, with the [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) and [Verl](https://github.com/verl-project/verl) RL frameworks. The framework provides standardized dataset adapters, answer extraction heuristics, and multiple scoring strategies to measure medical reasoning capabilities.

# Features

- **10 medical datasets** - MedQA, MedMCQA, PubMedQA, MedBullets, MetaMedQA, MMLU-Pro Health, MedXpertQA, HealthBench, MedCaseReasoning, PubHealthBench.
- **Dual evaluation frameworks** - Verifiers (environment-based) and Verl (async with Groq rollouts).
- **Five scoring strategies** - Multiple-choice Accuracy, LLM-as-judge, Rubric-based Multi-criteria, Semantic Equivalence, and Hybrid Routing.
- **Multi-strategy answer extraction** - XML tags, boxed LaTeX, anchored phrases, last-token fallback with negation detection.
- **Resilient API integration** - Exponential backoff retry logic with configurable limits for rate-limited APIs.
- **Streaming and in-memory modes** - Stream large datasets lazily or load them fully for random access.

## Evaluation Methods

- **Multiple-Choice Accuracy**: Letter/option extraction with normalization and negation-aware matching.
- **LLM-as-Judge**: Model-based correctness checks for open-ended responses where exact matching is insufficient.
- **Rubric-based Multi-Criteria Scoring**: Criterion-level judging with normalized totals for structured evaluation.
- **Semantic Equivalence**: Normalized matching utilities for free-form answers with valid phrasing variation.
- **Hybrid Routing**: Per-example routing between MCQ scoring and judge-based scoring for mixed-task
  datasets.

## Datasets

| Dataset | HuggingFace | Task Type | Evaluation Method | Why this strategy |
| :------ | :---------- | :-------- | :---------------- | :---------------- |
| MedQA | [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | MCQ (A-D) | MCQ accuracy | Canonical single-answer MCQ benchmark |
| MedMCQA | [lighteval/med_mcqa](https://huggingface.co/datasets/lighteval/med_mcqa) | MCQ (A-D) | MCQ accuracy | Structured options with deterministic key |
| PubMedQA | [openlifescienceai/pubmedqa](https://huggingface.co/datasets/openlifescienceai/pubmedqa) | MCQ (Yes/No/Maybe) | MCQ accuracy | Canonical 500-example test split with fixed labels |
| MedBullets | [mkieffer/MedBullets](https://huggingface.co/datasets/mkieffer/MedBullets) | MCQ (A-D or A-E) | MCQ accuracy | Option-based exam format |
| MetaMedQA | [maximegmd/MetaMedQA](https://huggingface.co/datasets/maximegmd/MetaMedQA) | MCQ (A-E) | MCQ accuracy | Option-based QA with deterministic answer |
| MMLU-Pro Health | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | MCQ (A-J) | MCQ accuracy | Broad option space, still closed-form grading |
| MedXpertQA | [TsinghuaC3I/MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | MCQ (expert-level) | MCQ accuracy | Expert MCQ format with fixed answer key |
| HealthBench | [neuralleap/healthbench-*](https://huggingface.co/neuralleap) | Rubric-based | Multi-criteria rubric scoring | Requires criterion-level quality judgment |
| MedCaseReasoning | [zou-lab/MedCaseReasoning](https://huggingface.co/datasets/zou-lab/MedCaseReasoning) | Open-ended diagnosis | LLM-as-judge | Open-form diagnoses need semantic judging |
| PubHealthBench | [Joshua-Harris/PubHealthBench](https://huggingface.co/datasets/Joshua-Harris/PubHealthBench) | Mixed (MCQ + freeform) | Hybrid (MCQ + judge) | Mixed examples require dynamic scoring route |

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management. First, ensure uv is installed:

```bash
# Install uv (if not already installed)
pip install uv
```

Then install the project dependencies:

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Configuration

Create a `.env` file in the repository root:

```env
GROQ_API_KEY=your-groq-key
```

## Quick Start

### 1) Load a dataset adapter

```python
from med_reason_evals import MedQADataset

dataset = MedQADataset(split="test", streaming=False)
verifiers_dataset = dataset.get_verifiers_dataset()
verl_dataset = dataset.get_verl_dataset()
```

### 2) Run evaluation with Verifiers

```python
import os
from openai import AsyncOpenAI
from med_reason_evals.verifiers import MedQAEvaluator

client = AsyncOpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

evaluator = MedQAEvaluator(use_think=True, answer_format="xml")
results = evaluator.evaluate(
    client=client,
    model="openai/gpt-oss-120b",
    num_examples=100,
)
```

### 3) Run evaluation with Verl

```python
import asyncio
from med_reason_evals.verl import MedQAEvaluator

evaluator = MedQAEvaluator(split="test")
results = asyncio.run(evaluator.evaluate(num_examples=100))
```

## Documentation

- **Package overview**: [`src/med_reason_evals/`](src/med_reason_evals/)
- **Dataset adapters**: [`src/med_reason_evals/data/`](src/med_reason_evals/data/)
- **Shared utilities**: [`src/med_reason_evals/utils/`](src/med_reason_evals/utils/)
- **Verifiers-based evaluation**: [`src/med_reason_evals/verifiers/`](src/med_reason_evals/verifiers/)
- **Verifiers utilities**: [`src/med_reason_evals/verifiers/utils/`](src/med_reason_evals/verifiers/utils/)
- **Verifiers rewards**: [`src/med_reason_evals/verifiers/rewards/`](src/med_reason_evals/verifiers/rewards/)
- **Verl-based evaluation**: [`src/med_reason_evals/verl/`](src/med_reason_evals/verl/)
- **Verl rewards**: [`src/med_reason_evals/verl/rewards/`](src/med_reason_evals/verl/rewards/)

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
