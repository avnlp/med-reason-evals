# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Cursor, Codex, and
others) when working with code in this repository. It is the canonical copy;
`CLAUDE.md` points here.

## Project Overview

Medical Reasoning Evaluation framework (`med-reason-evals`) for benchmarking medical LLMs across 10 datasets using two complementary evaluation ecosystems: **Verifiers** (environment-based RL evaluation) and **Verl** (Groq rollouts + reward functions for post-hoc evaluation).

## Commands

### Setup
```bash
make sync                # Install all dependencies via uv
source .venv/bin/activate
```

### Testing
```bash
make test                # Run unit tests (excludes integration tests)
make test-cov            # Run tests with coverage
make test-ci             # What CI runs: coverage XML + JUnit report
uv run pytest tests/path/to/test_file.py                    # Run a single test file
uv run pytest tests/path/to/test_file.py::test_function     # Run a single test
uv run pytest tests/path/to/test_file.py -x                 # Stop on first failure
```

### Linting & Formatting
```bash
make lint-all            # Format + lint + type check (run before committing)
make lint-fmt            # Format and auto-fix with ruff
make lint-check          # Check formatting and lint without modifying
make lint-typing         # Type check with mypy
make lint-typos          # Spell check with typos
```

### Security
```bash
make security            # Run bandit + pip-audit
```

## Architecture

### Dual Evaluation Ecosystems

```
Datasets (data/)
    ├── get_verifiers_dataset() → Verifiers Evaluators (verifiers/)
    │       Schema: question, answer, info
    │       Streaming: False (needs len()/shuffle())
    │       Entry: evaluator.evaluate(client, model, num_examples)   # sync
    │
    └── get_verl_dataset() → Verl Evaluators (verl/)
            Schema: prompt, ground_truth, data_source, metadata
            Streaming: True
            Entry: await evaluator.evaluate(num_examples)            # async
```

### Dataset Layer (`src/med_reason_evals/data/`)

All datasets extend `BaseDataset` and expose `get_verifiers_dataset()` and `get_verl_dataset()`. Only datasets are exported from the top-level `__init__.py` — this flat API is intentional to prevent breakage when internal modules reorganize.

`BaseDataset` is not iterable and has no `__len__`. Access data through
`get_verifiers_dataset()` / `get_verl_dataset()`, which return HuggingFace
`Dataset` or `IterableDataset` objects.

### Verifiers Evaluators (`src/med_reason_evals/verifiers/`)

Three base classes in `base.py`:
- `BaseVerifierEvaluator` — abstract base with lazy environment construction
- `BaseMCQEvaluator` — multiple-choice with XML or BOXED answer formats
- `BaseJudgeEvaluator` — open-ended tasks scored by LLM-as-judge via `JudgeRubric`

Builder pattern: subclasses implement `_load_datasets()`, `_build_parser_and_prompt()`, `_build_rubric()`.

`evaluate()` here is **synchronous**. Passing `streaming=True` raises — verifiers
environments need `len()`/`shuffle()`.

### Verl Evaluators (`src/med_reason_evals/verl/`)

`BaseVerlEvaluator` uses `GroqRollouts` for generation. Judge clients and rollouts are lazily initialized. `evaluate()` here is **async** — await it or wrap in `asyncio.run()`.

### Shared Utilities (`src/med_reason_evals/utils/`)

- `extraction.py` — layered answer extraction: XML tags → boxed LaTeX → anchored phrases → last-token fallback
- `parsing.py` — JSON parsing, yes/no parsing, think-tag stripping
- `text.py` — Unicode normalization, space normalization
- `retry.py` — `wrap_openai_call()` async retry with tenacity; configured via `MED_REASON_RETRY_*` env vars

### Scoring Strategies

Five strategies, split across the two ecosystems:

| Strategy | Verifiers | Verl |
| :------- | :-------- | :--- |
| Multiple-choice accuracy | `rewards/multiple_choice_accuracy.py` | `rewards/multiple_choice_accuracy.py` |
| LLM-as-judge | `rewards/llm_as_judge.py` | `rewards/llm_as_judge.py` |
| Rubric multi-criteria | `rewards/judge_rubric.py` | `rewards/healthbench_rubric.py` |
| Hybrid routing | `rewards/hybrid_pubhealthbench.py` | `rewards/hybrid_pubhealthbench.py` |
| Semantic equivalence | — | `rewards/semantic_equivalence.py` |

## Code Standards

- **Python**: >=3.10, tested on 3.10–3.14 (local `.python-version` pins 3.12)
- **Formatter/Linter**: ruff (line-length 88, double quotes, Google-style docstrings)
- **Type checking**: mypy in strict mode (`allow_untyped_defs = false`) — all functions must have type annotations
- **Dependencies**: use libraries already declared in `pyproject.toml`; check it before adding a new dependency
- **Tests**: pytest with `--disable-socket` by default; use `@pytest.mark.integration` for tests needing external services and `@pytest.mark.enable_socket` to allow network access
- **Async tests**: `asyncio_mode = "auto"` (no need for `@pytest.mark.asyncio`)
- **Test parallelism**: `pytest-xdist` with `-n auto --dist loadscope` by default

## Environment Variables

```bash
GROQ_API_KEY=...                    # Required for evaluators (loaded from .env automatically)
MED_REASON_RETRY_MAX_ATTEMPTS=50    # Optional retry tuning
MED_REASON_RETRY_MAX_WAIT=600.0
MED_REASON_RETRY_LOG=true
```
