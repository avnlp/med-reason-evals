# Verifiers Utilities

This module defines standardized prompts and parsing helpers so answer format instructions and extraction remain paired and consistent.

## What this module does

- **Provides prompt templates** for structured answer formats.
- **Pairs formats with parsers** to avoid mismatched extraction behavior.
- **Re-exports shared parsing helpers** for use in Verifiers-based evaluation.

## Core flow

1. Select an answer format.
2. Retrieve the matching system prompt template.
3. Parse model output with the matching extractor.

```python
from med_reason_evals.verifiers.utils import get_system_prompt

system_prompt = get_system_prompt(answer_format="xml", use_think=True)
```

## Inputs and outputs

- **Inputs** include answer format selection and optional reasoning guidance.
- **Outputs** include prompt text and parsing helpers that match the format.

## Key behaviors and edge handling

- **XML and boxed formats** are supported for structured answer extraction.
- **JSON format** is reserved for structured outputs but does not ship with a default prompt template.
- **Format pairing** ensures the prompt and parser stay consistent across evaluators.

## How it connects to other modules

- Used by the Verifiers-based evaluation path in [../README.md](../README.md).
- Uses shared extraction helpers from [../../utils/README.md](../../utils/README.md).

## Supported datasets

See [../../../README.md](../../../README.md) for the full dataset table and links.

## Directory map

```text
utils/
├── __init__.py   # Public exports
├── prompts.py    # Prompt templates and format pairing
└── parsers.py    # Re-exported parsing helpers
```

## References

- Verifiers-based evaluation: [../README.md](../README.md)
