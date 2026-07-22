# Shared Utilities

This module provides normalization, extraction, parsing, and retry helpers used by both evaluation paths.

## What this module does

- **Normalizes text** for reliable matching across scripts and casing.
- **Extracts answers** from structured and free-form model outputs.
- **Parses outputs** into structured forms such as JSON and yes/no verdicts.
- **Retries API calls** with exponential backoff for OpenAI-compatible clients.

## Core flow

1. Normalize raw model outputs.
2. Extract the most likely answer span.
3. Parse structured fragments or judge verdicts.
4. Retry failed API calls under controlled backoff.

```python
from med_reason_evals.utils import normalize_answer, extract_answer

normalized = normalize_answer(response_text, mode="basic")
answer = extract_answer(response_text)
```

## Inputs and outputs

- **Inputs** are raw model outputs or API callables.
- **Outputs** include normalized strings, extracted answers, parsed objects, and wrapped call results.

## Key behaviors and edge handling

- **Extraction order** tries structured tags first, then anchored phrases, then a tail-window fallback.
- **Tail-window focus** limits extraction to the last portion of the response to avoid reasoning noise.
- **JSON recovery** handles fenced or partial model outputs.
- **Judge verdict parsing** supports both plain text and JSON verdict formats.
- **Retry policy** handles rate limits and transient errors with configurable limits.

## How it connects to other modules

- Used by dataset adapters for normalization and parsing.
- Used by both evaluation paths for extraction and retry logic.

## Supported datasets

See [../../README.md](../../README.md) for the full dataset table and links.

## Directory map

```text
utils/
├── __init__.py   # Public exports
├── text.py       # Unicode normalization and comparison modes
├── extraction.py # Answer extraction heuristics
├── parsing.py    # JSON recovery and verdict parsing
└── retry.py      # Exponential backoff for API calls
```

## References

- Package overview: [../README.md](../README.md)
