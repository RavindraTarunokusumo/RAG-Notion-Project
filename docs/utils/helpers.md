# Utils: helpers

Source module: `src/utils/helpers.py`

## Purpose

Shared helper utilities used across ingestion, orchestration, and agent runtime.

## Expectations

- Helpers should remain side-effect minimal unless explicitly documented.
- Shared helper behavior should be deterministic where possible.
- Any helper behavior change that impacts runtime output should include test coverage.
