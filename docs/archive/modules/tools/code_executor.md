# src/tools/code_executor.py

## Purpose
Sandboxed Python execution tool agent for small compute/code tasks.

## Main responsibilities
- Validate imports against allowlist + stdlib.
- Execute code in subprocess with timeout and restricted environment.
- Return stdout/stderr/return code in standardized result payload.

## Key symbol
- `CodeExecutorAgent`

## Safety controls
- Import pre-validation via AST.
- Allowed module set plus stdlib only.
- Timeout capped at 30 seconds.
- Output truncation limit.
