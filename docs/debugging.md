# Debugging and Observability Guide

This document defines the debugging suite for the Notion Agentic RAG system.

## Scope

The suite provides full runtime traceability for:

- End-to-end query runs from CLI and Streamlit.
- Per-agent node execution inside the LangGraph workflow.
- Tool discovery, agent selection, and tool invocation events.
- Final run output state and error paths.

The goal is transparent, reproducible debugging of what happened, when, and why.

## Output Location

All runtime debugging artifacts are written under `logs/`.

- Structured trace files: `logs/trace-<UTCSTAMP>-<RUN_ID>.jsonl`
- App log files:
  - `logs/main.log`
  - `logs/app.log`
  - `logs/rag.log` (fallback/default logger bootstrap name)

`logs/` is git-ignored except `logs/.gitkeep`.

## Trace Format

Each trace line is a JSON object with this envelope:

```json
{
  "ts": "2026-02-21T22:18:44.888057+00:00",
  "run_id": "<hex>",
  "sequence": 1,
  "event_type": "run_start",
  "payload": { "...": "..." }
}
```

### Event Types

- `run_start`
- `node_start`
- `node_end`
- `node_error`
- `tool_discovery`
- `tool_selection`
- `tool_invoke_start`
- `tool_invoke_end`
- `run_end`
- `run_exception`

## What Is Captured

### Run level

- Query, mode (`invoke` or `stream`), full initial state.
- Final state on success.
- Exception metadata and elapsed duration on failure.

### Node level

- State before node execution.
- Node output.
- State after merge.
- Computed state delta (`before` and `after` by field).
- Per-node execution duration in milliseconds.

### Tool level

- Discovery capability filter and returned tool cards.
- Selection score and chosen tool.
- Invocation task payload, timeout, outcome metadata, and duration.

## Configuration

Debug behavior is configured in `config/settings.py` via `Settings.debug`:

- `enabled` (`bool`, default `True`)
- `log_dir` (`str`, default `"./logs"`)
- `log_level` (`str`, default `"INFO"`)

### Runtime controls

- CLI `--verbose` sets debug log level to `DEBUG`.
- Streamlit `Verbose Logging` sets debug log level dynamically.

## Usage

### CLI run

```powershell
python main.py "What is retrieval-augmented generation?" --verbose
```

Then inspect `logs/trace-*.jsonl` and `logs/main.log`.

### Streamlit run

```powershell
streamlit run app.py
```

Submit a prompt and inspect `logs/trace-*.jsonl` and `logs/app.log`.

## Reading Traces Quickly

PowerShell examples:

```powershell
Get-Content .\logs\trace-*.jsonl | ConvertFrom-Json | Select-Object event_type, sequence
```

```powershell
Get-Content .\logs\trace-*.jsonl | ConvertFrom-Json | Where-Object { $_.event_type -eq "node_end" }
```

## Implementation Map

- `src/utils/debugging.py`: trace session lifecycle, serialization, logging bootstrap.
- `src/orchestrator/graph.py`: node instrumentation wrappers.
- `src/tools/client.py`: tool lifecycle trace events.
- `src/orchestrator/state.py`: canonical initial state builder.
- `main.py` and `app.py`: run-level tracing context integration.

## Limitations and Notes

- Traces are local files; no external trace store is required.
- Payloads are intentionally full for transparency; redact sensitive data before sharing logs externally.
- Logging is idempotently configured and reconfigured only when signature changes (`log_dir`, `log_level`, app name).
