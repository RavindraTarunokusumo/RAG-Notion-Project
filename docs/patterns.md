# Engineering Patterns and Decisions

## Core Principles

1. Mechanical enforcement is preferred over repeated manual review comments.
2. Prefer simple, composable building blocks that agents can reason about.
3. Validate boundaries explicitly through contracts, types, or guards.
4. Keep changes legible through scoped diffs, tests, and explicit decisions.
5. If a rule repeats in review, graduate it into automation (lint/check/CI).

## Architectural Patterns

- Linear orchestration over ad-hoc agent chaining.
- Shared typed state contract between nodes.
- Explicit retrieval and synthesis separation.
- Optional tool-agent extension via a registry and agent-card capability model.

## Code Patterns

- Configuration through `config/settings.py` rather than hardcoded constants.
- Utility modules for cross-cutting concerns (`tracing`, `session_manager`, helpers).
- Fail-safe fallbacks where external APIs or optional features are unavailable.

## Documentation Patterns

- Policy rules belong in `AGENTS.md` and `CLAUDE.md`.
- Technical truth belongs in `docs/`.
- Repeated workflow instructions belong in `.codex/skills/`.
- Historical docs belong in `docs/archive/`.

## Change Hygiene

- Code changes should include test and docs updates where behavior changes.
- Keep pull requests/task diffs scoped to a single intent.
- Prefer explicit assumptions over implicit behavior.
