# Agent Runtime Contract

This file is the primary contract for agent behavior in this repository.

## Purpose

- Provide one reliable execution path for agents.
- Keep policy, technical context, and task playbooks separated.
- Enforce consistent quality gates before task completion.

## Required Navigation Order

1. Read `AGENTS.md` for hard constraints.
2. Read `docs/agent-harness.md` for documentation topology.
3. Read relevant technical docs in `docs/`.
4. Select the matching skill in `.codex/skills/`.
5. Execute work and validate with required checks.

## Allowed Actions

- Read and modify repository files required for the task.
- Create or update docs in `docs/` when behavior changes.
- Add or update skills in `.codex/skills/` for repeated workflows.
- Run non-destructive local commands for validation and debugging.

## Disallowed Actions

- Do not push secrets, API keys, or credentials.
- Do not use destructive Git commands unless explicitly requested.
- Do not bypass quality gates when code changes are made.
- Do not treat archived docs as active source of truth.

## Workflow Rules

### Branch and Worktree

- Use the active branch/worktree unless instructed otherwise.
- Keep changes scoped and atomic.
- Do not revert unrelated user changes.

### Source of Truth

- Policy source of truth: `AGENTS.md`
- Technical source of truth: `docs/`
- Skill source of truth: `.codex/skills/`
- Active backlog source of truth: `TODO.md`
- Historical context: `docs/archive/`

### Documentation Updates

- If `src/` behavior changes, update relevant docs in the same iteration.
- If execution workflow changes, update `AGENTS.md` and `CLAUDE.md`.
- If a task pattern repeats, add or revise a skill under `.codex/skills/`.

## Quality Gates

Run these before finishing any code-changing task:

```bash
uv run ruff check .
uv run pytest
```

If a gate fails due to environment constraints, report the exact failure and what remains unverified.

## Testing Placement

- Place tests under `tests/`.
- Prefer targeted unit tests for behavior changes.
- Keep end-to-end tests explicit about external dependencies.

## Compatibility

- `CLAUDE.md` mirrors this contract for cross-tool compatibility.
- Canonical skills live in `.codex/skills/`.
- Optional mirror root `.agents/skills/` is not enabled in this repo right now.
