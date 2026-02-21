# Agent Documentation Harness

## Purpose

- Give agents a consistent path to policy, technical context, skills, and validation.
- Keep operational rules, architecture knowledge, and task playbooks independent.
- Reduce drift between code changes and documentation updates.

## Layered Model

### Layer A: Repo Contract

- `AGENTS.md` (primary runtime contract)
- `CLAUDE.md` (cross-tool mirror)

Responsibilities:

- Allowed/disallowed actions
- Workflow and quality gate expectations
- Pointers to deeper technical and skill docs

### Layer B: Domain and System Context

- `docs/architecture.md`
- `docs/database.md`
- `docs/patterns.md`
- `docs/testing.md`
- `docs/commands.md`
- `docs/utils/*.md`

Responsibilities:

- Technical behavior and invariants
- Data model assumptions
- Operational commands and debugging routines

### Layer C: Task Skills

- Canonical root: `.codex/skills/<skill-name>/`
- Optional mirror root: `.agents/skills/<skill-name>/` (not enabled currently)

Skill package structure:

- `SKILL.md` (required)
- `references/` (optional)
- `scripts/` (optional)
- `assets/` (optional)
- `agents/` (optional)

Responsibilities:

- Task-specific execution workflow
- Input/output conventions
- Safety constraints and fallback paths

### Layer D: Tracking and Change History

- `TODO.md` (active backlog only)
- `docs/changelog.md` (notable behavior and architecture changes)
- `docs/archive/` (completed plans, legacy docs, historical logs)

## Agent Navigation Order

1. Read `AGENTS.md`.
2. Read relevant docs in `docs/`.
3. Select the matching skill in `.codex/skills/`.
4. Use skill `references/` and `scripts/` only as needed.
5. Run validation checks before finishing.

## Ownership and Source of Truth

- Policy: `AGENTS.md`
- Technical: `docs/`
- Skills: `.codex/skills/`
- Active work tracking: `TODO.md`
- Historical context: `docs/archive/`

If duplicate skill content exists across canonical and mirror roots, update canonical content first.

## Update Rules

- If `src/` behavior changes, update relevant `docs/` in the same iteration.
- If agent workflow changes, update `AGENTS.md` and `CLAUDE.md`.
- If a task repeats, codify it as a skill instead of repeating prompt instructions.
- Keep skills focused and composable.
