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

- **DO NOT** push secrets, API keys, or credentials.
- **DO NOT** use destructive Git commands unless explicitly requested.
- **DO NOT** bypass quality gates when code changes are made.
- **DO NOT** treat archived docs as active source of truth.

## Workflow Rules

### Source of Truth

- Policy source of truth: `AGENTS.md`
- Technical source of truth: `docs/`
- Module-level source of truth: `docs/modules`
- Skill source of truth: `.codex/skills/`
- Active backlog source of truth: `TODO.md`
- Historical context: `docs/archive/`

### Branch and Worktree

**Worktrees**: Use the active branch/worktree unless instructed otherwise.
- Create with: `git worktree add .worktrees/<branch-name> <branch-name>`
- Merge to `main` when iteration complete; remove worktree after merge

### Commits
**Atomic Commits**: Each TODO sub-item = one commit. 
- Message format: `type(parent-section): Sub-item title` with description body.
- Do not revert unrelated user changes.

**Attach Task Summary with Git Notes**: After every commit, attach a structured note to it:
```bash
# 1. Get the just-completed commit hash
git log -1 --format="%H"

# 2. Attach a note summarising the task
git notes add -m "<task name>
Summary: <what changed and why>
Files: <list of created/modified files>" <commit_hash>
```
The note should include: task name, summary of changes, list of all created/modified files, and the core reason for the change.

**Get and Record Checkpoint SHA:**
- Obtain the hash of the *just-created checkpoint commit* (`git log -1 --format="%H"`).
- Append in `TODO.md` the first 7 characters of the commit hash in the format `[checkpoint: <sha>]`.

**Rules**: 
- NEVER push to remote
- NEVER force push/reset --hard/amend
- Commit after every sub-item (don't batch)
- Stage specific files (not `git add -A`)

**When a TODO item is completed**:
1. Move the entire item (with all sub-items) from `TODO.md` to the appropriate archive file in `docs/archive/`
2. Place it in the iteration where it was primarily worked on
3. Update `TODO.md` to remove the completed item
4. Keep the active TODO list focused on current and future work only
5. Make sure headings are one level higher in the archive (e.g., ## Iteration X. in TODO.md -> # Iteration X. in archive)

### Documentation Updates

- If `src/` behavior changes, update relevant docs in the same iteration.
- If execution workflow changes, update `AGENTS.md` and `CLAUDE.md`.
- If a task pattern repeats, add or revise a skill under `.codex/skills/`.

## Quality Gates

Run these before every commit:

```bash
uv run ruff check --fix
uv run pytest
```

- Resolve leftover issues from ruff fix.
- If a gate fails due to environment constraints, report the exact failure and what remains unverified.

## Testing Placement

- Place tests under `tests/`.
- Prefer targeted unit tests for behavior changes.
- Keep end-to-end tests explicit about external dependencies.

## Compatibility

- `CLAUDE.md` mirrors this contract for cross-tool compatibility.
- Canonical skills live in `.codex/skills/`.
- Optional mirror root `.agents/skills/` is not enabled in this repo right now.
