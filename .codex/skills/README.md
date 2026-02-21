# Canonical Skills

This directory is the canonical skill source for this repository.

## Policy

- Canonical root: `.codex/skills/`
- Optional mirror root: `.agents/skills/` (not enabled in this repository)
- If mirrors are introduced, update canonical content first, then mirror.

## Required Skill Package Shape

```text
.codex/skills/<skill-name>/
  SKILL.md
  references/
  scripts/
  assets/
  agents/
```

Only `SKILL.md` is required. Other directories are optional and task-dependent.

## Authoring

Use `.codex/skills/_template/` as the starting point for new skills.
