# Utils: session_manager

Source module: `src/utils/session_manager.py`

## Purpose

Manage Streamlit chat session persistence, loading, deletion, and export.

## Runtime Behavior

- Sessions are stored under `data/sessions/`.
- Session files are JSON and intended for local persistence.
- UI interactions load and persist session state through this module.

## Invariants

- Session identifiers should remain stable and unique.
- Session serialization should be backward-compatible where possible.
- Export formats should not mutate stored source session data.
