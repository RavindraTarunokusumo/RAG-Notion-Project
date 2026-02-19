# src/utils/session_manager.py

## Purpose
Persistent session storage and export utilities for Streamlit chat history.

## Main responsibilities
- Save/load/delete/rename sessions as JSON files.
- List sessions with metadata and recency sorting.
- Export sessions as JSON or Markdown.
- Generate default session names from first user message.
- Expose singleton accessor.

## Key symbols
- `SessionManager`
- `get_session_manager()`

## Storage
- Default directory: `./data/sessions`
- One file per session: `<session_id>.json`
