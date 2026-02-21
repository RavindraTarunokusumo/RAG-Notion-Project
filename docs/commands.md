# Operational Commands

## Environment and Dependencies

```bash
uv sync
```

## Run Interfaces

```bash
streamlit run app.py
uv run python main.py "Your query"
```

## Ingestion and Maintenance

```bash
uv run python main.py --ingest
uv run python main.py --ingest --rebuild
uv run python main.py --test-conn
```

## Quality Gates

```bash
uv run ruff check .
uv run pytest
```

## Focused Debugging

```bash
uv run pytest tests/test_tool_agents.py -v
uv run python main.py "Your query" --verbose
```

## Notes

- Prefer `uv run` for command consistency.
- Use targeted test commands while iterating, then run full checks before handoff.
