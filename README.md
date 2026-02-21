# Notion Agentic RAG

Multi-agent RAG system for answering research questions from a Notion knowledge base and arXiv content.

## Core Pipeline

The query path is a linear LangGraph workflow:

`Planner -> Researcher -> Reasoner -> Synthesiser`

Optional A2A-style tool agents can be invoked at runtime for web search, code execution, citation validation, symbolic math, and diagram generation.

## Setup

Requirements: Python 3.11+

```bash
uv sync
cp .env.example .env
```

Populate `.env` with required API keys, then ingest your knowledge base:
- Required for default runtime: `DASHSCOPE_API_KEY`
- Optional (only if OpenAI provider is selected): `OPENAI_API_KEY`, `OPENAI_BASE_URL`

```bash
uv run python main.py --ingest
```

## Run

Web UI:

```bash
streamlit run app.py
```

CLI:

```bash
uv run python main.py "Tell me about my knowledge base."
uv run python main.py --test-conn
uv run python main.py --ingest --rebuild
```

## Validation

```bash
uv run ruff check .
uv run pytest
```

## Documentation Harness

Navigation order for agents and contributors:

1. `AGENTS.md` (primary runtime contract)
2. `docs/agent-harness.md`
3. Relevant technical docs in `docs/`
4. Matching task skill in `.codex/skills/`
5. Validation commands in `docs/testing.md` and `docs/commands.md`

Canonical docs:

- `docs/architecture.md`
- `docs/database.md`
- `docs/patterns.md`
- `docs/testing.md`
- `docs/commands.md`
- `docs/changelog.md`
- `docs/index.md`

Historical docs and prior planning artifacts are preserved in `docs/archive/`.
