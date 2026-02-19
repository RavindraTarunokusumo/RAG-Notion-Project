# Docs Index

This is the canonical documentation entry table for the repository.

## Core Documents

| Entry | Purpose |
|---|---|
| `README.md` | Repo overview, setup, usage |
| `docs/PROJECT.md` | Architecture and system design details |
| `TODO.md` | Backlog and next tasks |
| `docs/ISSUE_FORMAT.md` | Issue and reporting template |
| `docs/modules/README.md` | Module documentation index |
| `docs/exec-plans/active/` | Active execution plans for non-trivial work |
| `docs/sessions/` | Dated session plans/checkpoints and TODO execution tracking |

## Planning Documents

| Entry Pattern | Purpose |
|---|---|
| `docs/exec-plans/active/<slug>.md` | Active implementation plans that drive in-flight changes |
| `docs/sessions/<YYYY-MM-DD>-<slug>.md` | Session-level planning/checkpoints aligned to backlog execution |

## Module-to-Docs Map

| Area | Module | Documentation |
|---|---|---|
| Configuration | `config/settings.py` | `docs/modules/config/settings.md` |
| Entry point | `app.py` | `docs/modules/app.md` |
| Entry point | `main.py` | `docs/modules/main.md` |
| Entry point | `src/ingest.py` | `docs/modules/ingest.md` |
| Orchestration | `src/orchestrator/graph.py` | `docs/modules/orchestrator/graph.md` |
| Orchestration | `src/orchestrator/state.py` | `docs/modules/orchestrator/state.md` |
| Agents | `src/agents/__init__.py` | `docs/modules/agents/package.md` |
| Agents | `src/agents/llm_factory.py` | `docs/modules/agents/llm_factory.md` |
| Agents | `src/agents/planner.py` | `docs/modules/agents/planner.md` |
| Agents | `src/agents/researcher.py` | `docs/modules/agents/researcher.md` |
| Agents | `src/agents/reasoner.py` | `docs/modules/agents/reasoner.md` |
| Agents | `src/agents/synthesiser.py` | `docs/modules/agents/synthesiser.md` |
| RAG + retrieval | `src/rag/embeddings.py` | `docs/modules/rag/embeddings.md` |
| RAG + retrieval | `src/rag/retriever.py` | `docs/modules/rag/retriever.md` |
| RAG + retrieval | `src/rag/text_processing.py` | `docs/modules/rag/text_processing.md` |
| RAG + retrieval | `src/rag/vectorstore.py` | `docs/modules/rag/vectorstore.md` |
| Loaders | `src/loaders/notion_loader.py` | `docs/modules/loaders/notion_loader.md` |
| Loaders | `src/loaders/pipeline.py` | `docs/modules/loaders/pipeline.md` |
| Loaders | `src/loaders/arxiv_loader.py` | `docs/modules/loaders/arxiv_loader.md` |
| Utilities | `src/utils/helpers.py` | `docs/modules/utils/helpers.md` |
| Utilities | `src/utils/tracing.py` | `docs/modules/utils/tracing.md` |
| Utilities | `src/utils/session_manager.py` | `docs/modules/utils/session_manager.md` |
| Tool framework | `src/tools/__init__.py` | `docs/modules/tools/package.md` |
| Tool framework | `src/tools/base.py` | `docs/modules/tools/base.md` |
| Tool framework | `src/tools/registry.py` | `docs/modules/tools/registry.md` |
| Tool framework | `src/tools/client.py` | `docs/modules/tools/client.md` |
| Tool framework | `src/tools/web_searcher.py` | `docs/modules/tools/web_searcher.md` |
| Tool framework | `src/tools/code_executor.py` | `docs/modules/tools/code_executor.md` |
| Tool framework | `src/tools/citation_validator.py` | `docs/modules/tools/citation_validator.md` |
| Tool framework | `src/tools/math_solver.py` | `docs/modules/tools/math_solver.md` |
| Tool framework | `src/tools/diagram_generator.py` | `docs/modules/tools/diagram_generator.md` |
