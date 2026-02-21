# TODO - Notion Agentic RAG

> Derived from `docs/PROJECT.md` Future Backlog. Ordered by priority and dependency.
> Last reviewed: 2026-02-11

## Completed (Sessions 1-8)

- [x] Sessions 1-6: Core pipeline (Planner, Researcher, Reasoner, Synthesiser)
- [x] Session 7: Streamlit UI (NRAG-050 to NRAG-054)
- [x] Session 8: A2A Tool Agents (NRAG-027 to NRAG-033)

---

## Up Next

### AgentLightning Epic (NRAG-062 to NRAG-071)

> **Constraint:** All agents use hosted LLM APIs (no model weight access). RL weight
> training (PPO/DPO/GRPO via verl) is **not applicable**. However, AgentLightning's
> **Automatic Prompt Optimization (APO)** works with API-only models - it uses an LLM
> to critique trajectories and rewrite prompts, requiring no weight access.
>
> **Kept:** Store, emitters, trajectory tracking, feedback, APO, evaluation, analytics.
> **Dropped:** RL training scripts (PPO/DPO/GRPO weight training).

### Session 9: AgentLightning Foundation

Install AgentLightning, set up trajectory collection, and wire emitters into agent pipeline.

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-062 | AgentLightning Installation & Configuration | P1 | `config/agl_settings.py`, update `pyproject.toml`, `.env.example` |
| NRAG-063 | Lightning Store & Emitter Setup | P1 | `src/agl/store_manager.py`, `src/agl/emitter.py` |
| NRAG-064 | Agent Integration with Emitters | P0 | Wrap agent functions with `emit_agent_span` decorator |
| NRAG-065 | Trajectory Tracking | P1 | `src/agl/trajectory.py`, context manager for query-response cycles |

### Session 10: Feedback & Prompt Optimization

Collect user feedback and use APO to iteratively improve agent prompts.

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-066 | User Feedback Interface in Streamlit | P0 | Thumbs up/down, star rating, text feedback in `app.py` |
| NRAG-067 | Automatic Reward Generation | P1 | Intermediate rewards per agent stage (retrieval quality, confidence, etc.) |
| NRAG-068 | Reward Signal Pipeline Integration | P0 | Wire rewards into AGL store, user feedback overrides auto-rewards |
| NRAG-069 | APO Prompt Optimization | P1 | Use APO algorithm (LLM critiques trajectories → rewrites prompts). No weight training. Requires a critique LLM (e.g. GPT-4.1-mini). |

### Session 11: Evaluation & Analytics

Measure improvement from optimized prompts and monitor pipeline health.

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-070 | Evaluation Framework | P1 | Test query dataset, metrics (retrieval precision, answer quality), baseline vs optimized comparison |
| NRAG-071 | Analytics & Monitoring Dashboard | P2 | Streamlit page for trajectory stats, reward distribution, prompt version tracking |
| NRAG-072 | Documentation & Best Practices | P1 | Setup guide, APO workflow, reward design, troubleshooting |

---

## UI Enhancements (Backlog)

Branch: `epic/streamlit`

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-055 | Extended Model Configuration | P2 | Per-agent model selectors in settings panel |
| NRAG-056 | Session Title Generation | P3 | LLM-generated session titles from first query |
| NRAG-057 | Session Auto-save & History | P2 | Auto-save after each response, no data loss on refresh. Remove manual Save button. |
| NRAG-058 | Knowledge Base Management Buttons | P2 | UI controls for test connection, ingest, and rebuild operations |

## Model Ecosystem Expansion (Backlog)

Branch: `epic/model-ecosystem-expansion`

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-059 | Multi-Provider LLM Abstraction | P2 | Refactor `llm_factory.py` with Provider enum, unified interface |
| NRAG-060 | OpenAI & Anthropic Integration | P2 | `langchain-openai`, `langchain-anthropic`, model mapping |
| NRAG-061 | Gemini, Qwen & Deepseek Support | P2 | `langchain-google-genai`, OpenAI-compatible endpoints |

## Advanced Features (Backlog, Unscoped)

| Feature | Priority | Notes |
|---------|----------|-------|
| Conversation Memory | P2 | Cross-session context retention |
| Semantic Caching | P3 | Cache similar queries to avoid redundant LLM calls |
| Multi-Database Support | P2 | Support multiple Notion databases |
| Automated KB Sync | P2 | Periodic re-ingestion from Notion |

## Deployment (Backlog, Unscoped)

| Feature | Priority | Notes |
|---------|----------|-------|
| Docker Containerization | P2 | Dockerfile, docker-compose |
| Cloud Deployment | P2 | AWS/GCP/Azure hosting |
| REST API Exposure | P2 | FastAPI wrapper around RAG pipeline |

## Technical Debt

| ID | Issue | Priority |
|----|-------|----------|
| TD-001 | Comprehensive error handling | P1 |
| TD-002 | Structured logging | P1 |
| TD-003 | Input validation | P1 |
| TD-004 | Unit test coverage | P2 |
| TD-005 | Integration tests | P2 |
| TD-006 | Chunk size optimization | P2 |
| TD-007 | Type hints completion | P3 |
| TD-008 | API documentation | P3 |

---

## Suggested Order of Execution

1. **TD-001/002/003** - Harden the foundation (error handling, logging, validation)
5. **Model Ecosystem (NRAG-059 to 061)** - Multi-provider support (Cohere API requires rate limiting)
3. **NRAG-055/057/058** - UI polish (extended models, autosave, KB management)
4. **Session 10 (NRAG-066 to 068)** - Feedback & APO prompt optimization
2. **Session 9 (NRAG-062 to 065)** - AgentLightning foundation (collect trajectories once full system operational)
6. **Session 11 (NRAG-070 to 072)** - Evaluation & analytics
7. **Deployment** - Containerization, API, cloud
8. **Advanced Features** - Memory, caching, multi-DB
