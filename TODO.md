# TODO - Notion Agentic RAG

> Derived from `docs/PROJECT.md` Future Backlog. Ordered by priority and dependency.
> Last reviewed: 2026-02-11

## Completed (Sessions 1-8)

- [x] Sessions 1-6: Core pipeline (Planner, Researcher, Reasoner, Synthesiser)
- [x] Session 7: Streamlit UI (NRAG-050 to NRAG-054)
- [x] Session 8: A2A Tool Agents (NRAG-027 to NRAG-033)

---

## Up Next

### Session 9: AgentLightning Foundation

Install AgentLightning and wire it into the existing agent pipeline for trajectory collection.

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-060 | AgentLightning Installation & Configuration | P1 | `config/agl_settings.py`, update `pyproject.toml`, `.env.example` |
| NRAG-061 | Lightning Store & Emitter Setup | P1 | `src/agl/store_manager.py`, `src/agl/emitter.py` |
| NRAG-062 | Agent Integration with Emitters | P0 | Wrap agent functions with `emit_agent_span` decorator |
| NRAG-063 | Trajectory Tracking | P1 | `src/agl/trajectory.py`, context manager for query-response cycles |

### Session 10: User Feedback & Reward System

Collect user signals and generate reward data for RL training.

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-064 | User Feedback Interface in Streamlit | P0 | Thumbs up/down, star rating, text feedback in `app.py` |
| NRAG-065 | Automatic Reward Generation | P1 | Intermediate rewards per agent stage (retrieval quality, confidence, etc.) |
| NRAG-066 | Reward Signal Pipeline Integration | P0 | Wire rewards into AGL store, user feedback overrides auto-rewards |

### Session 11: Training Infrastructure & Evaluation

Build the training loop and measure improvement.

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-067 | Training Script Implementation | P1 | `scripts/train_agents.py`, PPO/DPO/GRPO, checkpoint mgmt |
| NRAG-068 | Evaluation Framework | P1 | Test dataset, metrics (F1, BLEU, quality), baseline comparison |
| NRAG-069 | Analytics & Monitoring Dashboard | P2 | Streamlit page for trajectory stats, reward distribution, training progress |
| NRAG-070 | Documentation & Best Practices | P1 | Setup guide, reward engineering, troubleshooting |

---

## UI Enhancements (Backlog)

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-055 | Extended Model Configuration | P2 | Per-agent model selectors in settings panel |
| NRAG-056 | Session Title Generation | P3 | LLM-generated session titles from first query |
| NRAG-057 | Session Autosave & History | P2 | Auto-save after each response, no data loss on refresh |
| NRAG-058 | Manual Reference Linking UI | P3 | Let users manually link/tag references |

## Model Ecosystem Expansion (Backlog)

Branch: `epic/model-ecosystem-expansion`

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-060 | Multi-Provider LLM Abstraction | P2 | Refactor `llm_factory.py` with Provider enum, unified interface |
| NRAG-061 | OpenAI & Anthropic Integration | P2 | `langchain-openai`, `langchain-anthropic`, model mapping |
| NRAG-062 | Gemini, Qwen & Deepseek Support | P2 | `langchain-google-genai`, OpenAI-compatible endpoints |

> **Note:** NRAG-060/061/062 IDs are reused between AgentLightning and Model Ecosystem epics in PROJECT.md. These are distinct tasks under different epics.

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
2. **Session 9** - AgentLightning foundation (data collection starts early)
3. **NRAG-055/057** - UI polish (extended models, autosave)
4. **Session 10** - Feedback & rewards
5. **Model Ecosystem** - Multi-provider support
6. **Session 11** - Training & evaluation
7. **Deployment** - Containerization, API, cloud
8. **Advanced Features** - Memory, caching, multi-DB
