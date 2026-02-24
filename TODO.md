# TODO - Notion Agentic RAG

Active backlog only.  
Historical session/task logs are archived under `docs/archive/`.

Last reviewed: 2026-02-21

Recent checkpoints:
[checkpoint: 52701f9]
[checkpoint: c1d4dce]
[checkpoint: aa9a54b]
[checkpoint: f747fe1]

## Up Next

### AgentLightning Epic (NRAG-062 to NRAG-071)

Constraint: All agents use hosted LLM APIs (no model weight access). RL weight training is out of scope. APO-style prompt optimization remains in scope.

### Session 9: AgentLightning Foundation

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-062 | AgentLightning Installation and Configuration | P1 | `config/agl_settings.py`, update `pyproject.toml`, `.env.example` |
| NRAG-063 | Lightning Store and Emitter Setup | P1 | `src/agl/store_manager.py`, `src/agl/emitter.py` |
| NRAG-064 | Agent Integration with Emitters | P0 | Wrap agent functions with `emit_agent_span` decorator |
| NRAG-065 | Trajectory Tracking | P1 | `src/agl/trajectory.py`, context manager for query-response cycles |

### Session 10: Feedback and Prompt Optimization

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-066 | User Feedback Interface in Streamlit | P0 | Thumbs up/down, star rating, text feedback in `app.py` |
| NRAG-067 | Automatic Reward Generation | P1 | Intermediate rewards per agent stage |
| NRAG-068 | Reward Signal Pipeline Integration | P0 | Wire rewards into AGL store |
| NRAG-069 | APO Prompt Optimization | P1 | LLM critiques trajectories and rewrites prompts |

### Session 11: Evaluation and Analytics

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-070 | Evaluation Framework | P1 | Query dataset, metrics, baseline vs optimized |
| NRAG-071 | Analytics and Monitoring Dashboard | P2 | Streamlit stats and prompt version tracking |
| NRAG-072 | Documentation and Best Practices | P1 | Setup guide, APO workflow, troubleshooting |

## UI Enhancements

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-055 | Extended Model Configuration | P2 | Per-agent model selectors in settings panel |
| NRAG-056 | Session Title Generation | P3 | LLM-generated session titles from first query |
| NRAG-057 | Session Auto-save and History | P2 | Auto-save after each response |
| NRAG-058 | Knowledge Base Management Buttons | P2 | Connection, ingest, rebuild controls |

## Model Ecosystem Expansion

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| NRAG-059 | Multi-Provider LLM Abstraction | P2 | Refactor `llm_factory.py` with provider abstraction |
| NRAG-060 | OpenAI and Anthropic Integration | P2 | Add provider adapters and model mapping |
| NRAG-061 | Gemini, Qwen and Deepseek Support | P2 | Add OpenAI-compatible endpoint integration |

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
