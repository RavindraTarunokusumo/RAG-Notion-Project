# Session Plan - Current TODO Items

Date: 2026-02-18
Source: `TODO.md` (last reviewed 2026-02-11)

## Goal
Execute the remaining backlog in a dependency-aware order, starting with foundation hardening and AgentLightning integration.

## Planned Sessions

### Session 9 - AgentLightning Foundation
- [ ] NRAG-062: AgentLightning installation and configuration (P1)
- [ ] NRAG-063: Lightning store and emitter setup (P1)
- [ ] NRAG-064: Agent integration with emitters (P0)
- [ ] NRAG-065: Trajectory tracking (P1)

### Session 10 - Feedback and Prompt Optimization
- [ ] NRAG-066: User feedback interface in Streamlit (P0)
- [ ] NRAG-067: Automatic reward generation (P1)
- [ ] NRAG-068: Reward signal pipeline integration (P0)
- [ ] NRAG-069: APO prompt optimization (P1)

### Session 11 - Evaluation and Analytics
- [ ] NRAG-070: Evaluation framework (P1)
- [ ] NRAG-071: Analytics and monitoring dashboard (P2)
- [ ] NRAG-072: Documentation and best practices (P1)

## Parallel Backlog Streams

### Technical Debt (do early)
- [ ] TD-001: Comprehensive error handling (P1)
- [ ] TD-002: Structured logging (P1)
- [ ] TD-003: Input validation (P1)
- [ ] TD-004: Unit test coverage (P2)
- [ ] TD-005: Integration tests (P2)
- [ ] TD-006: Chunk size optimization (P2)
- [ ] TD-007: Type hints completion (P3)
- [ ] TD-008: API documentation (P3)

### UI Enhancements
- [ ] NRAG-055: Extended model configuration (P2)
- [ ] NRAG-056: Session title generation (P3)
- [ ] NRAG-057: Session auto-save and history (P2)
- [ ] NRAG-058: Knowledge base management buttons (P2)

### Model Ecosystem Expansion
- [ ] NRAG-059: Multi-provider LLM abstraction (P2)
- [ ] NRAG-060: OpenAI and Anthropic integration (P2)
- [ ] NRAG-061: Gemini, Qwen, and Deepseek support (P2)

### Deployment and Advanced Features (Unscoped)
- [ ] Docker containerization (P2)
- [ ] Cloud deployment (P2)
- [ ] REST API exposure (P2)
- [ ] Conversation memory (P2)
- [ ] Multi-database support (P2)
- [ ] Automated KB sync (P2)
- [ ] Semantic caching (P3)

## Execution Order
1. TD-001/002/003 (stability baseline)
2. Session 9 (NRAG-062 to NRAG-065)
3. NRAG-055/057/058 (UI stability + operations)
4. Session 10 (NRAG-066 to NRAG-069)
5. Model ecosystem (NRAG-059 to NRAG-061)
6. Session 11 (NRAG-070 to NRAG-072)
7. Deployment and advanced features

## Notes
- APO is API-compatible and does not require model weight access.
- Keep `TODO.md` as backlog source; update this session plan as scope changes.
