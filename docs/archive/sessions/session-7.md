## Session 7: User Interface Enhancements

**Focus:** Streamlit UI enhancement backlog items

## Status Summary

- [x] **NRAG-055**: Extended Model Configuration
- [ ] **NRAG-056**: Session Title Generation
- [ ] **NRAG-057**: Session Auto-save and History
- [ ] **NRAG-058**: Knowledge Base Management Buttons

---

### NRAG-055: Extended Model Configuration

**Priority:** P2 - Medium  
**Token Estimate:** 8,000  
**Status:** Complete

**Description:**  
Enable model selection for Planner, Researcher, Reasoner, and Synthesiser so users can optimize quality, speed, and cost.

**Requirements:**

1. Granular control:
- Selectors for each agent type in the settings panel
- Support compatible model families per role

2. Configuration persistence:
- Save model choices in session state
- Ensure LLM factory respects per-agent model settings

**Implementation Notes:**

- Expand `app.py` settings with per-agent model selectors.
- Use distinct keys like `model_planner`, `model_researcher`, `model_reasoner`, and `model_synthesiser`.

---

### NRAG-056: Session Title Generation

**Priority:** P3 - Low  
**Token Estimate:** 5,000  
**Status:** Pending

**Description:**  
Generate a descriptive session title from the first prompt using a lightweight model.

**Requirements:**

1. Automatic generation:
- Trigger after first user prompt
- Update session title in sidebar without reload

2. Model selection:
- Use a low-cost model for this task
- Avoid heavy reasoning model use for titling

**Implementation Notes:**

- Add a title generation helper in `src/utils/session_manager.py`.
- Reuse planner-grade or lighter model invocation.

---

### NRAG-057: Session Auto-save and History

**Priority:** P2 - Medium  
**Token Estimate:** 6,000  
**Status:** Pending

**Description:**  
Auto-save sessions during prompt and response flow to reduce data loss risk on refresh/interruption.

**Requirements:**

1. Autosave trigger:
- Remove manual save dependency for normal flow
- Save immediately on user submit (persist prompt)
- Save again on assistant completion (persist response)
- Ensure parity across streaming and non-streaming modes

---

### NRAG-058: Knowledge Base Management Buttons

**Priority:** P2 - Medium  
**Token Estimate:** 8,000  
**Status:** Pending

**Description:**  
Add sidebar controls for connection tests, ingestion, and rebuild to expose current CLI-only operations in UI.

**Current CLI operations to surface:**

- `python main.py --ingest`
- `python main.py --ingest --rebuild`
- `python main.py --test-conn`

**Requirements:**

1. Test connection button:
- Check LangSmith (optional) and Cohere embedding connectivity
- Render inline pass/fail status
- Show embedding vector dimension on success

2. Ingest button:
- Run ingestion pipeline
- Display progress and completion summary

3. Rebuild button:
- Confirm destructive rebuild intent
- Show status/progress and completion result

**Acceptance Criteria:**

- [ ] Test connection button reports LangSmith and Cohere status
- [ ] Ingest button runs ingestion pipeline successfully
- [ ] Rebuild flow includes explicit confirmation step
- [ ] Long operations show progress indicators
- [ ] Success/error messages are clear and actionable
- [ ] Buttons prevent duplicate concurrent execution
- [ ] App state remains stable after operation completion

**Implementation Notes:**

- Add controls in `app.py` sidebar near existing settings/system blocks.
- Reuse ingestion and connection-test logic already implemented in CLI pathways.
