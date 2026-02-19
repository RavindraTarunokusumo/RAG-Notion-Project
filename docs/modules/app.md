# app.py

## Purpose
`app.py` is the Streamlit web interface for the Notion Agentic RAG system.
It handles chat UX, pipeline execution, source display, and session persistence controls.

## Main responsibilities
- Initialize Streamlit page config, styling, and app state.
- Build and reuse the LangGraph app via `create_rag_graph()`.
- Handle user prompts in streaming and non-streaming modes.
- Render workflow progress for planner -> researcher -> reasoner -> synthesiser.
- Render citation-linked source cards with filtering and search.
- Manage chat sessions through `src.utils.session_manager`.

## Key functions
- `initialize_session_state()`: bootstraps graph, tracing, and session manager.
- `enhance_citations(text)`: converts `[n]` markers into clickable anchors.
- `display_source_card(source, index)`: renders source metadata and preview.
- `render_workflow_progress(...)`: sidebar progress visualization.
- `process_query_streaming(prompt)`: yields pipeline progress + final answer events.
- `process_query(prompt)`: simple invoke path for non-streaming mode.
- `main()`: full Streamlit app composition.

## Inputs and outputs
- Input: user chat prompts from `st.chat_input`.
- Output: rendered assistant answer markdown and source cards.
- Internal output: appends conversation objects to `st.session_state.messages`.

## Dependencies
- Streamlit
- `src.orchestrator.graph.create_rag_graph`
- `src.utils.session_manager.get_session_manager`
- `src.utils.tracing.initialize_tracing`
- Runtime settings from `config.settings`

## Notes
- Streaming mode uses `graph.stream(initial_state)` and updates UI incrementally.
- Non-streaming mode uses `graph.invoke(initial_state)` and renders once.
