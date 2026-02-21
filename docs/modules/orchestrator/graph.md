# src/orchestrator/graph.py

## Purpose
Defines the LangGraph workflow topology for the core four-agent pipeline.

## Main responsibilities
- Build a typed `StateGraph` over `AgentState`.
- Register nodes: planner, researcher, reasoner, synthesiser.
- Configure linear edges and compile the graph object.

## Key function
- `create_rag_graph()`: returns compiled graph app.

## Pipeline
`planner -> researcher -> reasoner -> synthesiser -> END`

## Dependencies
- `langgraph.graph.StateGraph`
- Agent node callables from `src.agents`
- Shared state contract from `src.orchestrator.state.AgentState`
