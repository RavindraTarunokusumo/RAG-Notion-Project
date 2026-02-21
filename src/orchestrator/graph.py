import logging
import time

from langgraph.graph import END, StateGraph

from src.agents import (
    planner_node,
    reasoner_node,
    researcher_node,
    synthesiser_node,
)
from src.orchestrator.state import AgentState
from src.utils.debugging import (
    get_active_trace_session,
    merge_state,
)

logger = logging.getLogger(__name__)


def _instrument_node(node_name: str, node_func):
    """Wrap graph nodes to capture start/end/error state transitions."""

    def wrapper(state: AgentState) -> dict:
        trace_session = get_active_trace_session()
        state_before = dict(state)
        started = time.perf_counter()

        if trace_session is not None:
            trace_session.record_node_start(node_name, state_before)

        try:
            node_output = node_func(state)
            state_after = merge_state(state_before, node_output)

            if trace_session is not None:
                duration_ms = (time.perf_counter() - started) * 1000
                trace_session.record_node_end(
                    node_name=node_name,
                    state_before=state_before,
                    node_output=node_output,
                    state_after=state_after,
                    duration_ms=duration_ms,
                )
            return node_output
        except Exception as error:
            if trace_session is not None:
                trace_session.record_node_error(
                    node_name=node_name,
                    state_before=state_before,
                    error=error,
                )
            raise

    return wrapper

def create_rag_graph():
    """
    Creates the LangGraph workflow for the Notion Agentic RAG system.
    
    Flow:
    1. Planner: Decomposes query into sub-tasks
    2. Researcher: Retrieves documents for sub-tasks
    3. Reasoner: Analyzes retrieved docs against tasks
    4. Synthesiser: Generates final answer
    """
    logger.info("Building Agentic RAG Graph...")
    
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("planner", _instrument_node("planner", planner_node))
    workflow.add_node(
        "researcher",
        _instrument_node("researcher", researcher_node),
    )
    workflow.add_node("reasoner", _instrument_node("reasoner", reasoner_node))
    workflow.add_node(
        "synthesiser",
        _instrument_node("synthesiser", synthesiser_node),
    )

    # 2. Add Edges (Linear Flow)
    # Start -> Planner
    workflow.set_entry_point("planner")

    # Planner -> Researcher
    # (In a more complex graph, we might check if planning failed here)
    workflow.add_edge("planner", "researcher")

    # Researcher -> Reasoner
    workflow.add_edge("researcher", "reasoner")

    # Reasoner -> Synthesiser
    # (Could loop back to researcher if information is missing, but keeping it linear for V1)
    workflow.add_edge("reasoner", "synthesiser")

    # Synthesiser -> End
    workflow.add_edge("synthesiser", END)

    # 3. Compile
    app = workflow.compile()
    
    logger.info("Graph compiled successfully.")
    return app
