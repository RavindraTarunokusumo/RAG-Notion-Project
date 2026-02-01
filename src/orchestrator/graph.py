import logging

from langgraph.graph import END, StateGraph

from src.agents import (
    planner_node,
    reasoner_node,
    researcher_node,
    synthesiser_node,
)
from src.orchestrator.state import AgentState

logger = logging.getLogger(__name__)

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
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("synthesiser", synthesiser_node)

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
