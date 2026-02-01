import logging
import time

from src.agents.planner import planner_node
from src.agents.researcher import researcher_node
from src.orchestrator.state import AgentState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_planner_and_researcher():
    logger.info("Initializing Test State...")
    state: AgentState = {
        "query": "What are the key architectural improvements in RAG systems mentioned in the Notion database and how do they relate to recent Arxiv papers?",
        "sub_tasks": [],
        "planning_reasoning": "",
        "retrieved_docs": [],
        "retrieval_metadata": {},
        "analysis": [],
        "overall_assessment": "",
        "final_answer": "",
        "sources": [],
        "error": None,
        "current_agent": "start"
    }

    # 1. Test Planner
    logger.info("\n--- Testing Planner Agent ---")
    start_time = time.time()
    planner_result = planner_node(state)
    logger.info(f"Planner took: {time.time() - start_time:.2f}s")
    
    if "error" in planner_result:
        logger.error(f"Planner failed: {planner_result['error']}")
        return

    # Update state
    state.update(planner_result)
    logger.info("Planner Result:")
    logger.info(f"Reasoning: {state['planning_reasoning']}")
    for i, task in enumerate(state['sub_tasks']):
        logger.info(f"Task {i+1}:P{task} - {task['task']}")

    # 2. Test Researcher
    logger.info("\n--- Testing Researcher Agent ---")
    start_time = time.time()
    researcher_result = researcher_node(state)
    
    # Update state
    state.update(researcher_result)
    
    logger.info(f"Researcher took: {time.time() - start_time:.2f}s")
    logger.info(f"Total Documents Retrieved: {state['retrieval_metadata'].get('total_docs', 0)}")
    
    if state['retrieved_docs']:
        logger.info("\nSample Retrieved Documents:")
        for i, doc in enumerate(state['retrieved_docs'][:3]):
            logger.info(f"[{i+1}] Source: {doc.metadata.get('source', 'unknown')} | Content: {doc.page_content[:100]}...")

if __name__ == "__main__":
    test_planner_and_researcher()
