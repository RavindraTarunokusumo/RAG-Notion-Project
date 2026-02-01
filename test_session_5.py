import logging
import time
from src.agents import planner_node, researcher_node, reasoner_node, synthesiser_node
from src.orchestrator.state import AgentState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_full_agent_chain():
    logger.info("Initializing Full Chain Test State...")
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

    # 1. Planner
    logger.info("\n--- 1. Planner Agent ---")
    state.update(planner_node(state))
    if state['error']: return

    # 2. Researcher
    logger.info("\n--- 2. Researcher Agent ---")
    state.update(researcher_node(state))
    if state['error']: return
    
    # 3. Reasoner
    logger.info("\n--- 3. Reasoner Agent ---")
    start_time = time.time()
    state.update(reasoner_node(state))
    logger.info(f"Reasoner took: {time.time() - start_time:.2f}s")
    
    if state['error']:
        logger.error(state['error'])
        return
        
    logger.info(f"Overall Assessment: {state['overall_assessment'][:100]}...")

    # 4. Synthesiser
    logger.info("\n--- 4. Synthesiser Agent ---")
    start_time = time.time()
    state.update(synthesiser_node(state))
    logger.info(f"Synthesiser took: {time.time() - start_time:.2f}s")
    
    if state['error']:
        logger.error(state['error'])
        return

    logger.info("\n" + "="*60)
    logger.info("FINAL ANSWER")
    logger.info("="*60)
    logger.info(state['final_answer'])
    logger.info("\nSources:")
    for s in state['sources']:
        logger.info(f"- {s['title']} ({s['source']})")

if __name__ == "__main__":
    test_full_agent_chain()
