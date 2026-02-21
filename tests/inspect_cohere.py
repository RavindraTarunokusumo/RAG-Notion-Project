import logging

from src.agents.llm_factory import get_agent_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_client():
    llm = get_agent_llm("reasoner")
    logger.info(f"LLM type: {type(llm)}")
    try:
        response = llm.invoke("Return the word 'ok'.")
        logger.info(f"Invocation response type: {type(response)}")
    except Exception as e:
        logger.info(f"Invocation error: {e}")

if __name__ == "__main__":
    inspect_client()
