import logging

from src.agents.llm_factory import get_agent_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_reasoning_model():
    try:
        llm = get_agent_llm("reasoner")
        logger.info("Testing reasoner model via provider abstraction")
        
        response = llm.invoke("What is 2+2? Explain your reasoning.")
        logger.info("Success!")
        logger.info(response)
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reasoning_model()
