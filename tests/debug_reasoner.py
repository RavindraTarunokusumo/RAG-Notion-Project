import logging

from langchain_cohere import ChatCohere

from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_reasoning_model():
    try:
        logger.info(f"Testing model: {settings.models.reasoner_model}")
        llm = ChatCohere(
            model=settings.models.reasoner_model,
            cohere_api_key=settings.cohere_api_key,
            temperature=0.1
        )
        
        response = llm.invoke("What is 2+2? Explain your reasoning.")
        logger.info("Success!")
        logger.info(response)
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reasoning_model()
