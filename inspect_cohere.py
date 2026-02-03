from langchain_cohere import ChatCohere
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_client():
    llm = ChatCohere(
        model=settings.models.reasoner_model,
        cohere_api_key=settings.cohere_api_key
    )
    client = llm.client
    logger.info(f"Client type: {type(client)}")
    
    if hasattr(client, "v2"):
        v2 = client.v2
        logger.info(f"client.v2 type: {type(v2)}")
        logger.info(f"client.v2.chat: {v2.chat}")
    
    # Check if we can patch instance
    def my_chat(*args, **kwargs):
        logger.info("PATCHED CHAT CALLED")
        return "mock"
    
    # Try patching v2.chat
    if hasattr(client, "v2"):
        client.v2.chat = my_chat
        logger.info(f"client.v2.chat after patch: {client.v2.chat}")
    
    # Verify if langchain uses it
    try:
        # We catch the error because "mock" is not a valid response object
        llm.invoke("test") 
    except Exception as e:
        logger.info(f"Caught expected error: {e}")

if __name__ == "__main__":
    inspect_client()
