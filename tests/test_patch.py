import logging

import cohere
from langchain_cohere import ChatCohere

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MONKEY PATCH START
# Patch the ThinkingAssistantMessageResponseContentItem to have a text attribute
# which returns the actual thinking content, so langchain can consume it (or ignore it).

if hasattr(cohere, "ThinkingAssistantMessageResponseContentItem"):
    logger.info("Found ThinkingAssistantMessageResponseContentItem, patching it...")
    
    original_init = cohere.ThinkingAssistantMessageResponseContentItem.__init__
    
    # We can property patch it or just add the attribute
    # Since it might be a Pydantic model (likely), we might need to add a property
    
    class PatchedThinking(cohere.ThinkingAssistantMessageResponseContentItem):
        @property
        def text(self):
            return self.thinking

    # Re-assign the class in the module so langchain uses our patched version?
    # No, langchain uses the type returned by the API client.
    # We need to modify the class itself.
    
    def get_text(self):
        return self.thinking
        
    cohere.ThinkingAssistantMessageResponseContentItem.text = property(get_text)
    logger.info("Patch applied.")
else:
    logger.warning("ThinkingAssistantMessageResponseContentItem not found in cohere module.")

# MONKEY PATCH END


def test_reasoning_model():
    try:
        logger.info(f"Testing model: {settings.models.reasoner_model}")
        # Ensure we are using the reasoning model for the test
        reasoner_model = "command-a-reasoning-08-2025" # Explicitly use the one that failed
        
        llm = ChatCohere(
            model=reasoner_model,
            cohere_api_key=settings.cohere_api_key,
            temperature=0.1
        )
        
        response = llm.invoke("What is 1+1? Explain your reasoning.")
        logger.info("Success!")
        logger.info(response)
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reasoning_model()
