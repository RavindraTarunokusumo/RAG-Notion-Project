import cohere
import logging
from langchain_cohere import ChatCohere
from config.settings import settings
import functools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MONKEY PATCH STRATEGY
# 1. Patch ThinkingAssistantMessageResponseContentItem to have .text (prevents Attribute Error)
# 2. Patch cohere.ClientV2.chat to merge Thinking + Text content so LangChain (which only reads [0]) gets everything.

# Step 1: Attribute Error Fix
if hasattr(cohere, "ThinkingAssistantMessageResponseContentItem"):
    logger.info("Patching ThinkingAssistantMessageResponseContentItem...")
    def get_text(self):
        return f"<THINKING>\n{self.thinking}\n</THINKING>"
    cohere.ThinkingAssistantMessageResponseContentItem.text = property(get_text)

# Step 2: Content Merging Fix
def patch_cohere_client():
    # We need to find the Client class used by LangChain. 
    # Usually langchain_cohere instantiates cohere.Client or cohere.ClientV2
    
    # Let's inspect what ChatCohere uses
    llm = ChatCohere(cohere_api_key="test", model="command-r-plus")
    client = llm.client
    logger.info(f"LangChain is using client type: {type(client)}")
    
    # It seems it uses cohere.Client (which might dynamically use V1 or V2).
    # Based on logs, it hits V2 endpoints.
    
    # We will wrap the `chat` method of the client class.
    ClientClass = type(client)
    
    original_chat = ClientClass.chat
    
    @functools.wraps(original_chat)
    def patched_chat(self, *args, **kwargs):
        # Call original API
        response = original_chat(self, *args, **kwargs)
        
        # Check if response has structured content (V2 style)
        # response should be an instance of V2ChatResponse or similar
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            content = response.message.content
            if isinstance(content, list) and len(content) > 1:
                logger.info(f"Detected multi-part response with {len(content)} items. Merging...")
                
                # Check for Thinking + Text pattern
                merged_text = ""
                for item in content:
                    if hasattr(item, 'thinking'): # It's a thinking item
                        merged_text += f"<THINKING>\n{item.thinking}\n</THINKING>\n\n"
                    elif hasattr(item, 'text'):   # It's a text item
                        merged_text += item.text
                
                # Create a new single text item
                # We need to instantiate a TextAssistantMessageResponseContentItem or similar
                # Or just modify the first item to contain all text if it has a .text attribute (which we added in Step 1)
                
                # Since we added .text to Thinking item in Step 1, we can just modify the first item's source data
                # But Thinking item uses .thinking field.
                
                # Safer: Replace the content list with a single TextAssistantMessageResponseContentItem
                # We need to import it.
                from cohere import TextAssistantMessageResponseContentItem
                
                new_item = TextAssistantMessageResponseContentItem(
                    type="text",
                    text=merged_text
                )
                
                response.message.content = [new_item]
                logger.info("Merged content into single Text item.")
                
        return response

    ClientClass.chat = patched_chat
    logger.info("Patched cohere.Client.chat")

patch_cohere_client()

def test_reasoning_full_flow():
    try:
        logger.info(f"Testing model: command-a-reasoning-08-2025")
        
        llm = ChatCohere(
            model="command-a-reasoning-08-2025",
            cohere_api_key=settings.cohere_api_key,
            temperature=0.1
        )
        
        # This prompt asks for JSON to verify we get the Answer part, not just Thinking
        response = llm.invoke("Generate a JSON object with key 'result' and value 'success'. Explain your reasoning first.")
        
        logger.info("Invocation successful.")
        logger.info("-" * 20)
        logger.info(response.content)
        logger.info("-" * 20)
        
        if '"result": "success"' in response.content:
            logger.info("VERIFIED: JSON output found in response.")
        else:
            logger.error("FAILED: JSON output missing (likely only got thinking trace).")
            
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reasoning_full_flow()
