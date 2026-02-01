import logging

from src.loaders.pipeline import DocumentPipeline
from src.rag.vectorstore import get_vector_store
from src.utils.tracing import initialize_tracing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ingestion(rebuild: bool = False):
    """
    Run the full ingestion pipeline:
    1. Load and process documents from Notion and Arxiv.
    2. Store them in the Vector Database (ChromaDB).
    
    Args:
        rebuild (bool): If True, clears the vector store before ingestion.
    """
    initialize_tracing()
    
    logger.info("Starting ingestion process...")
    if rebuild:
        logger.info("Rebuild flag is set (clearing not implemented yet, proceeding with append).")
    
    pipeline = DocumentPipeline()
    try:
        documents = pipeline.run() # Sync call
    except Exception as e:
        logger.error(f"Pipeline run failed: {e}")
        return
    
    if not documents:
        logger.warning("No documents found/processed. Aborting ingestion.")
        return

    logger.info(f"Retrieved and processed {len(documents)} document chunks.")
    
    vector_store = get_vector_store()
    
    try:
        vector_store.add_documents(documents)
        logger.info("Successfully ingested documents into Vector Store.")
    except Exception as e:
        logger.error(f"Failed to ingest documents: {e}")
        raise e

if __name__ == "__main__":
    run_ingestion()
