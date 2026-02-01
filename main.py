import argparse
import logging
from src.utils.tracing import initialize_tracing
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Notion Agentic RAG System")
    parser.add_argument("--query", "-q", type=str, help="Query to process")
    parser.add_argument("--ingest", action="store_true", help="Run document ingestion")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tracing
    initialize_tracing()
    logger.info(f"LangSmith project: {settings.langchain_project}")
    
    if args.ingest:
        logger.info("Starting document ingestion...")
        # from src.loaders.pipeline import ingest_documents
        # ingest_documents()
        logger.info("Ingestion functionality not yet implemented (Session 2)")
        
    elif args.query:
        logger.info(f"Processing query: {args.query}")
        # result = agentic_rag.invoke(args.query)
        # print(result["final_answer"])
        logger.info("Query functionality not yet implemented (Session 6)")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
