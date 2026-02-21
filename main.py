import argparse
import logging
import sys
from datetime import datetime

from config.settings import settings
from src.orchestrator.graph import create_rag_graph
from src.orchestrator.state import build_initial_state
from src.utils.debugging import configure_logging, debug_run
from src.utils.tracing import initialize_tracing

logger = logging.getLogger(__name__)

def run_agentic_rag(query: str):
    """
    Executes the full Agentic RAG pipeline for a given query.
    """
    logger.info("="*60)
    logger.info(f"RAG QUERY PROCESSING: {query}")
    logger.info("="*60)
    
    # Initialize Graph
    try:
        app = create_rag_graph()
    except Exception as e:
        logger.error(f"Failed to initialize graph: {e}")
        sys.exit(1)
    
    # Initial State
    initial_state = build_initial_state(query)
    
    # Execute Graph
    start_time = datetime.now()
    try:
        # Use invoke for a single run
        with debug_run(query=query, initial_state=initial_state, mode="invoke") as trace_session:
            result = app.invoke(initial_state)
            if trace_session is not None:
                trace_session.record_run_end(result)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if result.get("error"):
            logger.error(f"Pipeline failed: {result['error']}")
            print(f"\nERROR: {result['error']}")
            return

        # Output Results
        print("\n" + "="*80)
        print("FINAL ANSWER")
        print("="*80)
        print(result["final_answer"])
        print("\n" + "-"*80)
        
        if result.get("sources"):
            print("SOURCES:")
            for source in result["sources"]:
                print(f" - {source.get('title', 'Untitled')} ({source.get('source', 'Unknown')})")
        else:
            print("Sources: None found.")
            
        print("="*80)
        logger.info(f"Request completed in {duration:.2f}s")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        print(f"\nSYSTEM ERROR: {e}")

def test_connection():
    """Verifies connections to external services."""
    print("Testing external connections...")
    
    # 1. Tracing (LangSmith) - Optional if API keys are set
    try:
        print(f"LangSmith: Checking project '{settings.langsmith_project}'...")
        initialize_tracing()
        print("   OK: Tracing initialized")
    except Exception as e:
        print(f"   ERROR: LangSmith failed: {e}")

    # 2. Embeddings (Qwen/DashScope)
    try:
        from src.rag.embeddings import get_embeddings
        print("Embedding provider: sending test query...")
        emb = get_embeddings()
        vec = emb.embed_query("ping")
        print(f"   OK: Success (Vector dim: {len(vec)})")
    except Exception as e:
        print(f"   ERROR: Embedding provider failed: {e}")
        
    print("\nConnection test complete.")

def main():
    parser = argparse.ArgumentParser(description="Notion Agentic RAG System")
    
    # Primary Commands
    parser.add_argument("query", nargs="?", type=str, help="The query to process (if not running a utility command)")
    
    # Utility Flags
    parser.add_argument("--ingest", action="store_true", help="Run document ingestion process")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild during ingestion (use with --ingest)")
    parser.add_argument(
        "--test-conn",
        action="store_true",
        help="Test API connections (LangSmith, LLM/embedding providers)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Handle Verbosity
    if args.verbose:
        settings.debug.log_level = "DEBUG"
    configure_logging(app_name="main")
    
    # Dispatch
    if args.test_conn:
        test_connection()
        return

    if args.ingest:
        from src.ingest import run_ingestion
        print("Starting ingestion pipeline...")
        run_ingestion(rebuild=args.rebuild)
        return

    if args.query:
        # Initialize tracing for the run
        initialize_tracing()
        run_agentic_rag(args.query)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
