import logging

from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.retrievers import BaseRetriever

from config.settings import settings
from src.rag.vectorstore import get_vector_store

logger = logging.getLogger(__name__)

def get_retriever(use_rerank: bool = True) -> BaseRetriever:
    """
    Factory function for retriever.
    Returns either a vector store retriever or a reranking retriever.
    """
    vector_store = get_vector_store()
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.retrieval_k}
    )
    
    if not use_rerank:
        return base_retriever
        
    logger.info("Initializing Cohere Rerank retriever...")
    try:
        compressor = CohereRerank(
            cohere_api_key=settings.cohere_api_key,
            model="rerank-english-v3.0",
            top_n=settings.rerank_top_n,
            user_agent="notion-agentic-rag"
        )
        
        reranker = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        return reranker
    except Exception as e:
        logger.error(f"Failed to initialize Reranker, falling back to base retriever: {e}")
        return base_retriever

