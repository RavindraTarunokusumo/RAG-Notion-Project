import logging

from langchain_core.documents import Document

from src.rag.embeddings import get_embeddings
from src.rag.retriever import get_retriever
from src.rag.vectorstore import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_components():
    logger.info("1. Testing Embeddings...")
    embeddings = get_embeddings()
    try:
        vec = embeddings.embed_query("test query")
        logger.info(f"Embedding generated: len={len(vec)}")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return

    logger.info("\n2. Testing VectorStore...")
    vs = get_vector_store()
    doc = Document(
        page_content="RAG systems combine retrieval and generation to improve LLM outputs.", 
        metadata={"source": "test_script"}
    )
    vs.add_documents([doc])
    
    logger.info("\n3. Testing Similarity Search...")
    results = vs.similarity_search("retrieval generation", k=1)
    logger.info(f"Search results: {len(results)}")
    if results:
        logger.info(f"Top result: {results[0].page_content}")
        
    logger.info("\n4. Testing Retriever (with Rerank)...")
    try:
        retriever = get_retriever(use_rerank=True)
        relevant_docs = retriever.invoke("improve LLM outputs")
        logger.info(f"Retriever results: {len(relevant_docs)}")
        if relevant_docs:
            logger.info(f"Top retrieved: {relevant_docs[0].page_content}")
    except Exception as e:
        logger.error(f"Retriever failed: {e}")

if __name__ == "__main__":
    test_rag_components()
