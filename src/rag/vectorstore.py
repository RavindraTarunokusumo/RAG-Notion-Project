import logging
import os

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config.settings import settings
from src.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages the ChromaDB vector store interactions.
    """
    def __init__(self):
        self.persist_directory = settings.chroma_persist_dir
        self.collection_name = settings.collection_name
        self.embeddings = get_embeddings()
        
        # Ensure the directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
    def add_documents(self, documents: list[Document]):
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents to add to vector store.")
            return
            
        logger.info(f"Adding {len(documents)} documents to ChromaDB...")
        self.vectorstore.add_documents(documents)
        logger.info("Documents added successfully.")

    def get_retriever(self, k: int = 10, search_type: str = "similarity"):
        """Get a retriever from the vector store."""
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    def as_retriever(self, **kwargs):
        """Pass through to inner vectorstore."""
        return self.vectorstore.as_retriever(**kwargs)
        
    def similarity_search(self, query: str, k: int = 5):
        return self.vectorstore.similarity_search(query, k=k)

# Singleton instance
_vector_store_manager: VectorStoreManager | None = None

def get_vector_store() -> VectorStoreManager:
    """Get the singleton vector store manager."""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager
