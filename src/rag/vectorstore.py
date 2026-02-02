import logging
import os
import time

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config.settings import settings
from src.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

def sanitize_metadata(metadata: dict) -> dict:
    """Convert list values in metadata to strings for ChromaDB compatibility."""
    if not metadata:
        return {}
    sanitized = {}
    for k, v in metadata.items():
        if isinstance(v, list):
            sanitized[k] = ", ".join([str(i) for i in v])
        else:
            sanitized[k] = v
    return sanitized

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
        
    def add_documents(self, documents: list[Document], batch_size: int = None, delay: float = None):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add.
            batch_size: Number of documents to process in one batch. Defaults to settings.
            delay: Time in seconds to wait between batches. Defaults to settings.
        """
        batch_size = batch_size or settings.embedding_batch_size
        delay = delay if delay is not None else settings.embedding_delay

        if not documents:
            logger.warning("No documents to add to vector store.")
            return
            
        logger.info(f"Adding {len(documents)} documents to ChromaDB in batches of {batch_size}...")
        
        # Pre-process documents to ensure metadata is compatible
        for doc in documents:
            doc.metadata = sanitize_metadata(doc.metadata)

        total_docs = len(documents)
        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} docs)")
            
            try:
                self.vectorstore.add_documents(batch)
                # Sleep between batches, but not after the last one
                if i + batch_size < total_docs:
                    logger.debug(f"Sleeping for {delay} seconds to respect rate limits...")
                    time.sleep(delay)
            except Exception as e:
                logger.error(f"Failed to add batch {batch_num}: {e}")
                # We raise to stop the process if ingestion fails
                raise e
                
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
        
    def clear(self):
        """Clear the vector store."""
        try:
            logger.info("Clearing Vector Store...")
            # For Chroma, we can delete the collection or get all IDs and delete them
            # Deleting collection and recreating is cleaner
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info("Vector Store cleared.")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")

# Singleton instance
_vector_store_manager: VectorStoreManager | None = None

def get_vector_store() -> VectorStoreManager:
    """Get the singleton vector store manager."""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager
