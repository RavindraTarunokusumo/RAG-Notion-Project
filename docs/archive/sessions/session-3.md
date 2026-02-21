## Session 3: Vector Store & Embeddings

**Session Token Budget:** ~80,000 tokens  
**Focus:** Cohere embeddings, ChromaDB setup, and retrieval with reranking

---

#### NRAG-012: Cohere Embeddings Configuration

**Priority:** P0 - Critical  
**Token Estimate:** 8,000 tokens  
**Status:** To Do

**Description:**  
Configure Cohere embeddings for document vectorization.

**Acceptance Criteria:**

- [x] `CohereEmbeddings` wrapper configured
- [x] Embed model selection (`embed-english-v3.0`)
- [x] Batch embedding support
- [x] Error handling for API failures

**Implementation:**

```python
# src/rag/embeddings.py
import logging
from typing import List
from langchain_cohere import CohereEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Manages document and query embeddings using Cohere.
    """
    
    def __init__(self, model: str = "embed-english-v3.0"):
        self.model = model
        self.embeddings = CohereEmbeddings(
            model=model,
            cohere_api_key=settings.cohere_api_key
        )
        logger.info(f"Initialized Cohere embeddings with model: {model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts."""
        logger.info(f"Embedding {len(texts)} documents...")
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        return self.embeddings.embed_query(query)
    
    def get_embeddings_model(self) -> CohereEmbeddings:
        """Get the underlying embeddings model for use with vectorstores."""
        return self.embeddings


def get_embeddings() -> CohereEmbeddings:
    """Factory function for embeddings."""
    return EmbeddingService().get_embeddings_model()
```

---

#### NRAG-013: ChromaDB Vector Store Setup

**Priority:** P0 - Critical  
**Token Estimate:** 12,000 tokens  
**Status:** To Do

**Description:**  
Configure ChromaDB for persistent vector storage and retrieval.

**Acceptance Criteria:**

- [x] Persistent ChromaDB configuration
- [x] Collection management (create, update, delete)
- [x] Document addition with metadata
- [x] Similarity search functionality
- [x] Collection statistics and health checks

**Implementation:**

```python
# src/rag/vectorstore.py
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from src.rag.embeddings import get_embeddings
from config.settings import settings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages ChromaDB vector store operations.
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None
    ):
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.embeddings = get_embeddings()
        self._vectorstore: Optional[Chroma] = None
    
    @property
    def vectorstore(self) -> Chroma:
        """Lazy initialization of vectorstore."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        return self._vectorstore
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Returns:
            List of document IDs
        """
        logger.info(f"Adding {len(documents)} documents to collection '{self.collection_name}'")
        
        ids = self.vectorstore.add_documents(documents)
        
        logger.info(f"Successfully added {len(ids)} documents")
        return ids
    
    def create_from_documents(self, documents: List[Document]) -> "VectorStoreManager":
        """
        Create a new vector store from documents (replaces existing).
        """
        logger.info(f"Creating new collection '{self.collection_name}' with {len(documents)} documents")
        
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        return self
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: dict = None
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results (default from settings)
            filter: Metadata filter
        
        Returns:
            List of relevant documents
        """
        k = k or settings.retrieval_k
        
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        
        logger.debug(f"Found {len(results)} documents for query")
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = None
    ) -> List[tuple[Document, float]]:
        """Similarity search with relevance scores."""
        k = k or settings.retrieval_k
        
        return self.vectorstore.similarity_search_with_score(query=query, k=k)
    
    def as_retriever(self, **kwargs):
        """Get as a LangChain retriever."""
        search_kwargs = {
            "k": kwargs.pop("k", settings.retrieval_k)
        }
        search_kwargs.update(kwargs.get("search_kwargs", {}))
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        collection = self.vectorstore._collection
        
        return {
            "name": self.collection_name,
            "count": collection.count(),
            "persist_directory": self.persist_directory
        }
    
    def delete_collection(self):
        """Delete the entire collection."""
        logger.warning(f"Deleting collection '{self.collection_name}'")
        self.vectorstore.delete_collection()
        self._vectorstore = None


# Singleton instance
_vector_store_manager: Optional[VectorStoreManager] = None

def get_vector_store() -> VectorStoreManager:
    """Get the singleton vector store manager."""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager
```

---

#### NRAG-014: Cohere Rerank Integration

**Priority:** P1 - High  
**Token Estimate:** 8,000 tokens  
**Status:** To Do

**Description:**  
Integrate Cohere Rerank for improved retrieval relevance.

**Acceptance Criteria:**

- [x] `CohereRerank` compressor configured
- [x] `ContextualCompressionRetriever` setup
- [x] Configurable `top_n` parameter
- [x] Fallback to standard retrieval if rerank fails

**Implementation:**

```python
# src/rag/retriever.py
import logging
from typing import List
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from src.rag.vectorstore import get_vector_store
from config.settings import settings

logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Retriever with Cohere Rerank for improved relevance.
    """
    
    def __init__(
        self,
        use_rerank: bool = True,
        rerank_top_n: int = None,
        retrieval_k: int = None
    ):
        self.use_rerank = use_rerank
        self.rerank_top_n = rerank_top_n or settings.rerank_top_n
        self.retrieval_k = retrieval_k or settings.retrieval_k
        
        self.vector_store = get_vector_store()
        self.base_retriever = self.vector_store.as_retriever(k=self.retrieval_k)
        
        if use_rerank:
            self.reranker = CohereRerank(
                model="rerank-english-v3.0",
                cohere_api_key=settings.cohere_api_key,
                top_n=self.rerank_top_n
            )
            
            self.retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=self.base_retriever
            )
            logger.info(f"Initialized retriever with Cohere Rerank (top_n={self.rerank_top_n})")
        else:
            self.retriever = self.base_retriever
            logger.info("Initialized retriever without reranking")
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
        
        Returns:
            List of relevant documents (reranked if enabled)
        """
        try:
            docs = self.retriever.invoke(query)
            logger.debug(f"Retrieved {len(docs)} documents for query")
            return docs
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            # Fallback to base retriever
            if self.use_rerank:
                logger.info("Falling back to base retriever")
                return self.base_retriever.invoke(query)
            raise
    
    def retrieve_for_tasks(self, tasks: List[dict]) -> List[Document]:
        """
        Retrieve documents for multiple sub-tasks.
        
        Args:
            tasks: List of task dictionaries with 'task' key
        
        Returns:
            Deduplicated list of documents from all tasks
        """
        from src.utils.helpers import deduplicate_documents
        
        all_docs = []
        for task in tasks:
            task_query = task.get("task", str(task))
            docs = self.retrieve(task_query)
            all_docs.extend(docs)
        
        unique_docs = deduplicate_documents(all_docs)
        logger.info(f"Retrieved {len(unique_docs)} unique documents for {len(tasks)} tasks")
        
        return unique_docs


def get_retriever(use_rerank: bool = True) -> RAGRetriever:
    """Factory function for retriever."""
    return RAGRetriever(use_rerank=use_rerank)
```

---

#### NRAG-015: Ingestion Integration Script

**Priority:** P0 - Critical  
**Token Estimate:** 6,000 tokens  
**Status:** To Do

**Description:**  
Create a complete ingestion script that ties together the pipeline and vector store.

**Acceptance Criteria:**

- [x] Single command to run full ingestion
- [x] Progress reporting
- [x] Option to rebuild or update collection
- [x] Summary statistics after completion

**Implementation:**

```python
# src/ingest.py
import logging
import argparse
from datetime import datetime

from src.loaders.pipeline import DocumentPipeline
from src.rag.vectorstore import VectorStoreManager
from src.utils.tracing import initialize_tracing
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_ingestion(rebuild: bool = False):
    """
    Run the complete document ingestion pipeline.
    
    Args:
        rebuild: If True, delete existing collection and rebuild
    """
    initialize_tracing()
    
    start_time = datetime.now()
    logger.info("="*60)
    logger.info("NOTION AGENTIC RAG - Document Ingestion")
    logger.info(f"Started at: {start_time.isoformat()}")
    logger.info("="*60)
    
    # Initialize vector store manager
    vs_manager = VectorStoreManager()
    
    # Check existing collection
    try:
        stats = vs_manager.get_collection_stats()
        logger.info(f"Existing collection: {stats['count']} documents")
        
        if rebuild and stats['count'] > 0:
            logger.warning("Rebuild requested - deleting existing collection...")
            vs_manager.delete_collection()
            vs_manager = VectorStoreManager()  # Reinitialize
    except Exception:
        logger.info("No existing collection found")
    
    # Run document pipeline
    logger.info("\n--- Running Document Pipeline ---")
    pipeline = DocumentPipeline()
    chunks = pipeline.run(include_non_arxiv=True)
    
    if not chunks:
        logger.error("No documents to ingest!")
        return
    
    # Add to vector store
    logger.info("\n--- Adding to Vector Store ---")
    vs_manager.create_from_documents(chunks)
    
    # Final stats
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    final_stats = vs_manager.get_collection_stats()
    pipeline_stats = pipeline.get_stats()
    
    logger.info("\n" + "="*60)
    logger.info("INGESTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Notion entries processed: {pipeline_stats['notion_entries_loaded']}")
    logger.info(f"Arxiv papers fetched: {pipeline_stats['arxiv_papers_fetched']}")
    logger.info(f"Total chunks created: {pipeline_stats['total_chunks']}")
    logger.info(f"Documents in collection: {final_stats['count']}")
    logger.info("="*60)
    
    return final_stats


def main():
    parser = argparse.ArgumentParser(description="Document Ingestion for Notion Agentic RAG")
    parser.add_argument(
        "--rebuild", 
        action="store_true",
        help="Delete existing collection and rebuild from scratch"
    )
    
    args = parser.parse_args()
    run_ingestion(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
```

---

### Session 3 Summary

| Item | Token Estimate | Cumulative |
|------|----------------|------------|
| NRAG-012: Cohere Embeddings | 8,000 | 8,000 |
| NRAG-013: ChromaDB Setup | 12,000 | 20,000 |
| NRAG-014: Cohere Rerank | 8,000 | 28,000 |
| NRAG-015: Ingestion Script | 6,000 | 34,000 |
| **Session Buffer** | ~46,000 | 80,000 |

**Buffer Use:** Testing embeddings, debugging vectorstore, integration testing

---

