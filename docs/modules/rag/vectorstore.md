# src/rag/vectorstore.py

## Purpose
Provides ChromaDB vector store management for document ingestion and retrieval.

## Main responsibilities
- Initialize persisted Chroma collection with configured embeddings.
- Add documents in batches with configurable delay.
- Sanitize metadata for Chroma compatibility.
- Expose retriever/search APIs.
- Clear and recreate collection when requested.

## Key symbols
- `sanitize_metadata(metadata)`: converts list metadata values to strings.
- `VectorStoreManager`: wrapper around Chroma operations.
- `get_vector_store()`: singleton accessor.

## Public methods
- `add_documents(documents, batch_size=None, delay=None)`
- `get_retriever(k=10, search_type="similarity")`
- `as_retriever(**kwargs)`
- `similarity_search(query, k=5)`
- `clear()`

## Dependencies
- `langchain_community.vectorstores.Chroma`
- `src.rag.embeddings.get_embeddings`
- runtime config from `config.settings`
