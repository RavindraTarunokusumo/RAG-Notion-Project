# src/rag/retriever.py

## Purpose
Constructs document retrievers from the vector store, optionally with Cohere reranking.

## Main responsibilities
- Build base retriever from Chroma via `as_retriever`.
- Optionally wrap with `ContextualCompressionRetriever` + `CohereRerank`.
- Fall back to base retriever if reranker initialization fails.

## Key function
- `get_retriever(use_rerank=True)`

## Dependencies
- `src.rag.vectorstore.get_vector_store`
- `langchain.retrievers.ContextualCompressionRetriever`
- `langchain_cohere.CohereRerank`
