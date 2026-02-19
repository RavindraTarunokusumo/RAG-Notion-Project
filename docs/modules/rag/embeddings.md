# src/rag/embeddings.py

## Purpose
Cohere embeddings factory used by vectorstore and retrieval components.

## Main responsibilities
- Instantiate `CohereEmbeddings` with project settings.
- Expose a simple factory for callers.

## Key symbols
- `EmbeddingService`
- `get_embeddings()`

## Defaults
- Model: `embed-english-v3.0`
- User agent: `notion-agentic-rag`
