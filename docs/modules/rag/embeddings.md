# src/rag/embeddings.py

## Purpose
DashScope embeddings factory used by vectorstore and retrieval components.

## Main responsibilities
- Instantiate `DashScopeEmbeddings` using configured provider/model.
- Expose a simple factory for callers.

## Key symbols
- `EmbeddingService`
- `get_embeddings()`

## Defaults
- Provider: `qwen`
- Model: `text-embedding-v4`
