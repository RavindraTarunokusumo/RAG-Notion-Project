# config/settings.py

## Purpose
Centralized runtime configuration using Pydantic settings loaded from environment and `.env`.

## Main responsibilities
- Define tool-agent feature flags (`ToolAgentConfig`).
- Define per-agent Cohere model and temperature defaults (`CohereModelConfig`).
- Define API keys, LangSmith settings, vector store settings, and RAG tunables (`Settings`).
- Instantiate a singleton `settings` object at import time.

## Key models
- `ToolAgentConfig`
- `CohereModelConfig`
- `Settings`

## Important fields
- API/auth: `cohere_api_key`, `notion_token`, `notion_database_id`, `langsmith_api_key`
- Tracing: `langsmith_tracing`, `langsmith_project`, `langsmith_endpoint`
- Retrieval: `chunk_size`, `chunk_overlap`, `retrieval_k`, `rerank_top_n`
- Vector store: `chroma_persist_dir`, `collection_name`, `embedding_batch_size`, `embedding_delay`
- Nested: `models`, `tool_agents`
