# src/ingest.py

## Purpose
Entrypoint for ingestion: run document pipeline and write resulting chunks into the vector store.

## Main responsibilities
- Optionally clear vector store when `rebuild=True`.
- Execute `DocumentPipeline.run()`.
- Add chunked documents to vector store with batching logic delegated to vectorstore manager.

## Key function
- `run_ingestion(rebuild=False)`

## Dependencies
- `src.loaders.pipeline.DocumentPipeline`
- `src.rag.vectorstore.get_vector_store`
