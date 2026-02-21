# src/loaders/pipeline.py

## Purpose
Coordinates end-to-end document ingestion from Notion and arXiv into chunked LangChain documents.

## Main responsibilities
- Load entries from Notion.
- Separate arXiv-linked entries from regular notes.
- Fetch and enrich arXiv documents.
- Convert non-arXiv entries to documents.
- Chunk all documents using recursive text splitting.
- Track ingestion statistics.

## Key symbols
- `PipelineStats`: dataclass for ingestion counters and errors.
- `DocumentPipeline`: orchestration class for ingestion.
- `ingest_documents()`: convenience runner.

## Core methods
- `run(include_non_arxiv=True)`: executes full pipeline and returns chunks.
- `_process_non_arxiv_entries(entries)`: turns Notion-only entries into documents.
- `get_stats()`: returns summary metrics.

## Dependencies
- `src.loaders.notion_loader.NotionKnowledgeBaseLoader`
- `src.loaders.arxiv_loader.ArxivPaperLoader`
- `langchain.text_splitter.RecursiveCharacterTextSplitter`
- runtime config from `config.settings`
