# src/loaders/arxiv_loader.py

## Purpose
Fetches arXiv papers and merges paper metadata/content with originating Notion entry metadata.

## Main responsibilities
- Iterate Notion entries with arXiv IDs.
- Fetch paper documents through LangChain `ArxivLoader`.
- Merge Notion metadata into returned paper documents.
- Optionally return full-text content or abstract-focused content.
- Provide direct query-based arXiv lookup utility.

## Key symbols
- `EnrichedDocument`: dataclass schema for enriched paper shape.
- `ArxivPaperLoader`: loader class.

## Core methods
- `load_papers_from_notion_entries(entries, include_full_text=False)`
- `_fetch_single_paper(entry, include_full_text)`
- `load_by_query(query, max_docs=5)`

## Dependencies
- `langchain_community.document_loaders.ArxivLoader`
- `src.loaders.notion_loader.NotionEntry`

## Operational note
- Applies a configurable delay between API calls for rate-limit safety.
