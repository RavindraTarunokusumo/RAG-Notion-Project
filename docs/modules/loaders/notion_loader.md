# src/loaders/notion_loader.py

## Purpose
Loads and parses records from Notion DB into structured entries used by ingestion.

## Main responsibilities
- Fetch Notion documents via `NotionDBLoader`.
- Parse loader metadata into `NotionEntry` objects.
- Detect arXiv IDs from source URLs.
- Cache parsed entries for repeated reads.

## Key symbols
- `NotionEntry`: dataclass for normalized Notion metadata.
- `NotionKnowledgeBaseLoader`: loader class.

## Core methods
- `load_entries(use_cache=True)`
- `_parse_notion_document(doc)`
- `get_arxiv_entries()`
- `get_entries_by_category(category)`
- `get_entries_by_topic(topic)`

## Dependencies
- `langchain_community.document_loaders.NotionDBLoader`
- `src.utils.helpers.extract_arxiv_id`
- runtime config from `config.settings`
