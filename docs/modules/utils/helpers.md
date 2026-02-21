# src/utils/helpers.py

## Purpose
Utility helpers for arXiv ID parsing, document deduplication, prompt formatting, and resilient JSON parsing.

## Functions
- `extract_arxiv_id(url)`: supports multiple arXiv URL formats and raw ID strings.
- `deduplicate_documents(docs)`: removes duplicates using MD5 hash of `page_content`.
- `format_documents_for_prompt(docs, max_chars=15000)`: creates bounded prompt text.
- `safe_json_parse(text)`: attempts direct JSON parse, markdown block extraction, and object extraction.

## Dependencies
- `re`, `hashlib`, `json`
- `langchain_core.documents.Document`
