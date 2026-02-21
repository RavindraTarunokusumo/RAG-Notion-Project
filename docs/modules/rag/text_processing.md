# src/rag/text_processing.py

## Purpose
Text chunking strategies for RAG documents, with optional markdown-aware splitting.

## Main responsibilities
- Provide recursive character splitting with configured chunk size/overlap.
- Use markdown header splitting for Notion content when applicable.
- Preserve and merge metadata through splitting.

## Key symbol
- `DocumentProcessor`

## Core methods
- `process_documents(documents)`
- `_split_single_document(doc)`
- `_split_markdown(doc)`

## Dependencies
- `RecursiveCharacterTextSplitter`
- `MarkdownHeaderTextSplitter`
- runtime config from `config.settings`
