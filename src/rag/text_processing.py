import logging

from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles text processing and splitting strategies for RAG documents.
    Supports specialized splitting for academic papers and markdown content.
    """
    
    def __init__(self):
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Headers to split on for Markdown, useful for Notion pages
        self.markdown_headers = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
    def process_documents(self, documents: list[Document]) -> list[Document]:
        """
        Process and split a list of documents.
        Applies appropriate splitting strategy based on document type/content.
        """
        logger.info(f"Processing {len(documents)} documents for splitting")
        all_splits = []
        
        for doc in documents:
            splits = self._split_single_document(doc)
            all_splits.extend(splits)
            
        logger.info(f"Generated {len(all_splits)} total splits")
        return all_splits
    
    def _split_single_document(self, doc: Document) -> list[Document]:
        """Split a single document with metadata preservation."""
        
        # Strategy 1: If it's a Notion page (Markdown-like), try Markdown splitting first
        # Note: NotionLoader content usually comes as unstructured text unless specific markdown export used.
        # But if we assume it has headers:
        if doc.metadata.get("source") == "notion":
            return self._split_markdown(doc)
            
        # Strategy 2: Academic Papers (Arxiv)
        # Often huge, might benefit from larger chunks or section-based splitting if we had it.
        # For now, RecursiveCharacterTextSplitter is robust enough.
        return self.default_splitter.split_documents([doc])
        
    def _split_markdown(self, doc: Document) -> list[Document]:
        """Split markdown content by headers, then by size."""
        try:
            # 1. Split by headers
            md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.markdown_headers)
            header_splits = md_splitter.split_text(doc.page_content)
            
            # 2. Merge metadata from original doc
            for split in header_splits:
                # Update with original metadata, but don't overwrite header info
                # header_splits have metadata like {'Header 1': 'Title'}
                original_meta = doc.metadata.copy()
                original_meta.update(split.metadata)
                split.metadata = original_meta
                
            # 3. Further split by character limit if chunks are still too big
            return self.default_splitter.split_documents(header_splits)
            
        except Exception as e:
            logger.warning(f"Markdown splitting failed for doc {doc.metadata.get('title')}: {e}. Falling back to default.")
            return self.default_splitter.split_documents([doc])
