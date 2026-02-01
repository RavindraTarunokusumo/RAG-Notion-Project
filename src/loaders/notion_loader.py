import logging
from dataclasses import dataclass
from typing import Any

from langchain_community.document_loaders import NotionDBLoader

from config.settings import settings
from src.utils.helpers import extract_arxiv_id

logger = logging.getLogger(__name__)

@dataclass
class NotionEntry:
    """Represents a single entry from the Notion knowledge base."""
    notion_id: str
    title: str
    source_url: str
    arxiv_id: str | None
    categories: list[str]
    topics: list[str]
    notes: str
    created_date: str
    
class NotionKnowledgeBaseLoader:
    """
    Loads metadata and links from Notion AI Knowledge Base.
    
    The Notion KB stores:
    - Article/paper titles
    - Source URLs (including Arxiv links)
    - Categories (e.g., "Research Paper", "Blog Post", "Tutorial")
    - Topics (e.g., "Agent Frameworks", "RAG", "Model Architecture")
    - User notes/summaries
    
    This loader extracts this metadata and identifies Arxiv papers
    for subsequent full-text fetching via ArxivLoader.
    """
    
    def __init__(self):
        self.loader = NotionDBLoader(
            integration_token=settings.notion_token,
            database_id=settings.notion_database_id,
            request_timeout_sec=30
        )
        self._cached_entries: list[NotionEntry] | None = None
    
    def load_entries(self, use_cache: bool = True) -> list[NotionEntry]:
        """Load all entries from Notion knowledge base."""
        if use_cache and self._cached_entries:
            logger.info(f"Returning {len(self._cached_entries)} cached Notion entries")
            return self._cached_entries
            
        logger.info("Loading documents from Notion Database...")
        try:
            documents = self.loader.load()
            logger.info(f"Fetched {len(documents)} documents from Notion")
            
            entries = []
            for doc in documents:
                entry = self._parse_notion_document(doc)
                if entry:
                    entries.append(entry)
            
            self._cached_entries = entries
            logger.info(f"Successfully successfully parsed {len(entries)} Notion entries")
            return entries
            
        except Exception as e:
            logger.error(f"Error loading from Notion: {str(e)}")
            raise

    def _parse_notion_document(self, doc) -> NotionEntry | None:
        """Parse a Notion document into a structured entry."""
        try:
            metadata = doc.metadata
            properties = metadata.get("properties", {})
            
            # Extract common properties - adjust keys based on your Notion DB schema
            # Assuming standard schema structure for properties
             
            # 1. ID and Title
            notion_id = metadata.get("id", "")
            
            # Helper to safely get property content (assuming property type text/title/rich_text)
            title = "Untitled"
            # Since NotionDBLoader might flatten or structure properties differently, 
            # we need to handle potential structures. 
            # Often NotionDBLoader puts main page content in page_content and metadata in metadata
            
            # Note: NotionDBLoader behavior depends on implementation version.
            # Assuming standard behavior where metadata contains raw properties or flattened dict.
            # If flattened:
            # title = metadata.get("Name", "Untitled") or metadata.get("Title", "Untitled")
            
            # Let's try to extract from 'properties' dict if it exists (standard Notion API response structure)
            # or from top-level keys if flattened.
            
            # Strategy: look for common keys
            title = properties.get("Name", properties.get("Title", "Untitled"))
            if isinstance(title, dict): # If it's the raw property object
                 # Simplification: This part depends highly on how NotionDBLoader processes the response.
                 # For safety, let's rely on what we can find.
                 pass

            # Update: NotionDBLoader usually returns a Document where page_content is the page body.
            # Metadata contains database properties.
            # Let's assume the metadata keys correspond to column names.
            
            title = metadata.get("Name") or metadata.get("title") or "Untitled"
            source_url = metadata.get("URL") or metadata.get("Source") or ""
            
            # User notes might be the page content itself or a specific property
            notes = doc.page_content
            
            # Categories and Topics (Multi-select)
            categories = self._parse_multi_select(metadata.get("Category") or metadata.get("Categories"))
            topics = self._parse_multi_select(metadata.get("Topic") or metadata.get("Topics"))
            created_date = metadata.get("Created time") or ""

            # Attempt to extract Arxiv ID
            arxiv_id = extract_arxiv_id(source_url)
            
            return NotionEntry(
                notion_id=notion_id,
                title=title,
                source_url=source_url,
                arxiv_id=arxiv_id,
                categories=categories,
                topics=topics,
                notes=notes,
                created_date=created_date
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse Notion document {doc.metadata.get('id')}: {e}")
            return None
    
    def _parse_multi_select(self, value: Any) -> list[str]:
        """Parse Notion multi-select property."""
        if not value:
            return []
        
        if isinstance(value, str):
            # Sometimes loaders join multi-select with commas
            return [v.strip() for v in value.split(",")]
            
        if isinstance(value, list):
            # If it's a list of strings
            if all(isinstance(v, str) for v in value):
                return value
            # If it's a list of dicts (options)
            return [v.get("name", "") for v in value if isinstance(v, dict)]
            
        return []
    
    def get_arxiv_entries(self) -> list[NotionEntry]:
        """Get only entries that have Arxiv links."""
        entries = self.load_entries()
        arxiv_entries = [e for e in entries if e.arxiv_id]
        logger.info(f"Found {len(arxiv_entries)} entries with Arxiv IDs")
        return arxiv_entries
    
    def get_entries_by_category(self, category: str) -> list[NotionEntry]:
        """Filter entries by category."""
        entries = self.load_entries()
        return [e for e in entries if category in e.categories]
    
    def get_entries_by_topic(self, topic: str) -> list[NotionEntry]:
        """Filter entries by topic."""
        entries = self.load_entries()
        return [e for e in entries if topic in e.topics]
