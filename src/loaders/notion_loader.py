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
    topic: str
    keywords: list[str]
    source_url: str
    arxiv_id: str | None
    entry_type: str
    notes: str
    publication_date: str
    
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
            logger.info(f"Successfully parsed {len(entries)} Notion entries")
            return entries
            
        except Exception as e:
            logger.error(f"Error loading from Notion: {str(e)}")
            raise

    def _parse_notion_document(self, doc) -> NotionEntry | None:
        """Parse a Notion document into a structured entry."""
        try:
            # Page content is usually empty; we rely on metadata
            metadata = doc.metadata
            
            notion_id = metadata.get("id", "")
            title = metadata.get("title", "Untitled")
            topic = metadata.get("topics", "")
            keywords = metadata.get("keywords", [])
            url = metadata.get("url", "") 
            publication_date = metadata.get("publication date", "")
            if publication_date:
                publication_date = publication_date.get("start", "")
            entry_type = metadata.get("type", "Unknown")
            notes = metadata.get("notes", "")
            
            # Attempt to extract Arxiv ID
            arxiv_id = extract_arxiv_id(url)
            
            return NotionEntry(
                notion_id=notion_id,
                title=title,
                topic=topic,
                keywords=keywords,
                source_url=url,
                arxiv_id=arxiv_id,
                entry_type=entry_type,
                notes=notes,
                publication_date=publication_date,
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse Notion document {doc.metadata.get('title')}: {e}")
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
