import logging
import time
from dataclasses import dataclass

from langchain_community.document_loaders import ArxivLoader
from langchain_core.documents import Document

from src.loaders.notion_loader import NotionEntry

logger = logging.getLogger(__name__)

@dataclass
class EnrichedDocument:
    """Document enriched with both Notion and Arxiv metadata."""
    content: str
    title: str
    arxiv_id: str
    authors: list[str]
    published_date: str
    abstract: str
    pdf_url: str
    # From Notion
    notion_category: str
    notion_keywords: list[str]
    notion_topics: list[str]
    notion_notes: str

class ArxivPaperLoader:
    """
    Fetches full paper content from Arxiv using IDs extracted from Notion.
    
    This loader:
    1. Takes Arxiv IDs from NotionKnowledgeBaseLoader
    2. Fetches full paper abstracts and content
    3. Merges with Notion metadata for enriched documents
    """
    
    def __init__(self, rate_limit_delay: float = 3.0):
        self.rate_limit_delay = rate_limit_delay
    
    def load_papers_from_notion_entries(
        self, 
        entries: list[NotionEntry],
        include_full_text: bool = False
    ) -> list[Document]:
        """
        Load papers corresponding to the given Notion entries.
        Returns LangChain Documents enriched with metadata.
        """
        documents = []
        entries_with_arxiv = [e for e in entries if e.arxiv_id]
        logger.info(f"Processing {len(entries_with_arxiv)} entries with Arxiv IDs")
        
        for entry in entries_with_arxiv:
            if not entry.arxiv_id:
                continue
                
            try:
                # Add delay to respect Arxiv API rate limits
                time.sleep(self.rate_limit_delay)
                
                doc = self._fetch_single_paper(entry, include_full_text)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error fetching Arxiv paper {entry.arxiv_id}: {e}")
                
        return documents
    
    def _fetch_single_paper(
        self, 
        entry: NotionEntry,
        include_full_text: bool
    ) -> Document | None:
        """Fetch a single paper and merge with Notion metadata."""
        try:
            logger.info(f"Fetching Arxiv paper: {entry.arxiv_id}")
            # load_max_docs=1 because arxiv_id should be unique
            loader = ArxivLoader(
                query=entry.arxiv_id, 
                load_max_docs=1,
                load_all_available_meta=True
            )
            
            # ArxivLoader returns a list of Documents
            docs = loader.load()
            
            if not docs:
                logger.warning(f"No paper found for ID: {entry.arxiv_id}")
                return None
                
            paper_doc = docs[0]
            
            # Merge Metadata
            # Notion metadata takes precedence for user-defined categorization
            paper_doc.metadata.update({
                "source": "arxiv",
                "title": paper_doc.metadata.get("Title", entry.title),
                "arxiv_id": entry.arxiv_id,
                "notion_id": entry.notion_id,
                "notion_title": entry.title,
                "category": entry.entry_type,
                "topic": entry.topic,
                "keywords": entry.keywords,
                "user_notes": entry.notes,
                "notion_url": entry.source_url,
                "publication_date": entry.publication_date
            })
            
            # Prepare content enrichment strings
            kw_str = ", ".join(entry.keywords) if isinstance(entry.keywords, list) else str(entry.keywords)
            metadata_header = f"TOPIC: {entry.topic}\nKEYWORDS: {kw_str}"

            # If we don't want full text, we might just want abstract + notes
            if not include_full_text:
                # Use abstract as content if full text not requested
                abstract = paper_doc.metadata.get("Summary", "")
                paper_doc.page_content = f"ABSTRACT:\n{abstract}\n\n{metadata_header}\n\nUSER NOTES:\n{entry.notes}"
            else:
                 # Prepend notes to content
                 paper_doc.page_content = f"{metadata_header}\n\nUSER NOTES:\n{entry.notes}\n\nPAPER CONTENT:\n{paper_doc.page_content}"
            
            return paper_doc
            
        except Exception as e:
            logger.error(f"Failed to fetch {entry.arxiv_id}: {str(e)}")
            return None
    
    def load_by_query(self, query: str, max_docs: int = 5) -> list[Document]:
        """
        Direct search on Arxiv (useful for Planner/Researcher later).
        """
        try:
            loader = ArxivLoader(query=query, load_max_docs=max_docs)
            return loader.load()
        except Exception as e:
            logger.error(f"Arxiv search failed: {e}")
            return []
