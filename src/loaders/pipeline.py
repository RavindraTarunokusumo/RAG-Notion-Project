import logging
from dataclasses import dataclass, field
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import settings
from src.loaders.arxiv_loader import ArxivPaperLoader
from src.loaders.notion_loader import NotionEntry, NotionKnowledgeBaseLoader

logger = logging.getLogger(__name__)

@dataclass
class PipelineStats:
    """Statistics from document pipeline execution."""
    notion_entries_loaded: int = 0
    arxiv_papers_found: int = 0
    arxiv_papers_fetched: int = 0
    non_arxiv_entries: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    errors: list[str] = field(default_factory=list)

class DocumentPipeline:
    """
    Orchestrates the full document ingestion pipeline:
    
    1. Load metadata & links from Notion KB
    2. Identify Arxiv papers and fetch full content
    3. Process non-Arxiv entries (use Notion notes as content)
    4. Split documents into chunks
    5. Prepare for embedding
    """
    
    def __init__(self):
        self.notion_loader = NotionKnowledgeBaseLoader()
        self.arxiv_loader = ArxivPaperLoader()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.stats = PipelineStats()
    
    def run(self, include_non_arxiv: bool = True) -> list[Document]:
        """
        Run the full pipeline.
        
        Returns:
            List[Document]: Chunked documents ready for vector store.
        """
        logger.info("Starting Document Pipeline...")
        all_documents = []
        
        try:
            # 1. Load from Notion
            notion_entries = self.notion_loader.load_entries(use_cache=False)
            self.stats.notion_entries_loaded = len(notion_entries)
            
            # 2. Separate Arxiv vs Non-Arxiv
            arxiv_entries = [e for e in notion_entries if e.arxiv_id]
            non_arxiv_entries = [e for e in notion_entries if not e.arxiv_id]
            
            self.stats.arxiv_papers_found = len(arxiv_entries)
            self.stats.non_arxiv_entries = len(non_arxiv_entries)
            
            # 3. Fetch Arxiv Papers
            if arxiv_entries:
                arxiv_docs = self.arxiv_loader.load_papers_from_notion_entries(
                    arxiv_entries, 
                    include_full_text=True
                )
                self.stats.arxiv_papers_fetched = len(arxiv_docs)
                all_documents.extend(arxiv_docs)
            
            # 4. Process Non-Arxiv Entries (if requested)
            if include_non_arxiv and non_arxiv_entries:
                non_arxiv_docs = self._process_non_arxiv_entries(non_arxiv_entries)
                all_documents.extend(non_arxiv_docs)
            
            self.stats.total_documents = len(all_documents)
            logger.info(f"Collected value {len(all_documents)} total documents for processing")
            
            # 5. Split Text
            chunks = self.text_splitter.split_documents(all_documents)
            self.stats.total_chunks = len(chunks)
            logger.info(f"Generated {len(chunks)} text chunks")
            
            return chunks
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
            raise

    def _process_non_arxiv_entries(self, entries: list[NotionEntry]) -> list[Document]:
        """Convert basic Notion entries to LangChain Documents."""
        docs = []
        for entry in entries:
            # Skip empty entries
            if not entry.notes and not entry.title:
                continue
                
            # Flatten keywords for content inclusion
            kw_str = ", ".join(entry.keywords) if isinstance(entry.keywords, list) else str(entry.keywords)
            
            content = f"TITLE: {entry.title}\nTOPIC: {entry.topic}\nKEYWORDS: {kw_str}\n\nNOTES:\n{entry.notes}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "notion",
                    "notion_id": entry.notion_id,
                    "title": entry.title,
                    "topic": entry.topic,
                    "keywords": entry.keywords,
                    "url": entry.source_url,
                    "category": entry.entry_type,
                    "publication_date": entry.publication_date
                }
            )
            docs.append(doc)
        return docs
    
    def get_stats(self) -> dict[str, Any]:
        """Get pipeline execution statistics."""
        return {
            "notion_entries": self.stats.notion_entries_loaded,
            "arxiv_found": self.stats.arxiv_papers_found,
            "arxiv_fetched": self.stats.arxiv_papers_fetched,
            "chunks_generated": self.stats.total_chunks,
            "errors": self.stats.errors
        }


def ingest_documents() -> list[Document]:
    """Convenience function for document ingestion."""
    pipeline = DocumentPipeline()
    chunks = pipeline.run()
    
    # Print stats
    stats = pipeline.get_stats()
    print("\nIngestion Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
        
    return chunks
