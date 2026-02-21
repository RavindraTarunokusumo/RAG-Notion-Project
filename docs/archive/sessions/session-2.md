## Session 2: Notion Integration & Document Pipeline

**Session Token Budget:** ~80,000 tokens  
**Focus:** Notion document loading, Arxiv paper fetching, and metadata merging

---

#### NRAG-008: Notion Loader Implementation

**Priority:** P0 - Critical  
**Token Estimate:** 12,000 tokens  
**Status:** To Do

**Description:**  
Implement Notion document loader that extracts metadata and Arxiv links from the knowledge base.

**Important Clarification:**  
The Notion knowledge base stores **links and metadata only**, not full paper content. The loader must:

1. Load all entries from Notion with their metadata (title, categories, tags, source URL)
2. Extract Arxiv links/IDs from each entry
3. Return structured data for downstream Arxiv fetching

**Acceptance Criteria:**

- [x] `NotionDBLoader` configured with authentication
- [x] Metadata extraction (title, categories, topics, source URL)
- [x] Arxiv link identification and ID extraction
- [x] Error handling for API rate limits
- [x] Caching of loaded metadata to avoid repeated API calls

**Implementation:**

```python
# src/loaders/notion_loader.py
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain_community.document_loaders import NotionDBLoader
from src.utils.helpers import extract_arxiv_id
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class NotionEntry:
    """Represents a single entry from the Notion knowledge base."""
    notion_id: str
    title: str
    source_url: str
    arxiv_id: str | None
    categories: List[str]
    topics: List[str]
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
        self._cached_entries: List[NotionEntry] | None = None
    
    def load_entries(self, use_cache: bool = True) -> List[NotionEntry]:
        """Load all entries from Notion knowledge base."""
        if use_cache and self._cached_entries is not None:
            logger.info(f"Using cached entries: {len(self._cached_entries)} items")
            return self._cached_entries
        
        logger.info("Loading entries from Notion...")
        raw_docs = self.loader.load()
        logger.info(f"Loaded {len(raw_docs)} raw documents from Notion")
        
        entries = []
        for doc in raw_docs:
            entry = self._parse_notion_document(doc)
            if entry:
                entries.append(entry)
        
        self._cached_entries = entries
        logger.info(f"Parsed {len(entries)} valid entries")
        
        return entries
    
    def _parse_notion_document(self, doc) -> NotionEntry | None:
        """Parse a Notion document into a structured entry."""
        try:
            metadata = doc.metadata
            
            # Extract source URL from content or metadata
            source_url = metadata.get("source", "") or metadata.get("url", "")
            
            # Try to extract Arxiv ID if it's an Arxiv link
            arxiv_id = extract_arxiv_id(source_url)
            
            return NotionEntry(
                notion_id=metadata.get("id", ""),
                title=metadata.get("title", "Untitled"),
                source_url=source_url,
                arxiv_id=arxiv_id,
                categories=self._parse_multi_select(metadata.get("Type", [])),
                topics=self._parse_multi_select(metadata.get("Topics", [])),
                notes=doc.page_content,
                created_date=metadata.get("created_time", "")
            )
        except Exception as e:
            logger.warning(f"Failed to parse Notion document: {e}")
            return None
    
    def _parse_multi_select(self, value: Any) -> List[str]:
        """Parse Notion multi-select property."""
        if isinstance(value, list):
            return [item.get("name", str(item)) if isinstance(item, dict) else str(item) 
                    for item in value]
        elif isinstance(value, str):
            return [value]
        return []
    
    def get_arxiv_entries(self) -> List[NotionEntry]:
        """Get only entries that have Arxiv links."""
        entries = self.load_entries()
        arxiv_entries = [e for e in entries if e.arxiv_id is not None]
        logger.info(f"Found {len(arxiv_entries)} entries with Arxiv links")
        return arxiv_entries
    
    def get_entries_by_category(self, category: str) -> List[NotionEntry]:
        """Filter entries by category."""
        entries = self.load_entries()
        return [e for e in entries if category in e.categories]
    
    def get_entries_by_topic(self, topic: str) -> List[NotionEntry]:
        """Filter entries by topic."""
        entries = self.load_entries()
        return [e for e in entries if topic in e.topics]
```

---

#### NRAG-009: Arxiv Loader for Full Paper Content

**Priority:** P0 - Critical  
**Token Estimate:** 10,000 tokens  
**Status:** To Do

**Description:**  
Implement Arxiv loader that fetches full paper content using IDs extracted from Notion entries.

**Workflow:**

1. Receive list of Arxiv IDs from Notion loader
2. Fetch full paper content and metadata via ArxivLoader
3. Merge Notion metadata (user tags, categories) with Arxiv data
4. Return enriched documents ready for embedding

**Acceptance Criteria:**

- [x] Batch fetching of papers by Arxiv ID
- [x] Full abstract and paper content extraction
- [x] Metadata merging (Notion categories + Arxiv metadata)
- [x] Rate limiting to avoid API blocks
- [x] Graceful handling of unavailable papers

**Implementation:**

```python
# src/loaders/arxiv_loader.py
import logging
import time
from typing import List
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
    authors: List[str]
    published_date: str
    abstract: str
    pdf_url: str
    # From Notion
    notion_categories: List[str]
    notion_topics: List[str]
    notion_notes: str

class ArxivPaperLoader:
    """
    Fetches full paper content from Arxiv using IDs extracted from Notion.
    
    This loader:
    1. Takes Arxiv IDs from NotionKnowledgeBaseLoader
    2. Fetches full paper abstracts and content
    3. Merges with Notion metadata for enriched documents
    """
    
    def __init__(self, rate_limit_delay: float = 1.0):
        self.rate_limit_delay = rate_limit_delay
    
    def load_papers_from_notion_entries(
        self, 
        notion_entries: List[NotionEntry],
        include_full_text: bool = False
    ) -> List[Document]:
        """
        Load Arxiv papers based on Notion entries.
        
        Args:
            notion_entries: Entries with arxiv_id populated
            include_full_text: If True, attempts to load full PDF text (slower)
        
        Returns:
            List of LangChain Documents with merged metadata
        """
        documents = []
        arxiv_entries = [e for e in notion_entries if e.arxiv_id]
        
        logger.info(f"Fetching {len(arxiv_entries)} papers from Arxiv...")
        
        for i, entry in enumerate(arxiv_entries):
            try:
                logger.debug(f"Fetching paper {i+1}/{len(arxiv_entries)}: {entry.arxiv_id}")
                
                doc = self._fetch_single_paper(entry, include_full_text)
                if doc:
                    documents.append(doc)
                
                # Rate limiting
                if i < len(arxiv_entries) - 1:
                    time.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {entry.arxiv_id}: {e}")
                continue
        
        logger.info(f"Successfully fetched {len(documents)} papers")
        return documents
    
    def _fetch_single_paper(
        self, 
        notion_entry: NotionEntry,
        include_full_text: bool
    ) -> Document | None:
        """Fetch a single paper and merge with Notion metadata."""
        try:
            loader = ArxivLoader(
                query=notion_entry.arxiv_id,
                load_max_docs=1,
                load_all_available_meta=True
            )
            
            arxiv_docs = loader.load()
            
            if not arxiv_docs:
                logger.warning(f"No content found for {notion_entry.arxiv_id}")
                return None
            
            arxiv_doc = arxiv_docs[0]
            
            # Merge metadata from both sources
            merged_metadata = {
                # Arxiv metadata
                "source": "arxiv",
                "arxiv_id": notion_entry.arxiv_id,
                "title": arxiv_doc.metadata.get("Title", notion_entry.title),
                "authors": arxiv_doc.metadata.get("Authors", []),
                "published": arxiv_doc.metadata.get("Published", ""),
                "abstract": arxiv_doc.metadata.get("Summary", ""),
                "pdf_url": f"https://arxiv.org/pdf/{notion_entry.arxiv_id}.pdf",
                "arxiv_url": f"https://arxiv.org/abs/{notion_entry.arxiv_id}",
                "primary_category": arxiv_doc.metadata.get("Primary Category", ""),
                
                # Notion metadata (user-provided context)
                "notion_id": notion_entry.notion_id,
                "notion_categories": notion_entry.categories,
                "notion_topics": notion_entry.topics,
                "notion_notes": notion_entry.notes,
                "user_title": notion_entry.title,  # User's title might differ
            }
            
            return Document(
                page_content=arxiv_doc.page_content,
                metadata=merged_metadata
            )
            
        except Exception as e:
            logger.error(f"Error fetching {notion_entry.arxiv_id}: {e}")
            return None
    
    def load_by_query(self, query: str, max_docs: int = 5) -> List[Document]:
        """
        Load papers by search query (for enrichment/discovery).
        
        This can be used to find related papers not yet in the Notion KB.
        """
        loader = ArxivLoader(
            query=query,
            load_max_docs=max_docs,
            load_all_available_meta=True
        )
        
        docs = loader.load()
        
        # Add standard metadata structure
        for doc in docs:
            doc.metadata["source"] = "arxiv_search"
            doc.metadata["search_query"] = query
            doc.metadata["notion_categories"] = []
            doc.metadata["notion_topics"] = []
            doc.metadata["notion_notes"] = ""
        
        return docs
```

---

#### NRAG-010: Document Pipeline Orchestration

**Priority:** P0 - Critical  
**Token Estimate:** 10,000 tokens  
**Status:** To Do

**Description:**  
Create the unified document ingestion pipeline that orchestrates Notion â†’ Arxiv â†’ Processing flow.

**Acceptance Criteria:**

- [x] Single entry point for document ingestion
- [x] Configurable pipeline stages
- [x] Progress logging and metrics
- [x] Support for incremental updates

**Implementation:**

```python
# src/loaders/pipeline.py
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.loaders.notion_loader import NotionKnowledgeBaseLoader, NotionEntry
from src.loaders.arxiv_loader import ArxivPaperLoader
from config.settings import settings

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
    errors: List[str] = field(default_factory=list)

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
        self.arxiv_loader = ArxivPaperLoader(rate_limit_delay=1.5)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.stats = PipelineStats()
    
    def run(self, include_non_arxiv: bool = True) -> List[Document]:
        """
        Execute the full document pipeline.
        
        Args:
            include_non_arxiv: Include Notion entries without Arxiv links
                              (using notes as content)
        
        Returns:
            List of chunked Documents ready for embedding
        """
        logger.info("="*50)
        logger.info("Starting Document Pipeline")
        logger.info("="*50)
        
        # Step 1: Load from Notion
        logger.info("Step 1: Loading entries from Notion KB...")
        notion_entries = self.notion_loader.load_entries()
        self.stats.notion_entries_loaded = len(notion_entries)
        
        # Step 2: Separate Arxiv and non-Arxiv entries
        arxiv_entries = [e for e in notion_entries if e.arxiv_id]
        non_arxiv_entries = [e for e in notion_entries if not e.arxiv_id]
        
        self.stats.arxiv_papers_found = len(arxiv_entries)
        self.stats.non_arxiv_entries = len(non_arxiv_entries)
        
        logger.info(f"  - Arxiv papers: {len(arxiv_entries)}")
        logger.info(f"  - Non-Arxiv entries: {len(non_arxiv_entries)}")
        
        all_documents = []
        
        # Step 3: Fetch Arxiv papers
        logger.info("Step 2: Fetching Arxiv papers...")
        arxiv_docs = self.arxiv_loader.load_papers_from_notion_entries(arxiv_entries)
        self.stats.arxiv_papers_fetched = len(arxiv_docs)
        all_documents.extend(arxiv_docs)
        
        # Step 4: Process non-Arxiv entries (optional)
        if include_non_arxiv:
            logger.info("Step 3: Processing non-Arxiv entries...")
            non_arxiv_docs = self._process_non_arxiv_entries(non_arxiv_entries)
            all_documents.extend(non_arxiv_docs)
        
        self.stats.total_documents = len(all_documents)
        
        # Step 5: Chunk documents
        logger.info("Step 4: Chunking documents...")
        chunks = self.text_splitter.split_documents(all_documents)
        self.stats.total_chunks = len(chunks)
        
        logger.info("="*50)
        logger.info("Pipeline Complete!")
        logger.info(f"  Total chunks: {len(chunks)}")
        logger.info("="*50)
        
        return chunks
    
    def _process_non_arxiv_entries(self, entries: List[NotionEntry]) -> List[Document]:
        """
        Create documents from non-Arxiv Notion entries.
        Uses the user's notes as the primary content.
        """
        documents = []
        
        for entry in entries:
            # Skip entries with minimal content
            if len(entry.notes.strip()) < 50:
                continue
            
            doc = Document(
                page_content=f"# {entry.title}\n\n{entry.notes}",
                metadata={
                    "source": "notion",
                    "notion_id": entry.notion_id,
                    "title": entry.title,
                    "source_url": entry.source_url,
                    "notion_categories": entry.categories,
                    "notion_topics": entry.topics,
                    "created_date": entry.created_date,
                }
            )
            documents.append(doc)
        
        logger.info(f"  Created {len(documents)} documents from Notion notes")
        return documents
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        return {
            "notion_entries_loaded": self.stats.notion_entries_loaded,
            "arxiv_papers_found": self.stats.arxiv_papers_found,
            "arxiv_papers_fetched": self.stats.arxiv_papers_fetched,
            "non_arxiv_entries": self.stats.non_arxiv_entries,
            "total_documents": self.stats.total_documents,
            "total_chunks": self.stats.total_chunks,
            "fetch_success_rate": (
                self.stats.arxiv_papers_fetched / self.stats.arxiv_papers_found 
                if self.stats.arxiv_papers_found > 0 else 0
            ),
            "errors": self.stats.errors
        }


def ingest_documents() -> List[Document]:
    """Convenience function for document ingestion."""
    pipeline = DocumentPipeline()
    chunks = pipeline.run()
    
    # Print stats
    stats = pipeline.get_stats()
    print("\nIngestion Statistics:")
    for key, value in stats.items():
        if key != "errors":
            print(f"  {key}: {value}")
    
    return chunks
```

---

#### NRAG-011: Document Processing & Text Splitting

**Priority:** P0 - Critical  
**Token Estimate:** 6,000 tokens  
**Status:** To Do

**Description:**  
Implement configurable text splitting strategies optimized for RAG retrieval.

**Acceptance Criteria:**

- [x] `RecursiveCharacterTextSplitter` with optimal settings
- [x] Metadata preservation through splits
- [x] Special handling for academic paper structure
- [x] Configurable chunk size based on embedding model

**Implementation:**

```python
# src/rag/text_processing.py
import logging
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from config.settings import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Processes documents with appropriate splitting strategies.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Standard splitter for general content
        self.standard_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
        # Markdown-aware splitter for structured documents
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
            ]
        )
    
    def process_documents(
        self, 
        documents: List[Document],
        use_markdown_splitting: bool = False
    ) -> List[Document]:
        """
        Process and split documents.
        
        Args:
            documents: Raw documents to process
            use_markdown_splitting: Use markdown-aware splitting
        
        Returns:
            List of chunked documents
        """
        if use_markdown_splitting:
            return self._process_with_markdown(documents)
        else:
            return self._process_standard(documents)
    
    def _process_standard(self, documents: List[Document]) -> List[Document]:
        """Standard recursive character splitting."""
        chunks = self.standard_splitter.split_documents(documents)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def _process_with_markdown(self, documents: List[Document]) -> List[Document]:
        """Markdown-aware splitting that preserves structure."""
        all_chunks = []
        
        for doc in documents:
            # First split by markdown headers
            md_splits = self.markdown_splitter.split_text(doc.page_content)
            
            # Then apply size-based splitting to each section
            for md_doc in md_splits:
                # Preserve original metadata and add header info
                combined_metadata = {**doc.metadata, **md_doc.metadata}
                md_doc.metadata = combined_metadata
            
            # Apply size splitting
            sized_chunks = self.standard_splitter.split_documents(md_splits)
            all_chunks.extend(sized_chunks)
        
        logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks (markdown-aware)")
        return all_chunks
```

---

### Session 2 Summary

| Item | Token Estimate | Cumulative |
|------|----------------|------------|
| NRAG-008: Notion Loader | 12,000 | 12,000 |
| NRAG-009: Arxiv Loader | 10,000 | 22,000 |
| NRAG-010: Pipeline Orchestration | 10,000 | 32,000 |
| NRAG-011: Text Processing | 6,000 | 38,000 |
| **Session Buffer** | ~42,000 | 80,000 |

**Buffer Use:** Testing loaders, debugging API integration, handling edge cases

---

