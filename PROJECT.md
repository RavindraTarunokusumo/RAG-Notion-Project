# Notion Agentic RAG System - Product Backlog (AI Agent Edition)

> **Project:** Notion Agentic RAG with A2A Protocol  
> **Tech Stack:** Python, LangChain/LangGraph, Cohere API, LangSmith, A2A Protocol  
> **Execution Model:** AI Agent Sessions (Token-Based)  
> **Session Token Budget:** ~80,000 tokens per session  
> **Total Sessions:** 6 Sessions (Core Implementation)

---

## Table of Contents

1. [Project Vision](#project-vision)
2. [Architecture Overview](#architecture-overview)
3. [Execution Model](#execution-model)
4. [Agent Responsibility](#commit-responsibility)
5. [Session 1: Project Foundation](#session-1-project-foundation)
6. [Session 2: Notion Integration & Document Pipeline](#session-2-notion-integration--document-pipeline)
7. [Session 3: Vector Store & Embeddings](#session-3-vector-store--embeddings)
8. [Session 4: Planner & Researcher Agents](#session-4-planner--researcher-agents)
9. [Session 5: Reasoner & Synthesiser Agents](#session-5-reasoner--synthesiser-agents)
10. [Session 6: Orchestration & Testing](#session-6-orchestration--testing)
11. [Future Backlog](#future-backlog)
12. [Technical Debt & Improvements](#technical-debt--improvements)

---

## Project Vision

Build an Agentic RAG system that retrieves and synthesizes information from a Notion knowledge base using four specialized AI agents (Planner, Researcher, Reasoner, Synthesiser) communicating via the A2A protocol, powered by Cohere's Command R/R+ models through LangChain/LangGraph.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                       │
│                    (LangSmith Tracing)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Planner     │───>│  Researcher   │───>│   Reasoner    │
│ (Command R)   │    │ (Command R)   │    │(Command R+)   │
└───────────────┘    └───────────────┘    └───────────────┘
                              │                     │
                              ▼                     ▼
                     ┌───────────────┐    ┌───────────────┐
                     │  Vector Store │    │  Synthesiser  │
                     │  (ChromaDB)   │    │ (Command R+)  │
                     └───────────────┘    └───────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────┐                          ┌───────────────┐
│ Notion Loader │                          │ Arxiv Loader  │
│ (Metadata &   │─────────────────────────>│ (Full Papers) │
│  Arxiv Links) │                          │               │
└───────────────┘                          └───────────────┘
```

### Document Pipeline Flow

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Notion KB       │     │  Extract Arxiv   │     │  ArxivLoader     │
│  (Links + Meta)  │────>│  IDs from Links  │────>│  (Full Papers)   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                                                  │
         │ Metadata:                                        │ Content:
         │ - Title                                          │ - Full Abstract
         │ - Categories                                     │ - Paper Content
         │ - Source URL                                     │ - Authors
         │ - User Tags                                      │ - Publication Date
         │                                                  │
         └──────────────────────┬───────────────────────────┘
                                ▼
                    ┌──────────────────────┐
                    │  Merged Documents    │
                    │  (Content + Meta)    │
                    └──────────────────────┘
```

---

## Execution Model

### Token Budget Guidelines

| Task Type | Estimated Tokens | Description |
|-----------|------------------|-------------|
| File Creation (Small) | 2,000 - 5,000 | Config files, simple modules |
| File Creation (Medium) | 5,000 - 15,000 | Agent implementations, utilities |
| File Creation (Large) | 15,000 - 30,000 | Complex modules with multiple functions |
| Code Review & Iteration | 3,000 - 8,000 | Debugging, refactoring |
| Testing & Validation | 5,000 - 12,000 | Test execution, result analysis |
| Documentation | 2,000 - 6,000 | README updates, inline comments |

### Session Structure

Each session operates within a **~80,000 token budget** and should:

1. Begin with context loading (reading previous session outputs)
2. Execute backlog items in priority order
3. Validate outputs before session end
4. Document any blockers or carryover items

### Carryover Protocol

If a session exhausts its token budget before completing all items:

1. Mark incomplete items with `Status: Carryover`
2. Document progress made and remaining work
3. Next session begins by completing carryover items first


---

## Session 1: Project Foundation

**Session Token Budget:** ~80,000 tokens  
**Focus:** Project setup, environment configuration, and core infrastructure

---

#### NRAG-001: Project Structure Initialization

**Priority:** P0 - Critical  
**Token Estimate:** 8,000 tokens  
**Status:** To Do

**Description:**  
Create the complete project directory structure and initialize all placeholder files.

**Acceptance Criteria:**

- [ ] Project directory structure created
- [ ] All `__init__.py` files in place
- [ ] `.gitignore` configured for Python projects
- [ ] Basic `README.md` with project overview

**Directory Structure:**

```
notion-agentic-rag/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner.py
│   │   ├── researcher.py
│   │   ├── reasoner.py
│   │   └── synthesiser.py
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── notion_loader.py
│   │   └── arxiv_loader.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   └── vectorstore.py
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   └── state.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   └── test_agents.py
├── config/
│   └── settings.py
├── data/
│   └── .gitkeep
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── main.py
```

---

#### NRAG-002: Dependency Configuration

**Priority:** P0 - Critical  
**Token Estimate:** 5,000 tokens  
**Status:** To Do

**Description:**  
Create `pyproject.toml` with all required dependencies and version constraints using `uv`.

**Acceptance Criteria:**

- [ ] `pyproject.toml` created with standard PEP 621 configuration (uv)
- [ ] All core dependencies specified with version ranges
- [ ] Development dependencies separated
- [ ] Python version constraint: `>=3.11`

**Dependencies:**

```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "langchain>=0.3.0",
    "langchain-cohere>=0.5.0",
    "langchain-community>=0.3.0",
    "langgraph>=0.2.0",
    "langsmith>=0.2.0",
    "chromadb>=0.5.0",
    "python-dotenv>=1.0.0",
    "arxiv>=2.1.0",
    "pydantic>=2.0.0",
    "httpx>=0.27.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "black>=24.0.0",
    "ruff>=0.3.0",
]
```

---

#### NRAG-003: Configuration Module

**Priority:** P0 - Critical  
**Token Estimate:** 6,000 tokens  
**Status:** To Do

**Description:**  
Create centralized configuration management using Pydantic Settings.

**Acceptance Criteria:**

- [ ] `config/settings.py` with Pydantic BaseSettings
- [ ] Environment variable loading from `.env`
- [ ] Validation for required API keys
- [ ] Model configuration presets

**Implementation:**

```python
# config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal

class CohereModelConfig(BaseSettings):
    planner_model: str = "command-r-08-2024"
    researcher_model: str = "command-r-08-2024"
    reasoner_model: str = "command-r-plus-08-2024"
    synthesiser_model: str = "command-r-plus-08-2024"
    
    planner_temperature: float = 0.0
    researcher_temperature: float = 0.0
    reasoner_temperature: float = 0.1
    synthesiser_temperature: float = 0.3

class Settings(BaseSettings):
    # API Keys
    cohere_api_key: str = Field(..., env="COHERE_API_KEY")
    notion_token: str = Field(..., env="NOTION_TOKEN")
    notion_database_id: str = Field(..., env="NOTION_DATABASE_ID")
    langsmith_api_key: str = Field(..., env="LANGCHAIN_API_KEY")
    
    # LangSmith
    langsmith_tracing: bool = True
    langsmith_project: str = "notion-agentic-rag"
    
    # Vector Store
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "notion_knowledge_base"
    
    # RAG Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 10
    rerank_top_n: int = 5
    
    # Model Config
    models: CohereModelConfig = CohereModelConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

---

#### NRAG-004: Environment Template

**Priority:** P0 - Critical  
**Token Estimate:** 2,000 tokens  
**Status:** To Do

**Description:**  
Create `.env.example` template with all required environment variables.

**Acceptance Criteria:**

- [ ] All required variables documented
- [ ] Placeholder values clearly marked
- [ ] Comments explaining each variable

**Content:**

```bash
# .env.example

# ===========================================
# REQUIRED: API Keys
# ===========================================

# Cohere API Key (https://dashboard.cohere.com/api-keys)
COHERE_API_KEY=your_cohere_api_key_here

# Notion Integration Token (https://www.notion.so/my-integrations)
NOTION_TOKEN=your_notion_integration_token_here

# Notion Database ID (from your knowledge base URL)
# URL format: https://notion.so/{workspace}/{database_id}?v={view_id}
NOTION_DATABASE_ID=your_database_id_here

# LangSmith API Key (https://smith.langchain.com/)
LANGSMITH_API_KEY=your_langsmith_api_key_here

# ===========================================
# OPTIONAL: LangSmith Configuration
# ===========================================
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=notion-agentic-rag
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com

# ===========================================
# OPTIONAL: Vector Store
# ===========================================
CHROMA_PERSIST_DIR=./data/chroma_db

# ===========================================
# FUTURE: AgentLightning (Not yet implemented)
# ===========================================
# AGENT_LIGHTNING_API_KEY=your_agent_lightning_key
```

---

#### NRAG-005: LangSmith Tracing Setup

**Priority:** P1 - High  
**Token Estimate:** 5,000 tokens  
**Status:** To Do

**Description:**  
Implement LangSmith integration utilities for comprehensive tracing.

**Acceptance Criteria:**

- [ ] Tracing initialization function
- [ ] Custom decorators for agent functions
- [ ] Metadata tagging utilities
- [ ] Environment variable validation

**Implementation:**

```python
# src/utils/tracing.py
import os
from functools import wraps
from typing import Callable, Any
from langsmith import traceable
from config.settings import settings

def initialize_tracing():
    """Initialize LangSmith tracing with project settings."""
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langsmith_tracing).lower()
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"

def agent_trace(
    agent_name: str,
    model: str = None,
    tags: list[str] = None
) -> Callable:
    """Decorator for tracing agent function calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @traceable(
            name=f"{agent_name}_agent",
            tags=["agent", agent_name] + (tags or []),
            metadata={"model": model, "agent_type": agent_name}
        )
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

#### NRAG-006: Utility Helpers Module

**Priority:** P1 - High  
**Token Estimate:** 6,000 tokens  
**Status:** To Do

**Description:**  
Create common utility functions used across the project.

**Acceptance Criteria:**

- [ ] Document deduplication function
- [ ] Arxiv ID extraction from URLs
- [ ] JSON parsing utilities with error handling
- [ ] Document formatting helpers

**Implementation:**

```python
# src/utils/helpers.py
import re
import hashlib
from typing import List, Any
from langchain_core.documents import Document

def extract_arxiv_id(url: str) -> str | None:
    """
    Extract Arxiv ID from various URL formats.
    
    Supported formats:
    - https://arxiv.org/abs/2401.12345
    - https://arxiv.org/pdf/2401.12345.pdf
    - http://arxiv.org/abs/2401.12345v2
    - arxiv:2401.12345
    """
    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+)(v\d+)?',
        r'arxiv\.org/pdf/(\d+\.\d+)(v\d+)?',
        r'arxiv:(\d+\.\d+)(v\d+)?',
        r'(\d{4}\.\d{4,5})(v\d+)?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """Remove duplicate documents based on content hash."""
    seen_hashes = set()
    unique_docs = []
    
    for doc in docs:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    
    return unique_docs

def format_documents_for_prompt(docs: List[Document], max_chars: int = 15000) -> str:
    """Format documents for inclusion in LLM prompts."""
    formatted_parts = []
    total_chars = 0
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "Untitled")
        
        doc_text = f"[Document {i}]\nTitle: {title}\nSource: {source}\n\n{doc.page_content}\n"
        
        if total_chars + len(doc_text) > max_chars:
            break
            
        formatted_parts.append(doc_text)
        total_chars += len(doc_text)
    
    return "\n---\n".join(formatted_parts)

def safe_json_parse(text: str) -> dict | None:
    """Safely parse JSON from LLM output, handling common issues."""
    import json
    
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object in text
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None
```

---

#### NRAG-007: Main Entry Point

**Priority:** P1 - High  
**Token Estimate:** 4,000 tokens  
**Status:** To Do

**Description:**  
Create the main application entry point with CLI interface.

**Acceptance Criteria:**

- [ ] `main.py` with basic CLI
- [ ] Initialization of all components
- [ ] Simple query execution
- [ ] Error handling and logging

**Implementation:**

```python
# main.py
import argparse
import logging
from src.utils.tracing import initialize_tracing
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Notion Agentic RAG System")
    parser.add_argument("--query", "-q", type=str, help="Query to process")
    parser.add_argument("--ingest", action="store_true", help="Run document ingestion")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tracing
    initialize_tracing()
    logger.info(f"LangSmith project: {settings.langsmith_project}")
    
    if args.ingest:
        logger.info("Starting document ingestion...")
        # TODO: Implement in Session 2
        from src.loaders.notion_loader import ingest_documents
        ingest_documents()
        
    elif args.query:
        logger.info(f"Processing query: {args.query}")
        # TODO: Implement in Session 6
        from src.orchestrator.graph import create_workflow
        app = create_workflow()
        result = app.invoke({"query": args.query})
        print("\n" + "="*50)
        print("RESPONSE:")
        print("="*50)
        print(result["final_answer"])
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

---

### Session 1 Summary

| Item | Token Estimate | Cumulative |
|------|----------------|------------|
| NRAG-001: Project Structure | 8,000 | 8,000 |
| NRAG-002: Dependencies | 5,000 | 13,000 |
| NRAG-003: Configuration | 6,000 | 19,000 |
| NRAG-004: Environment Template | 2,000 | 21,000 |
| NRAG-005: LangSmith Setup | 5,000 | 26,000 |
| NRAG-006: Utility Helpers | 6,000 | 32,000 |
| NRAG-007: Main Entry Point | 4,000 | 36,000 |
| **Session Buffer** | ~44,000 | 80,000 |

**Buffer Use:** Code review, iteration, debugging, documentation updates

---

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

- [ ] `NotionDBLoader` configured with authentication
- [ ] Metadata extraction (title, categories, topics, source URL)
- [ ] Arxiv link identification and ID extraction
- [ ] Error handling for API rate limits
- [ ] Caching of loaded metadata to avoid repeated API calls

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

- [ ] Batch fetching of papers by Arxiv ID
- [ ] Full abstract and paper content extraction
- [ ] Metadata merging (Notion categories + Arxiv metadata)
- [ ] Rate limiting to avoid API blocks
- [ ] Graceful handling of unavailable papers

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
Create the unified document ingestion pipeline that orchestrates Notion → Arxiv → Processing flow.

**Acceptance Criteria:**

- [ ] Single entry point for document ingestion
- [ ] Configurable pipeline stages
- [ ] Progress logging and metrics
- [ ] Support for incremental updates

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

- [ ] `RecursiveCharacterTextSplitter` with optimal settings
- [ ] Metadata preservation through splits
- [ ] Special handling for academic paper structure
- [ ] Configurable chunk size based on embedding model

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

## Session 3: Vector Store & Embeddings

**Session Token Budget:** ~80,000 tokens  
**Focus:** Cohere embeddings, ChromaDB setup, and retrieval with reranking

---

#### NRAG-012: Cohere Embeddings Configuration

**Priority:** P0 - Critical  
**Token Estimate:** 8,000 tokens  
**Status:** To Do

**Description:**  
Configure Cohere embeddings for document vectorization.

**Acceptance Criteria:**

- [ ] `CohereEmbeddings` wrapper configured
- [ ] Embed model selection (`embed-english-v3.0`)
- [ ] Batch embedding support
- [ ] Error handling for API failures

**Implementation:**

```python
# src/rag/embeddings.py
import logging
from typing import List
from langchain_cohere import CohereEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Manages document and query embeddings using Cohere.
    """
    
    def __init__(self, model: str = "embed-english-v3.0"):
        self.model = model
        self.embeddings = CohereEmbeddings(
            model=model,
            cohere_api_key=settings.cohere_api_key
        )
        logger.info(f"Initialized Cohere embeddings with model: {model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts."""
        logger.info(f"Embedding {len(texts)} documents...")
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        return self.embeddings.embed_query(query)
    
    def get_embeddings_model(self) -> CohereEmbeddings:
        """Get the underlying embeddings model for use with vectorstores."""
        return self.embeddings


def get_embeddings() -> CohereEmbeddings:
    """Factory function for embeddings."""
    return EmbeddingService().get_embeddings_model()
```

---

#### NRAG-013: ChromaDB Vector Store Setup

**Priority:** P0 - Critical  
**Token Estimate:** 12,000 tokens  
**Status:** To Do

**Description:**  
Configure ChromaDB for persistent vector storage and retrieval.

**Acceptance Criteria:**

- [ ] Persistent ChromaDB configuration
- [ ] Collection management (create, update, delete)
- [ ] Document addition with metadata
- [ ] Similarity search functionality
- [ ] Collection statistics and health checks

**Implementation:**

```python
# src/rag/vectorstore.py
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from src.rag.embeddings import get_embeddings
from config.settings import settings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages ChromaDB vector store operations.
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None
    ):
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.embeddings = get_embeddings()
        self._vectorstore: Optional[Chroma] = None
    
    @property
    def vectorstore(self) -> Chroma:
        """Lazy initialization of vectorstore."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        return self._vectorstore
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Returns:
            List of document IDs
        """
        logger.info(f"Adding {len(documents)} documents to collection '{self.collection_name}'")
        
        ids = self.vectorstore.add_documents(documents)
        
        logger.info(f"Successfully added {len(ids)} documents")
        return ids
    
    def create_from_documents(self, documents: List[Document]) -> "VectorStoreManager":
        """
        Create a new vector store from documents (replaces existing).
        """
        logger.info(f"Creating new collection '{self.collection_name}' with {len(documents)} documents")
        
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        return self
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: dict = None
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results (default from settings)
            filter: Metadata filter
        
        Returns:
            List of relevant documents
        """
        k = k or settings.retrieval_k
        
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        
        logger.debug(f"Found {len(results)} documents for query")
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = None
    ) -> List[tuple[Document, float]]:
        """Similarity search with relevance scores."""
        k = k or settings.retrieval_k
        
        return self.vectorstore.similarity_search_with_score(query=query, k=k)
    
    def as_retriever(self, **kwargs):
        """Get as a LangChain retriever."""
        search_kwargs = {
            "k": kwargs.pop("k", settings.retrieval_k)
        }
        search_kwargs.update(kwargs.get("search_kwargs", {}))
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        collection = self.vectorstore._collection
        
        return {
            "name": self.collection_name,
            "count": collection.count(),
            "persist_directory": self.persist_directory
        }
    
    def delete_collection(self):
        """Delete the entire collection."""
        logger.warning(f"Deleting collection '{self.collection_name}'")
        self.vectorstore.delete_collection()
        self._vectorstore = None


# Singleton instance
_vector_store_manager: Optional[VectorStoreManager] = None

def get_vector_store() -> VectorStoreManager:
    """Get the singleton vector store manager."""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager
```

---

#### NRAG-014: Cohere Rerank Integration

**Priority:** P1 - High  
**Token Estimate:** 8,000 tokens  
**Status:** To Do

**Description:**  
Integrate Cohere Rerank for improved retrieval relevance.

**Acceptance Criteria:**

- [ ] `CohereRerank` compressor configured
- [ ] `ContextualCompressionRetriever` setup
- [ ] Configurable `top_n` parameter
- [ ] Fallback to standard retrieval if rerank fails

**Implementation:**

```python
# src/rag/retriever.py
import logging
from typing import List
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from src.rag.vectorstore import get_vector_store
from config.settings import settings

logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Retriever with Cohere Rerank for improved relevance.
    """
    
    def __init__(
        self,
        use_rerank: bool = True,
        rerank_top_n: int = None,
        retrieval_k: int = None
    ):
        self.use_rerank = use_rerank
        self.rerank_top_n = rerank_top_n or settings.rerank_top_n
        self.retrieval_k = retrieval_k or settings.retrieval_k
        
        self.vector_store = get_vector_store()
        self.base_retriever = self.vector_store.as_retriever(k=self.retrieval_k)
        
        if use_rerank:
            self.reranker = CohereRerank(
                model="rerank-english-v3.0",
                cohere_api_key=settings.cohere_api_key,
                top_n=self.rerank_top_n
            )
            
            self.retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=self.base_retriever
            )
            logger.info(f"Initialized retriever with Cohere Rerank (top_n={self.rerank_top_n})")
        else:
            self.retriever = self.base_retriever
            logger.info("Initialized retriever without reranking")
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
        
        Returns:
            List of relevant documents (reranked if enabled)
        """
        try:
            docs = self.retriever.invoke(query)
            logger.debug(f"Retrieved {len(docs)} documents for query")
            return docs
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            # Fallback to base retriever
            if self.use_rerank:
                logger.info("Falling back to base retriever")
                return self.base_retriever.invoke(query)
            raise
    
    def retrieve_for_tasks(self, tasks: List[dict]) -> List[Document]:
        """
        Retrieve documents for multiple sub-tasks.
        
        Args:
            tasks: List of task dictionaries with 'task' key
        
        Returns:
            Deduplicated list of documents from all tasks
        """
        from src.utils.helpers import deduplicate_documents
        
        all_docs = []
        for task in tasks:
            task_query = task.get("task", str(task))
            docs = self.retrieve(task_query)
            all_docs.extend(docs)
        
        unique_docs = deduplicate_documents(all_docs)
        logger.info(f"Retrieved {len(unique_docs)} unique documents for {len(tasks)} tasks")
        
        return unique_docs


def get_retriever(use_rerank: bool = True) -> RAGRetriever:
    """Factory function for retriever."""
    return RAGRetriever(use_rerank=use_rerank)
```

---

#### NRAG-015: Ingestion Integration Script

**Priority:** P0 - Critical  
**Token Estimate:** 6,000 tokens  
**Status:** To Do

**Description:**  
Create a complete ingestion script that ties together the pipeline and vector store.

**Acceptance Criteria:**

- [ ] Single command to run full ingestion
- [ ] Progress reporting
- [ ] Option to rebuild or update collection
- [ ] Summary statistics after completion

**Implementation:**

```python
# src/ingest.py
import logging
import argparse
from datetime import datetime

from src.loaders.pipeline import DocumentPipeline
from src.rag.vectorstore import VectorStoreManager
from src.utils.tracing import initialize_tracing
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_ingestion(rebuild: bool = False):
    """
    Run the complete document ingestion pipeline.
    
    Args:
        rebuild: If True, delete existing collection and rebuild
    """
    initialize_tracing()
    
    start_time = datetime.now()
    logger.info("="*60)
    logger.info("NOTION AGENTIC RAG - Document Ingestion")
    logger.info(f"Started at: {start_time.isoformat()}")
    logger.info("="*60)
    
    # Initialize vector store manager
    vs_manager = VectorStoreManager()
    
    # Check existing collection
    try:
        stats = vs_manager.get_collection_stats()
        logger.info(f"Existing collection: {stats['count']} documents")
        
        if rebuild and stats['count'] > 0:
            logger.warning("Rebuild requested - deleting existing collection...")
            vs_manager.delete_collection()
            vs_manager = VectorStoreManager()  # Reinitialize
    except Exception:
        logger.info("No existing collection found")
    
    # Run document pipeline
    logger.info("\n--- Running Document Pipeline ---")
    pipeline = DocumentPipeline()
    chunks = pipeline.run(include_non_arxiv=True)
    
    if not chunks:
        logger.error("No documents to ingest!")
        return
    
    # Add to vector store
    logger.info("\n--- Adding to Vector Store ---")
    vs_manager.create_from_documents(chunks)
    
    # Final stats
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    final_stats = vs_manager.get_collection_stats()
    pipeline_stats = pipeline.get_stats()
    
    logger.info("\n" + "="*60)
    logger.info("INGESTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Notion entries processed: {pipeline_stats['notion_entries_loaded']}")
    logger.info(f"Arxiv papers fetched: {pipeline_stats['arxiv_papers_fetched']}")
    logger.info(f"Total chunks created: {pipeline_stats['total_chunks']}")
    logger.info(f"Documents in collection: {final_stats['count']}")
    logger.info("="*60)
    
    return final_stats


def main():
    parser = argparse.ArgumentParser(description="Document Ingestion for Notion Agentic RAG")
    parser.add_argument(
        "--rebuild", 
        action="store_true",
        help="Delete existing collection and rebuild from scratch"
    )
    
    args = parser.parse_args()
    run_ingestion(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
```

---

### Session 3 Summary

| Item | Token Estimate | Cumulative |
|------|----------------|------------|
| NRAG-012: Cohere Embeddings | 8,000 | 8,000 |
| NRAG-013: ChromaDB Setup | 12,000 | 20,000 |
| NRAG-014: Cohere Rerank | 8,000 | 28,000 |
| NRAG-015: Ingestion Script | 6,000 | 34,000 |
| **Session Buffer** | ~46,000 | 80,000 |

**Buffer Use:** Testing embeddings, debugging vectorstore, integration testing

---

## Session 4: Planner & Researcher Agents

**Session Token Budget:** ~80,000 tokens  
**Focus:** Implement the first two agents in the pipeline

---

#### NRAG-016: Agent State Schema

**Priority:** P0 - Critical  
**Token Estimate:** 5,000 tokens  
**Status:** To Do

**Description:**  
Define the shared state schema for all agents in the LangGraph workflow.

**Acceptance Criteria:**

- [ ] TypedDict for agent state
- [ ] All required fields defined
- [ ] Clear documentation of each field
- [ ] Support for error states

**Implementation:**

```python
# src/orchestrator/state.py
from typing import TypedDict, List, Any, Optional
from langchain_core.documents import Document

class SubTask(TypedDict):
    """A single sub-task created by the Planner."""
    id: int
    task: str
    priority: str  # "high", "medium", "low"
    keywords: List[str]

class Analysis(TypedDict):
    """Analysis result from the Reasoner."""
    sub_task_id: int
    key_findings: List[str]
    supporting_evidence: List[str]
    contradictions: List[str]
    confidence: float  # 0.0 to 1.0
    gaps: List[str]

class AgentState(TypedDict):
    """
    Shared state passed between all agents in the workflow.
    
    Flow:
    1. User provides `query`
    2. Planner populates `sub_tasks`
    3. Researcher populates `retrieved_docs`
    4. Reasoner populates `analysis`
    5. Synthesiser populates `final_answer`
    """
    # Input
    query: str
    
    # Planner output
    sub_tasks: List[SubTask]
    planning_reasoning: str
    
    # Researcher output
    retrieved_docs: List[Document]
    retrieval_metadata: dict
    
    # Reasoner output
    analysis: List[Analysis]
    overall_assessment: str
    
    # Synthesiser output
    final_answer: str
    sources: List[dict]
    
    # Error handling
    error: Optional[str]
    current_agent: str
```

---

#### NRAG-017: LLM Factory

**Priority:** P0 - Critical  
**Token Estimate:** 6,000 tokens  
**Status:** To Do

**Description:**  
Create factory for instantiating Cohere LLMs with agent-specific configurations.

**Acceptance Criteria:**

- [ ] Centralized LLM creation
- [ ] Agent-specific model/temperature settings
- [ ] Retry logic configuration
- [ ] Consistent error handling

**Implementation:**

```python
# src/agents/llm_factory.py
import logging
from typing import Literal
from langchain_cohere import ChatCohere
from config.settings import settings

logger = logging.getLogger(__name__)

AgentType = Literal["planner", "researcher", "reasoner", "synthesiser"]

# Model assignments based on task complexity
AGENT_CONFIGS = {
    "planner": {
        "model": settings.models.planner_model,
        "temperature": settings.models.planner_temperature,
        "max_tokens": 1024,
        "description": "Task decomposition - fast, focused"
    },
    "researcher": {
        "model": settings.models.researcher_model,
        "temperature": settings.models.researcher_temperature,
        "max_tokens": 2048,
        "description": "Query formulation - precise, systematic"
    },
    "reasoner": {
        "model": settings.models.reasoner_model,
        "temperature": settings.models.reasoner_temperature,
        "max_tokens": 4096,
        "description": "Complex analysis - powerful, nuanced"
    },
    "synthesiser": {
        "model": settings.models.synthesiser_model,
        "temperature": settings.models.synthesiser_temperature,
        "max_tokens": 4096,
        "description": "Response generation - creative, coherent"
    }
}

def get_agent_llm(agent_type: AgentType) -> ChatCohere:
    """
    Get a configured LLM for the specified agent type.
    
    Model sizes:
    - Planner: command-r-08-2024 (35B) - Fast task decomposition
    - Researcher: command-r-08-2024 (35B) - Efficient query handling
    - Reasoner: command-r-plus-08-2024 (104B) - Deep analysis
    - Synthesiser: command-r-plus-08-2024 (104B) - Quality generation
    """
    if agent_type not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    config = AGENT_CONFIGS[agent_type]
    
    llm = ChatCohere(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        cohere_api_key=settings.cohere_api_key
    )
    
    logger.debug(f"Created LLM for {agent_type}: {config['model']} ({config['description']})")
    
    return llm

def get_model_info(agent_type: AgentType) -> dict:
    """Get information about the model used for an agent."""
    return AGENT_CONFIGS.get(agent_type, {})
```

---

#### NRAG-018: Planner Agent Implementation

**Priority:** P0 - Critical  
**Token Estimate:** 12,000 tokens  
**Status:** To Do

**Description:**  
Implement the Planner agent that decomposes queries into sub-tasks.

**Acceptance Criteria:**

- [ ] Query decomposition into 2-5 sub-tasks
- [ ] Priority assignment for each sub-task
- [ ] Keyword extraction for retrieval
- [ ] JSON output parsing with fallback
- [ ] LangSmith tracing integration

**Implementation:**

```python
# src/agents/planner.py
import logging
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.orchestrator.state import AgentState, SubTask
from src.agents.llm_factory import get_agent_llm
from src.utils.tracing import agent_trace
from src.utils.helpers import safe_json_parse

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a Planning Agent specialized in breaking down complex questions into actionable research tasks.

Your role is to analyze the user's query and decompose it into 2-5 specific sub-tasks that can be researched independently from a knowledge base about AI, machine learning, and related topics.

Guidelines:
1. Each sub-task should be a focused, specific question
2. Sub-tasks should cover different aspects of the query
3. Assign priority based on importance to answering the main query
4. Extract keywords that will help with document retrieval
5. Consider what information would be most valuable

Output your response as a JSON object with this exact structure:
{{
    "original_query": "the user's original question",
    "sub_tasks": [
        {{
            "id": 1,
            "task": "Specific question or research task",
            "priority": "high|medium|low",
            "keywords": ["keyword1", "keyword2"]
        }}
    ],
    "reasoning": "Brief explanation of why you decomposed the query this way"
}}

Important: Output ONLY the JSON object, no additional text."""

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT),
    ("human", "{query}")
])

class PlannerAgent:
    """
    Decomposes user queries into actionable sub-tasks.
    
    Uses Command R (smaller model) for fast, focused task decomposition.
    """
    
    def __init__(self):
        self.llm = get_agent_llm("planner")
        self.chain = PLANNER_PROMPT | self.llm
    
    @agent_trace("planner", model="command-r-08-2024")
    def __call__(self, state: AgentState) -> AgentState:
        """
        Process state and generate sub-tasks.
        
        Args:
            state: Current agent state with query
        
        Returns:
            Updated state with sub_tasks populated
        """
        logger.info(f"Planner processing query: {state['query'][:100]}...")
        
        try:
            response = self.chain.invoke({"query": state["query"]})
            content = response.content
            
            # Parse JSON output
            parsed = safe_json_parse(content)
            
            if not parsed:
                logger.warning("Failed to parse planner output, using fallback")
                parsed = self._create_fallback_tasks(state["query"])
            
            sub_tasks = parsed.get("sub_tasks", [])
            reasoning = parsed.get("reasoning", "")
            
            logger.info(f"Planner created {len(sub_tasks)} sub-tasks")
            
            return {
                **state,
                "sub_tasks": sub_tasks,
                "planning_reasoning": reasoning,
                "current_agent": "planner"
            }
            
        except Exception as e:
            logger.error(f"Planner error: {e}")
            return {
                **state,
                "sub_tasks": self._create_fallback_tasks(state["query"])["sub_tasks"],
                "planning_reasoning": f"Fallback due to error: {e}",
                "error": str(e),
                "current_agent": "planner"
            }
    
    def _create_fallback_tasks(self, query: str) -> dict:
        """Create simple fallback tasks when parsing fails."""
        return {
            "original_query": query,
            "sub_tasks": [
                {
                    "id": 1,
                    "task": query,
                    "priority": "high",
                    "keywords": query.lower().split()[:5]
                }
            ],
            "reasoning": "Fallback: using original query as single task"
        }


def planner_agent(state: AgentState) -> AgentState:
    """Functional interface for LangGraph."""
    agent = PlannerAgent()
    return agent(state)
```

---

#### NRAG-019: Researcher Agent Implementation

**Priority:** P0 - Critical  
**Token Estimate:** 14,000 tokens  
**Status:** To Do

**Description:**  
Implement the Researcher agent that retrieves relevant documents.

**Acceptance Criteria:**

- [ ] Process each sub-task from Planner
- [ ] Execute retrieval with reranking
- [ ] Deduplicate across sub-tasks
- [ ] Capture retrieval metadata
- [ ] Handle empty results gracefully

**Implementation:**

```python
# src/agents/researcher.py
import logging
from typing import List
from langchain_core.documents import Document

from src.orchestrator.state import AgentState
from src.rag.retriever import RAGRetriever
from src.utils.tracing import agent_trace
from src.utils.helpers import deduplicate_documents

logger = logging.getLogger(__name__)

class ResearcherAgent:
    """
    Retrieves relevant documents for each sub-task.
    
    Uses Cohere embeddings + reranking for high-quality retrieval.
    """
    
    def __init__(self, use_rerank: bool = True):
        self.retriever = RAGRetriever(use_rerank=use_rerank)
    
    @agent_trace("researcher", model="command-r-08-2024")
    def __call__(self, state: AgentState) -> AgentState:
        """
        Retrieve documents for all sub-tasks.
        
        Args:
            state: Current state with sub_tasks
        
        Returns:
            Updated state with retrieved_docs
        """
        sub_tasks = state.get("sub_tasks", [])
        
        if not sub_tasks:
            logger.warning("No sub-tasks provided to researcher")
            return {
                **state,
                "retrieved_docs": [],
                "retrieval_metadata": {"error": "No sub-tasks"},
                "current_agent": "researcher"
            }
        
        logger.info(f"Researcher processing {len(sub_tasks)} sub-tasks")
        
        all_docs = []
        task_results = {}
        
        for task in sub_tasks:
            task_id = task.get("id", 0)
            task_query = task.get("task", "")
            keywords = task.get("keywords", [])
            
            # Combine task with keywords for better retrieval
            search_query = f"{task_query} {' '.join(keywords)}"
            
            logger.debug(f"Searching for task {task_id}: {search_query[:100]}...")
            
            try:
                docs = self.retriever.retrieve(search_query)
                all_docs.extend(docs)
                task_results[task_id] = {
                    "query": search_query,
                    "docs_found": len(docs)
                }
            except Exception as e:
                logger.error(f"Retrieval error for task {task_id}: {e}")
                task_results[task_id] = {
                    "query": search_query,
                    "docs_found": 0,
                    "error": str(e)
                }
        
        # Deduplicate documents
        unique_docs = deduplicate_documents(all_docs)
        
        logger.info(f"Retrieved {len(unique_docs)} unique documents (from {len(all_docs)} total)")
        
        return {
            **state,
            "retrieved_docs": unique_docs,
            "retrieval_metadata": {
                "total_retrieved": len(all_docs),
                "unique_docs": len(unique_docs),
                "task_results": task_results
            },
            "current_agent": "researcher"
        }


def researcher_agent(state: AgentState) -> AgentState:
    """Functional interface for LangGraph."""
    agent = ResearcherAgent()
    return agent(state)
```

---

### Session 4 Summary

| Item | Token Estimate | Cumulative |
|------|----------------|------------|
| NRAG-016: State Schema | 5,000 | 5,000 |
| NRAG-017: LLM Factory | 6,000 | 11,000 |
| NRAG-018: Planner Agent | 12,000 | 23,000 |
| NRAG-019: Researcher Agent | 14,000 | 37,000 |
| **Session Buffer** | ~43,000 | 80,000 |

**Buffer Use:** Testing agents, prompt iteration, debugging

---

## Session 5: Reasoner & Synthesiser Agents

**Session Token Budget:** ~80,000 tokens  
**Focus:** Implement the analysis and synthesis agents

---

#### NRAG-020: Reasoner Agent Implementation

**Priority:** P0 - Critical  
**Token Estimate:** 16,000 tokens  
**Status:** To Do

**Description:**  
Implement the Reasoner agent that applies logical analysis to retrieved documents.

**Acceptance Criteria:**

- [ ] Analyze documents against each sub-task
- [ ] Identify key findings and evidence
- [ ] Detect contradictions and gaps
- [ ] Assign confidence scores
- [ ] Use larger model (Command R+) for complex reasoning

**Implementation:**

```python
# src/agents/reasoner.py
import logging
from langchain_core.prompts import ChatPromptTemplate

from src.orchestrator.state import AgentState, Analysis
from src.agents.llm_factory import get_agent_llm
from src.utils.tracing import agent_trace
from src.utils.helpers import safe_json_parse, format_documents_for_prompt

logger = logging.getLogger(__name__)

REASONER_SYSTEM_PROMPT = """You are a Reasoning Agent specialized in logical analysis and critical evaluation.

Your role is to analyze retrieved documents against the research sub-tasks and provide structured analysis.

For each sub-task, you must:
1. Identify key findings relevant to the task
2. Extract supporting evidence with specific citations
3. Note any contradictions or conflicting information
4. Assess your confidence in answering the task (0.0 to 1.0)
5. Identify gaps where information is missing or insufficient

Guidelines:
- Be thorough but concise
- Support findings with specific document references
- Be honest about uncertainty
- Consider multiple perspectives when documents conflict

Output your analysis as a JSON object:
{{
    "analyses": [
        {{
            "sub_task_id": 1,
            "key_findings": ["Finding 1", "Finding 2"],
            "supporting_evidence": ["Document X states...", "According to Document Y..."],
            "contradictions": ["Document A says X while Document B says Y"],
            "confidence": 0.85,
            "gaps": ["No information found about..."]
        }}
    ],
    "overall_assessment": "Summary of the analysis quality and completeness",
    "synthesis_recommendations": "Suggestions for how to synthesize the final answer"
}}

Important: Output ONLY the JSON object."""

REASONER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REASONER_SYSTEM_PROMPT),
    ("human", """Original Query: {query}

Sub-tasks to analyze:
{sub_tasks}

Retrieved Documents:
{documents}

Please analyze the documents against each sub-task.""")
])

class ReasonerAgent:
    """
    Applies logical analysis to retrieved documents.
    
    Uses Command R+ (larger model) for nuanced reasoning and analysis.
    """
    
    def __init__(self):
        self.llm = get_agent_llm("reasoner")
        self.chain = REASONER_PROMPT | self.llm
    
    @agent_trace("reasoner", model="command-r-plus-08-2024")
    def __call__(self, state: AgentState) -> AgentState:
        """
        Analyze retrieved documents.
        
        Args:
            state: Current state with sub_tasks and retrieved_docs
        
        Returns:
            Updated state with analysis
        """
        docs = state.get("retrieved_docs", [])
        sub_tasks = state.get("sub_tasks", [])
        
        if not docs:
            logger.warning("No documents to analyze")
            return {
                **state,
                "analysis": [],
                "overall_assessment": "No documents available for analysis",
                "current_agent": "reasoner"
            }
        
        logger.info(f"Reasoner analyzing {len(docs)} documents for {len(sub_tasks)} tasks")
        
        try:
            # Format inputs
            docs_text = format_documents_for_prompt(docs, max_chars=12000)
            tasks_text = self._format_tasks(sub_tasks)
            
            response = self.chain.invoke({
                "query": state["query"],
                "sub_tasks": tasks_text,
                "documents": docs_text
            })
            
            parsed = safe_json_parse(response.content)
            
            if not parsed:
                logger.warning("Failed to parse reasoner output")
                parsed = self._create_fallback_analysis(sub_tasks)
            
            analyses = parsed.get("analyses", [])
            overall = parsed.get("overall_assessment", "Analysis completed")
            
            logger.info(f"Reasoner completed analysis with {len(analyses)} task analyses")
            
            return {
                **state,
                "analysis": analyses,
                "overall_assessment": overall,
                "current_agent": "reasoner"
            }
            
        except Exception as e:
            logger.error(f"Reasoner error: {e}")
            return {
                **state,
                "analysis": [],
                "overall_assessment": f"Analysis failed: {e}",
                "error": str(e),
                "current_agent": "reasoner"
            }
    
    def _format_tasks(self, tasks: list) -> str:
        """Format sub-tasks for the prompt."""
        lines = []
        for task in tasks:
            lines.append(f"Task {task['id']} [{task['priority']}]: {task['task']}")
        return "\n".join(lines)
    
    def _create_fallback_analysis(self, sub_tasks: list) -> dict:
        """Create fallback analysis when parsing fails."""
        return {
            "analyses": [
                {
                    "sub_task_id": t["id"],
                    "key_findings": ["Analysis parsing failed"],
                    "supporting_evidence": [],
                    "contradictions": [],
                    "confidence": 0.3,
                    "gaps": ["Unable to perform detailed analysis"]
                }
                for t in sub_tasks
            ],
            "overall_assessment": "Fallback analysis due to parsing error"
        }


def reasoner_agent(state: AgentState) -> AgentState:
    """Functional interface for LangGraph."""
    agent = ReasonerAgent()
    return agent(state)
```

---

#### NRAG-021: Synthesiser Agent Implementation

**Priority:** P0 - Critical  
**Token Estimate:** 16,000 tokens  
**Status:** To Do

**Description:**  
Implement the Synthesiser agent that creates the final coherent response.

**Acceptance Criteria:**

- [ ] Combine all findings into coherent answer
- [ ] Include citations for factual claims
- [ ] Acknowledge uncertainty appropriately
- [ ] Adapt format to query type
- [ ] Include sources section

**Implementation:**

```python
# src/agents/synthesiser.py
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.orchestrator.state import AgentState
from src.agents.llm_factory import get_agent_llm
from src.utils.tracing import agent_trace
from src.utils.helpers import format_documents_for_prompt

logger = logging.getLogger(__name__)

SYNTHESISER_SYSTEM_PROMPT = """You are a Synthesis Agent specialized in creating comprehensive, well-structured answers.

Your role is to combine analysis results and source documents into a coherent, informative response that directly answers the user's question.

Guidelines:
1. Start with a direct answer to the main question
2. Organize information logically (use headers if helpful)
3. Include inline citations [Source N] for factual claims
4. Acknowledge areas of uncertainty or conflicting information
5. Be comprehensive but avoid unnecessary repetition
6. End with a "Sources" section listing referenced documents

Response Formatting:
- For explanatory questions: Structured explanation with examples
- For comparison questions: Clear comparison with key differences
- For how-to questions: Step-by-step guidance
- For exploratory questions: Comprehensive overview with multiple perspectives

Confidence Guidelines:
- High confidence (>0.8): State findings directly
- Medium confidence (0.5-0.8): Use phrases like "evidence suggests" or "likely"
- Low confidence (<0.5): Clearly note limitations and gaps

Remember: Quality over quantity. Be thorough but concise."""

SYNTHESISER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYNTHESISER_SYSTEM_PROMPT),
    ("human", """Original Question: {query}

Analysis Summary:
{analysis_summary}

Key Findings by Task:
{findings}

Source Documents:
{documents}

Please synthesize a comprehensive response to the original question.""")
])

class SynthesiserAgent:
    """
    Creates the final coherent response.
    
    Uses Command R+ (larger model) for high-quality text generation.
    """
    
    def __init__(self):
        self.llm = get_agent_llm("synthesiser")
        self.chain = SYNTHESISER_PROMPT | self.llm | StrOutputParser()
    
    @agent_trace("synthesiser", model="command-r-plus-08-2024")
    def __call__(self, state: AgentState) -> AgentState:
        """
        Generate final response.
        
        Args:
            state: Current state with all prior agent outputs
        
        Returns:
            Updated state with final_answer
        """
        analysis = state.get("analysis", [])
        docs = state.get("retrieved_docs", [])
        overall = state.get("overall_assessment", "")
        
        logger.info("Synthesiser generating final response")
        
        try:
            # Format analysis for prompt
            analysis_summary = overall
            findings = self._format_findings(analysis)
            docs_text = format_documents_for_prompt(docs, max_chars=10000)
            
            response = self.chain.invoke({
                "query": state["query"],
                "analysis_summary": analysis_summary,
                "findings": findings,
                "documents": docs_text
            })
            
            # Extract sources for metadata
            sources = self._extract_sources(docs)
            
            logger.info(f"Synthesiser generated response ({len(response)} chars)")
            
            return {
                **state,
                "final_answer": response,
                "sources": sources,
                "current_agent": "synthesiser"
            }
            
        except Exception as e:
            logger.error(f"Synthesiser error: {e}")
            return {
                **state,
                "final_answer": f"I apologize, but I encountered an error while generating the response: {e}",
                "sources": [],
                "error": str(e),
                "current_agent": "synthesiser"
            }
    
    def _format_findings(self, analyses: list) -> str:
        """Format analysis findings for the prompt."""
        if not analyses:
            return "No analysis available."
        
        lines = []
        for analysis in analyses:
            task_id = analysis.get("sub_task_id", "?")
            confidence = analysis.get("confidence", 0)
            findings = analysis.get("key_findings", [])
            gaps = analysis.get("gaps", [])
            
            lines.append(f"\n### Task {task_id} (Confidence: {confidence:.0%})")
            
            if findings:
                lines.append("Findings:")
                for f in findings:
                    lines.append(f"  - {f}")
            
            if gaps:
                lines.append("Gaps:")
                for g in gaps:
                    lines.append(f"  - {g}")
        
        return "\n".join(lines)
    
    def _extract_sources(self, docs: list) -> list:
        """Extract source metadata from documents."""
        sources = []
        seen = set()
        
        for i, doc in enumerate(docs, 1):
            source_id = doc.metadata.get("arxiv_id") or doc.metadata.get("notion_id") or f"doc_{i}"
            
            if source_id in seen:
                continue
            seen.add(source_id)
            
            sources.append({
                "id": i,
                "title": doc.metadata.get("title", "Untitled"),
                "source": doc.metadata.get("source", "unknown"),
                "url": doc.metadata.get("arxiv_url") or doc.metadata.get("source_url", ""),
                "arxiv_id": doc.metadata.get("arxiv_id", "")
            })
        
        return sources


def synthesiser_agent(state: AgentState) -> AgentState:
    """Functional interface for LangGraph."""
    agent = SynthesiserAgent()
    return agent(state)
```

---

#### NRAG-022: Agent Module Exports

**Priority:** P1 - High  
**Token Estimate:** 3,000 tokens  
**Status:** To Do

**Description:**  
Create clean module exports for all agents.

**Implementation:**

```python
# src/agents/__init__.py
"""
Agentic RAG Agent Implementations

Agent Pipeline:
1. Planner (Command R) - Decomposes queries into sub-tasks
2. Researcher (Command R) - Retrieves relevant documents  
3. Reasoner (Command R+) - Analyzes and evaluates findings
4. Synthesiser (Command R+) - Creates final coherent response
"""

from src.agents.planner import PlannerAgent, planner_agent
from src.agents.researcher import ResearcherAgent, researcher_agent
from src.agents.reasoner import ReasonerAgent, reasoner_agent
from src.agents.synthesiser import SynthesiserAgent, synthesiser_agent
from src.agents.llm_factory import get_agent_llm, AGENT_CONFIGS

__all__ = [
    # Classes
    "PlannerAgent",
    "ResearcherAgent", 
    "ReasonerAgent",
    "SynthesiserAgent",
    # Functional interfaces (for LangGraph)
    "planner_agent",
    "researcher_agent",
    "reasoner_agent",
    "synthesiser_agent",
    # Utilities
    "get_agent_llm",
    "AGENT_CONFIGS",
]
```

---

### Session 5 Summary

| Item | Token Estimate | Cumulative |
|------|----------------|------------|
| NRAG-020: Reasoner Agent | 16,000 | 16,000 |
| NRAG-021: Synthesiser Agent | 16,000 | 32,000 |
| NRAG-022: Module Exports | 3,000 | 35,000 |
| **Session Buffer** | ~45,000 | 80,000 |

**Buffer Use:** Prompt refinement, output quality testing, integration

---

## Session 6: Orchestration & Testing

**Session Token Budget:** ~80,000 tokens  
**Focus:** LangGraph workflow, end-to-end testing, documentation

---

#### NRAG-023: LangGraph Workflow Implementation

**Priority:** P0 - Critical  
**Token Estimate:** 15,000 tokens  
**Status:** To Do

**Description:**  
Create the complete LangGraph orchestration workflow.

**Acceptance Criteria:**

- [ ] StateGraph with all agent nodes
- [ ] Linear flow with proper edges
- [ ] Error handling nodes
- [ ] Compiled graph ready for execution

**Implementation:**

```python
# src/orchestrator/graph.py
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.orchestrator.state import AgentState
from src.agents import (
    planner_agent,
    researcher_agent,
    reasoner_agent,
    synthesiser_agent
)

logger = logging.getLogger(__name__)

def should_continue(state: AgentState) -> str:
    """Determine if workflow should continue or handle error."""
    if state.get("error"):
        return "handle_error"
    return "continue"

def handle_error(state: AgentState) -> AgentState:
    """Handle errors in the workflow."""
    error = state.get("error", "Unknown error")
    current = state.get("current_agent", "unknown")
    
    logger.error(f"Error in {current}: {error}")
    
    return {
        **state,
        "final_answer": f"I encountered an issue while processing your query. "
                       f"Error occurred in the {current} stage: {error}. "
                       f"Please try rephrasing your question or try again later."
    }

def create_workflow(with_memory: bool = False) -> StateGraph:
    """
    Create the Agentic RAG workflow.
    
    Flow:
    Query → Planner → Researcher → Reasoner → Synthesiser → Response
    
    Args:
        with_memory: Enable conversation memory (for multi-turn)
    
    Returns:
        Compiled LangGraph application
    """
    logger.info("Creating Agentic RAG workflow...")
    
    # Initialize graph with state schema
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("planner", planner_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("reasoner", reasoner_agent)
    workflow.add_node("synthesiser", synthesiser_agent)
    workflow.add_node("error_handler", handle_error)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Define edges (linear flow with error handling)
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {
            "continue": "researcher",
            "handle_error": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "researcher",
        should_continue,
        {
            "continue": "reasoner",
            "handle_error": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "reasoner",
        should_continue,
        {
            "continue": "synthesiser",
            "handle_error": "error_handler"
        }
    )
    
    # Terminal edges
    workflow.add_edge("synthesiser", END)
    workflow.add_edge("error_handler", END)
    
    # Compile with optional memory
    if with_memory:
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        logger.info("Workflow compiled with memory checkpointer")
    else:
        app = workflow.compile()
        logger.info("Workflow compiled (stateless)")
    
    return app


class AgenticRAG:
    """
    Main interface for the Agentic RAG system.
    """
    
    def __init__(self, with_memory: bool = False):
        self.app = create_workflow(with_memory=with_memory)
        self.with_memory = with_memory
    
    def query(
        self, 
        question: str,
        thread_id: str = None
    ) -> dict:
        """
        Process a query through the agentic pipeline.
        
        Args:
            question: User's question
            thread_id: Optional thread ID for conversation continuity
        
        Returns:
            Dict with final_answer and metadata
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        initial_state = {
            "query": question,
            "sub_tasks": [],
            "retrieved_docs": [],
            "analysis": [],
            "overall_assessment": "",
            "final_answer": "",
            "sources": [],
            "error": None,
            "current_agent": ""
        }
        
        config = {}
        if self.with_memory and thread_id:
            config = {"configurable": {"thread_id": thread_id}}
        
        result = self.app.invoke(initial_state, config)
        
        return {
            "answer": result.get("final_answer", ""),
            "sources": result.get("sources", []),
            "sub_tasks": result.get("sub_tasks", []),
            "analysis_summary": result.get("overall_assessment", ""),
            "error": result.get("error")
        }
    
    def query_with_details(self, question: str) -> AgentState:
        """
        Process query and return full state (for debugging).
        """
        initial_state = {
            "query": question,
            "sub_tasks": [],
            "retrieved_docs": [],
            "analysis": [],
            "overall_assessment": "",
            "final_answer": "",
            "sources": [],
            "error": None,
            "current_agent": ""
        }
        
        return self.app.invoke(initial_state)
```

---

#### NRAG-024: End-to-End Testing Suite

**Priority:** P0 - Critical  
**Token Estimate:** 12,000 tokens  
**Status:** To Do

**Description:**  
Create comprehensive test suite for the complete pipeline.

**Implementation:**

```python
# tests/test_e2e.py
import pytest
import logging
from unittest.mock import patch, MagicMock

from src.orchestrator.graph import AgenticRAG, create_workflow
from src.orchestrator.state import AgentState

logger = logging.getLogger(__name__)

class TestAgenticRAGE2E:
    """End-to-end tests for the Agentic RAG system."""
    
    @pytest.fixture
    def rag_system(self):
        """Create RAG system for testing."""
        return AgenticRAG(with_memory=False)
    
    def test_simple_query(self, rag_system):
        """Test basic query processing."""
        result = rag_system.query("What is RAG?")
        
        assert "answer" in result
        assert len(result["answer"]) > 0
        assert result["error"] is None
    
    def test_complex_query(self, rag_system):
        """Test multi-faceted query."""
        result = rag_system.query(
            "Compare the A2A protocol with MCP for multi-agent systems"
        )
        
        assert "answer" in result
        assert "sources" in result
        assert result["error"] is None
    
    def test_query_with_details(self, rag_system):
        """Test full state return."""
        state = rag_system.query_with_details("Explain agentic RAG")
        
        assert "sub_tasks" in state
        assert "retrieved_docs" in state
        assert "analysis" in state
        assert "final_answer" in state


class TestWorkflowComponents:
    """Test individual workflow components."""
    
    def test_workflow_creation(self):
        """Test workflow can be created."""
        app = create_workflow()
        assert app is not None
    
    def test_workflow_with_memory(self):
        """Test workflow with memory checkpointer."""
        app = create_workflow(with_memory=True)
        assert app is not None


# Test fixtures for mocking
@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    with patch('src.rag.vectorstore.get_vector_store') as mock:
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        mock.return_value = mock_vs
        yield mock_vs
```

---

#### NRAG-025: CLI Enhancement & Documentation

**Priority:** P1 - High  
**Token Estimate:** 8,000 tokens  
**Status:** To Do

**Description:**  
Enhance CLI and create usage documentation.

**Implementation:**

```python
# main.py (updated)
import argparse
import logging
import sys
from datetime import datetime

from src.utils.tracing import initialize_tracing
from src.orchestrator.graph import AgenticRAG
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_interactive():
    """Run interactive query mode."""
    print("\n" + "="*60)
    print("NOTION AGENTIC RAG - Interactive Mode")
    print("="*60)
    print("Type your questions below. Enter 'quit' or 'exit' to stop.\n")
    
    rag = AgenticRAG()
    
    while True:
        try:
            query = input("\n📝 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            print("\n⏳ Processing...\n")
            start = datetime.now()
            
            result = rag.query(query)
            
            duration = (datetime.now() - start).total_seconds()
            
            print("="*60)
            print("📖 RESPONSE")
            print("="*60)
            print(result["answer"])
            
            if result["sources"]:
                print("\n📚 SOURCES")
                print("-"*40)
                for src in result["sources"][:5]:
                    print(f"  [{src['id']}] {src['title']}")
                    if src.get('url'):
                        print(f"      {src['url']}")
            
            print(f"\n⏱️ Response time: {duration:.2f}s")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Notion Agentic RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "What is the A2A protocol?"
  python main.py --interactive
  python main.py --ingest --rebuild
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to process"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run document ingestion"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector store (with --ingest)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tracing
    initialize_tracing()
    
    if args.ingest:
        from src.ingest import run_ingestion
        run_ingestion(rebuild=args.rebuild)
        
    elif args.interactive:
        run_interactive()
        
    elif args.query:
        rag = AgenticRAG()
        result = rag.query(args.query)
        
        print("\n" + "="*60)
        print("RESPONSE")
        print("="*60)
        print(result["answer"])
        
        if result["error"]:
            print(f"\n⚠️ Error: {result['error']}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

---

#### NRAG-026: README Documentation

**Priority:** P1 - High  
**Token Estimate:** 10,000 tokens  
**Status:** To Do

**Description:**  
Create comprehensive README with setup and usage instructions.

---

### Session 6 Summary

| Item | Token Estimate | Cumulative |
|------|----------------|------------|
| NRAG-023: LangGraph Workflow | 15,000 | 15,000 |
| NRAG-024: E2E Testing | 12,000 | 27,000 |
| NRAG-025: CLI Enhancement | 8,000 | 35,000 |
| NRAG-026: README | 10,000 | 45,000 |
| **Session Buffer** | ~35,000 | 80,000 |

**Buffer Use:** Integration testing, debugging, final documentation

---

## Future Backlog

### Epic: Dynamic Tool Agents with A2A Protocol (Completed - Session 8)

> **Completed 2026-02-11.** All items implemented with 34/34 tests passing. See `src/tools/` for implementation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Fixed LangGraph Pipeline                      │
│  Planner ──> Researcher ──> Reasoner ──> Synthesiser            │
└─────────────────────────────────────────────────────────────────┘
                    │              │            │
                    ▼              ▼            ▼
         ┌─────────────────────────────────────────────┐
         │     Dynamic A2A Tool Agents (Optional)       │
         │  ┌───────────┐ ┌───────────┐ ┌────────────┐ │
         │  │ Web       │ │ Code      │ │ Citation   │ │
         │  │ Searcher  │ │ Executor  │ │ Validator  │ │
         │  └───────────┘ └───────────┘ └────────────┘ │
         │  ┌───────────┐ ┌───────────┐ ┌────────────┐ │
         │  │ Math      │ │ Diagram   │ │ Fact       │ │
         │  │ Solver    │ │ Generator │ │ Checker    │ │
         │  └───────────┘ └───────────┘ └────────────┘ │
         └─────────────────────────────────────────────┘
                         ▲
                         │
              A2A Agent Cards for Discovery
              (Capabilities, Input/Output Schemas)
```

| ID | Item | Status |
|----|------|--------|
| NRAG-027 | A2A Tool Agent Framework (`src/tools/base.py`, `src/tools/registry.py`) | [x] Done |
| NRAG-028 | Web Searcher (`src/tools/web_searcher.py`) | [x] Done |
| NRAG-029 | Code Executor (`src/tools/code_executor.py`) | [x] Done |
| NRAG-030 | Citation Validator (`src/tools/citation_validator.py`) | [x] Done |
| NRAG-031 | Math Solver (`src/tools/math_solver.py`) | [x] Done |
| NRAG-032 | Diagram Generator (`src/tools/diagram_generator.py`) | [x] Done |
| NRAG-033 | A2A Discovery & Invocation Client (`src/tools/client.py`) | [x] Done |

---

### Epic: User Interface (Partially Completed - Session 7)

> NRAG-050 through NRAG-054 completed in Session 7. Remaining items are backlog.

| ID | Item | Token Estimate | Priority | Status |
|----|------|----------------|----------|--------|
| NRAG-050 | Streamlit Chat Interface - Foundation | 15,000 | P1 | [x] Done |
| NRAG-051 | Chat Session Management | 10,000 | P1 | [x] Done |
| NRAG-052 | Source Display & Citation UI | 8,000 | P1 | [x] Done |
| NRAG-053 | Agent Progress Visualization | 12,000 | P2 | [x] Done |
| NRAG-054 | Settings & Configuration Panel | 6,000 | P2 | [x] Done |
| NRAG-055 | Extended Model Configuration | 8,000 | P2 | Backlog |
| NRAG-056 | Session Title Generation | 5,000 | P3 | Backlog |
| NRAG-057 | Session Auto-save & History | 6,000 | P2 | Backlog |
| NRAG-058 | Manual Reference Linking UI | 10,000 | P3 | Backlog |
| NRAG-059 | Knowledge Base Management Buttons | 8,000 | P2 | Backlog |

---

#### NRAG-050: Streamlit Chat Interface - Foundation

**Priority:** P1 - High  
**Token Estimate:** 15,000 tokens  
**Status:** Backlog

**Description:**  
Build a Streamlit-based chat interface that provides an intuitive way to interact with the RAG system. Replace the CLI with a web-based chat experience that shows the multi-agent workflow in action.

**Requirements:**

1. **Chat Input/Output**
   - Text input box for user queries
   - Chat message display with user/assistant roles
   - Streaming response support (if available)
   - Markdown rendering for formatted responses

2. **Basic Layout**
   - Sidebar for configuration and status
   - Main chat area
   - Footer with system status

3. **Integration**
   - Connect to existing `run_agentic_rag()` function
   - Capture and display pipeline execution
   - Error handling and user-friendly error messages

**Acceptance Criteria:**

- [ ] Streamlit app with chat interface (`st.chat_input`, `st.chat_message`)
- [ ] Query submission triggers RAG pipeline
- [ ] Response displayed in chat format with proper formatting
- [ ] Basic error handling and status messages
- [ ] App can be launched with `streamlit run app.py`

**Implementation:**

```python
# app.py
import streamlit as st
import logging
from datetime import datetime

from config.settings import settings
from src.orchestrator.graph import create_rag_graph
from src.utils.tracing import initialize_tracing

# Page config
st.set_page_config(
    page_title="Notion Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    initialize_tracing()
    st.session_state.graph = create_rag_graph()

# Sidebar
with st.sidebar:
    st.title("🤖 Notion RAG")
    st.markdown("**Multi-Agent Research Assistant**")
    st.divider()
    
    st.markdown("### System Status")
    st.success("✓ Graph Initialized")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("Research Assistant")
st.markdown("Ask questions about your knowledge base")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(f"- **{source.get('title', 'Untitled')}** ({source.get('source', 'Unknown')})")

# Chat input
if prompt := st.chat_input("Ask a question about your knowledge base..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process query
    with st.chat_message("assistant"):
        with st.spinner("🔍 Researching..."):
            try:
                # Initial state
                initial_state = {
                    "query": prompt,
                    "sub_tasks": [],
                    "planning_reasoning": "",
                    "retrieved_docs": [],
                    "retrieval_metadata": {},
                    "analysis": [],
                    "overall_assessment": "",
                    "final_answer": "",
                    "sources": [],
                    "error": None,
                    "current_agent": "start"
                }
                
                # Execute graph
                result = st.session_state.graph.invoke(initial_state)
                
                if result.get("error"):
                    st.error(f"❌ Error: {result['error']}")
                    response_text = f"I encountered an error: {result['error']}"
                    sources = []
                else:
                    response_text = result["final_answer"]
                    sources = result.get("sources", [])
                    
                    # Display response
                    st.markdown(response_text)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                st.markdown(f"- **{source.get('title', 'Untitled')}** ({source.get('source', 'Unknown')})")
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"System error: {str(e)}"
                st.error(f"❌ {error_msg}")
                logger.exception("Error processing query")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })
```

**Technical Notes:**

- Use `st.session_state` to persist chat history
- Leverage `st.chat_message()` and `st.chat_input()` for native chat UI
- Consider adding status indicators for each agent phase
- Markdown rendering supports citations and formatting from synthesiser

---

#### NRAG-051: Chat Session Management

**Priority:** P1 - High  
**Token Estimate:** 10,000 tokens  
**Status:** Backlog

**Description:**  
Implement session persistence and management features to allow users to save, load, and manage multiple chat sessions.

**Requirements:**

1. **Session Persistence**
   - Save chat history to local storage or database
   - Load previous sessions on app restart
   - Export/import chat sessions

2. **Multi-Session Support**
   - Create new sessions
   - Switch between sessions
   - Rename sessions
   - Delete sessions

3. **Session Metadata**
   - Track session creation time
   - Count messages per session
   - Store session-specific settings

**Acceptance Criteria:**

- [ ] Sessions saved to `./data/sessions/` directory
- [ ] Session selector in sidebar
- [ ] New session button
- [ ] Export session as JSON/Markdown
- [ ] Session metadata display (created date, message count)

**Implementation Approach:**

```python
# In app.py - add session management to sidebar

import json
from pathlib import Path

SESSIONS_DIR = Path("./data/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

def save_session(session_id: str, messages: list):
    """Save session to disk"""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    with open(session_file, "w") as f:
        json.dump({
            "id": session_id,
            "messages": messages,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }, f, indent=2)

def load_session(session_id: str) -> list:
    """Load session from disk"""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        with open(session_file) as f:
            data = json.load(f)
            return data.get("messages", [])
    return []

# Add to sidebar
with st.sidebar:
    st.markdown("### Sessions")
    
    # Session selector
    sessions = [f.stem for f in SESSIONS_DIR.glob("*.json")]
    if sessions:
        selected_session = st.selectbox("Load Session", ["Current"] + sessions)
        if selected_session != "Current":
            st.session_state.messages = load_session(selected_session)
            st.rerun()
    
    if st.button("💾 Save Session"):
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_session(session_id, st.session_state.messages)
        st.success(f"Saved as {session_id}")
```

---

#### NRAG-052: Source Display & Citation UI

**Priority:** P1 - High  
**Token Estimate:** 8,000 tokens  
**Status:** Backlog

**Description:**  
Enhance the display of sources and citations with rich metadata, clickable links, and visual hierarchy.

**Requirements:**

1. **Rich Source Cards**
   - Title with link to original source
   - Authors (for papers)
   - Publication date
   - Source type badge (Arxiv, Notion, etc.)
   - Snippet of relevant content

2. **Citation Highlighting**
   - In-text citation markers (e.g., [1], [2])
   - Hover tooltips showing source preview
   - Click to jump to source details

3. **Source Filtering**
   - Filter by source type
   - Filter by date range
   - Search within sources

**Acceptance Criteria:**

- [ ] Sources displayed as expandable cards with metadata
- [ ] Clickable links to original sources (Arxiv URLs, Notion pages)
- [ ] Visual distinction between source types
- [ ] Citation numbers linked to source references
- [ ] Source preview on hover (if feasible in Streamlit)

**Implementation Approach:**

```python
# Enhanced source display component

def display_source_card(source: dict, index: int):
    """Display a rich source card"""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Title with link
            title = source.get("title", "Untitled")
            url = source.get("url") or source.get("arxiv_url") or source.get("notion_url")
            
            if url:
                st.markdown(f"**[{index+1}. {title}]({url})**")
            else:
                st.markdown(f"**{index+1}. {title}**")
            
            # Authors
            if authors := source.get("authors"):
                st.caption(f"👤 {', '.join(authors[:3])}" + (" et al." if len(authors) > 3 else ""))
            
            # Snippet
            if snippet := source.get("snippet"):
                st.text(snippet[:200] + "..." if len(snippet) > 200 else snippet)
        
        with col2:
            # Source badge
            source_type = source.get("source", "unknown")
            badge_color = {
                "arxiv": "🔬",
                "notion": "📝",
                "other": "🔗"
            }.get(source_type, "📄")
            
            st.markdown(f"{badge_color} {source_type.upper()}")
            
            # Date
            if pub_date := source.get("published"):
                st.caption(f"📅 {pub_date}")
        
        st.divider()

# Use in main chat
if sources:
    st.markdown("### 📚 Sources")
    for i, source in enumerate(sources):
        display_source_card(source, i)
```

---

#### NRAG-053: Agent Progress Visualization

**Priority:** P2 - Medium  
**Token Estimate:** 12,000 tokens  
**Status:** Backlog

**Description:**  
Add real-time visualization of the multi-agent workflow, showing which agent is currently active and the progress through the pipeline.

**Requirements:**

1. **Agent Status Display**
   - Visual indicator of current agent
   - Progress bar or stepper UI
   - Agent-specific icons/colors

2. **Real-time Updates**
   - Stream agent transitions
   - Show intermediate results (sub-tasks, retrieved docs count)
   - Estimated time remaining

3. **Workflow Diagram**
   - Visual representation of the agent pipeline
   - Highlight current stage
   - Show decision points (if any)

**Acceptance Criteria:**

- [ ] Status bar showing current agent (Planner → Researcher → Reasoner → Synthesiser)
- [ ] Progress indicator during execution
- [ ] Display intermediate results (e.g., "Found 5 documents", "Generated 3 sub-tasks")
- [ ] Agent-specific status messages
- [ ] Visual workflow diagram in sidebar

**Implementation Approach:**

```python
# Agent progress component

def show_agent_progress(current_agent: str):
    """Display current agent in pipeline"""
    agents = ["planner", "researcher", "reasoner", "synthesiser"]
    icons = ["🎯", "🔍", "🧠", "✍️"]
    
    cols = st.columns(len(agents))
    
    for i, (agent, icon) in enumerate(zip(agents, icons)):
        with cols[i]:
            if agent == current_agent:
                st.markdown(f"### {icon}")
                st.markdown(f"**{agent.title()}**")
                st.progress(1.0)
            elif agents.index(current_agent) > i:
                st.markdown(f"### ✓")
                st.markdown(f"~~{agent.title()}~~")
            else:
                st.markdown(f"### {icon}")
                st.markdown(f"{agent.title()}")
                st.progress(0.0)

# In main chat processing
progress_placeholder = st.empty()

# Hook into graph execution to update progress
# (Requires modifying graph to emit progress events or polling state)
with st.spinner("Processing..."):
    show_agent_progress("planner")
    # ... continue with execution
```

**Technical Notes:**

- May require graph modifications to emit progress events
- Consider using `st.status()` context manager for nested progress
- LangSmith integration could provide real-time agent states

---

#### NRAG-054: Settings & Configuration Panel

**Priority:** P2 - Medium  
**Token Estimate:** 6,000 tokens  
**Status:** Backlog

**Description:**  
Add a settings panel in the sidebar to allow runtime configuration of RAG parameters without editing `.env` files.

**Requirements:**

1. **Model Selection**
   - Choose models for each agent
   - Adjust temperature settings
   - Toggle reasoning model

2. **Retrieval Settings**
   - Adjust k (number of documents)
   - Rerank top N
   - Similarity threshold

3. **Display Settings**
   - Show/hide intermediate results
   - Verbose mode toggle
   - Theme selection

**Acceptance Criteria:**

- [ ] Settings panel in sidebar with collapsible sections
- [ ] Model selection dropdowns for each agent
- [ ] Sliders for temperature and retrieval parameters
- [ ] Settings persisted in session state
- [ ] Apply button to update configuration
- [ ] Reset to defaults button

**Implementation Approach:**

```python
# In sidebar
with st.sidebar:
    with st.expander("⚙️ Settings"):
        st.markdown("#### Models")
        planner_model = st.selectbox(
            "Planner Model",
            ["command-r-08-2024", "command-r-plus-08-2024"],
            index=0
        )
        
        st.markdown("#### Retrieval")
        retrieval_k = st.slider("Documents to Retrieve", 5, 20, 10)
        rerank_top_n = st.slider("Rerank Top N", 3, 10, 5)
        
        st.markdown("#### Display")
        show_intermediate = st.checkbox("Show Intermediate Results", value=False)
        verbose_mode = st.checkbox("Verbose Logging", value=False)
        
        if st.button("Apply Settings"):
            # Update settings in session state
            st.session_state.settings = {
                "planner_model": planner_model,
                "retrieval_k": retrieval_k,
                "rerank_top_n": rerank_top_n,
                "show_intermediate": show_intermediate,
                "verbose": verbose_mode
            }
            st.success("Settings updated!")
```

---

#### NRAG-055: Extended Model Configuration

**Priority:** P2 - Medium
**Token Estimate:** 8,000 tokens
**Status:** Backlog

**Description:**
Enable extensive model selection for all four sub-agents (Planner, Researcher, Reasoner, Synthesiser), allowing the user to mix and match models for optimal performance and cost.

**Requirements:**

1.  **Granular Control**
    *   Selectors for each agent type in the Settings panel
    *   Support differing model families if compatible

2.  **Configuration Persistence**
    *   Save agent specific model choices in session state
    *   Update LLM factory to respect granular settings

**Implementation Approach:**

```python
# app.py settings expansion
with st.expander("🤖 Agent Models"):
    col1, col2 = st.columns(2)
    with col1:
        planner_model = st.selectbox("Planner", ["command-r", "command-r-plus"], key="model_planner")
        researcher_model = st.selectbox("Researcher", ["command-r", "command-r-plus"], key="model_researcher")
    with col2:
        reasoner_model = st.selectbox("Reasoner", ["command-r", "command-r-plus", "command-a-reasoning"], key="model_reasoner")
        synthesiser_model = st.selectbox("Synthesiser", ["command-r-plus", "command-r"], key="model_synthesiser")
```

---

#### NRAG-056: Session Title Generation

**Priority:** P3 - Low
**Token Estimate:** 5,000 tokens
**Status:** Backlog

**Description:**
Implement a lightweight "Title Generator" model (or use a smaller LLM) to automatically generate descriptive titles for chat sessions based on the first query or conversation content.

**Requirements:**

1.  **Automatic Generation**
    *   Trigger title generation after the first user prompt
    *   Update session name in sidebar without page reload

2.  **Model Selection**
    *   Use a fast, cheap model (e.g., `command-r-08-2024` or lighter) specific for this task
    *   Avoid using the heavy "Reasoner" model for simple titling

**Implementation Approach:**

```python
# src/utils/session_manager.py
def generate_session_title(first_prompt: str) -> str:
    llm = get_agent_llm("planner") # Reuse planner or lighter model
    response = llm.invoke(f"Summarize this query into a 3-5 word title: {first_prompt}")
    return response.content.strip().strip('"')
```

---

#### NRAG-057: Session Auto-save & History

**Priority:** P2 - Medium
**Token Estimate:** 6,000 tokens
**Status:** Backlog

**Description:**
Enhance session management to automatically save chat history to disk/database immediately upon sending a query and receiving a response, preventing data loss during browser refreshes.

**Requirements:**

1.  **Autosave Trigger**
    * Remove the sidebar Save button for sessions.
    * Auto-save immediately when a user submits a query (persist prompt).
    * Auto-save again when assistant output generation completes (persist response).
    * Ensure behavior is consistent in both streaming and non-streaming flows.

---

#### NRAG-058: Manual Reference Linking UI

**Priority:** P3 - Low
**Token Estimate:** 10,000 tokens
**Status:** Backlog

**Description:**
Allow users to manually link or tag references in the chat interface, enabling them to associate specific sources with particular parts of the conversation.

**Requirements:**

1.  **Manual Source Tagging**
    *   UI control to manually add/link sources to a response
    *   Search/browse available sources from knowledge base
    *   Associate sources with specific response sections

2.  **Reference Management**
    *   View linked references for each message
    *   Edit or remove manual links
    *   Visual distinction between auto-linked and manually-linked sources

**Acceptance Criteria:**

- [ ] Users can manually add source links to responses
- [ ] Manual links are persisted in session history
- [ ] Manual links are visually distinguished from automatic citations
- [ ] Interface for managing (add/remove/edit) manual references

---

#### NRAG-059: Knowledge Base Management Buttons

**Priority:** P2 - Medium
**Token Estimate:** 8,000 tokens
**Status:** Backlog

**Description:**
Add interactive buttons to the Streamlit sidebar for managing the knowledge base and testing system connections, providing easy access to CLI-only features currently available in `main.py`.

**Current State:**
The following management operations are only available via CLI:
- `python main.py --ingest` - Ingest knowledge base from Notion and arXiv
- `python main.py --ingest --rebuild` - Rebuild vector store from scratch
- `python main.py --test-conn` - Test connections to LangSmith and Cohere

**Requirements:**

1.  **Test Connection Button**
    *   Tests connections to LangSmith (optional) and Cohere embeddings
    *   Displays results inline with success/failure indicators
    *   Shows vector dimension on successful embedding test

2.  **Ingest Knowledge Base Button**
    *   Runs the document ingestion pipeline
    *   Loads documents from Notion and arXiv into vector store
    *   Shows progress indicator during ingestion
    *   Displays summary of documents ingested

3.  **Rebuild Vector Store Button**
    *   Clears existing vector store and re-ingests all documents
    *   Includes confirmation dialog ("Are you sure?")
    *   Shows progress with status updates
    *   Displays completion summary

**Implementation Approach:**

```python
# app.py - Add to sidebar between Agent Pipeline and Settings
st.divider()

# System Management
with st.expander("🛠️ System Management"):
    st.markdown("#### Connection Test")
    
    if st.button("🔌 Test Connection", use_container_width=True):
        with st.spinner("Testing connections..."):
            try:
                # Test LangSmith (optional)
                st.write("🔹 LangSmith: Checking...")
                try:
                    initialize_tracing()
                    st.success("   ✅ Tracing initialized")
                except Exception as e:
                    st.warning(f"   ⚠️ LangSmith: {e}")
                
                # Test Cohere Embeddings
                st.write("🔹 Cohere Embeddings: Testing...")
                emb = get_embeddings()
                vec = emb.embed_query("ping")
                st.success(f"   ✅ Success (Vector dim: {len(vec)})")
                
            except Exception as e:
                st.error(f"❌ Connection test failed: {e}")
    
    st.divider()
    st.markdown("#### Knowledge Base")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Ingest", use_container_width=True):
            with st.spinner("Ingesting documents..."):
                try:
                    run_ingestion(rebuild=False)
                    st.success("✅ Ingestion completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")
    
    with col2:
        if st.button("🔄 Rebuild", use_container_width=True):
            # Show confirmation
            if "confirm_rebuild" not in st.session_state:
                st.session_state.confirm_rebuild = False
            
            if not st.session_state.confirm_rebuild:
                st.warning("⚠️ This will delete all existing data!")
                if st.button("⚠️ Confirm Rebuild"):
                    st.session_state.confirm_rebuild = True
                    st.rerun()
            else:
                with st.spinner("Rebuilding vector store..."):
                    try:
                        run_ingestion(rebuild=True)
                        st.success("✅ Rebuild completed!")
                        st.session_state.confirm_rebuild = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Rebuild failed: {e}")
                        st.session_state.confirm_rebuild = False
```

**Acceptance Criteria:**

- [ ] Test Connection button displays LangSmith and Cohere status
- [ ] Ingest button successfully runs ingestion pipeline
- [ ] Rebuild button includes confirmation dialog
- [ ] All operations show progress indicators during execution
- [ ] Success/error messages are clear and actionable
- [ ] Buttons are disabled during operation to prevent duplicate runs
- [ ] Operations complete without breaking the Streamlit app state

**Technical Notes:**

- Location: Add to `app.py` sidebar, between "Agent Pipeline" expander and "Settings" expander (around line 620)
- Required imports: `from src.ingest import run_ingestion`, `from src.rag.embeddings import get_embeddings`
- Use `st.spinner()` for long-running operations
- Consider adding vector store statistics (document count) display
- Test connection logic from `main.py:81-103`
- Ingestion logic from `src/ingest.py:10-48`

---

### Epic: Model Ecosystem Expansion

| ID | Item | Token Estimate | Priority |
|----|------|----------------|----------|
| NRAG-060 | Multi-Provider LLM Abstraction | 15,000 | P2 |
| NRAG-061 | OpenAI & Anthropic Integration | 12,000 | P2 |
| NRAG-062 | Gemini, Qwen & Deepseek Support | 15,000 | P2 |

#### NRAG-060: Multi-Provider LLM Abstraction

**Priority:** P2 - Medium
**Token Estimate:** 15,000 tokens
**Status:** Backlog

**Description:**
Refactor `src/agents/llm_factory.py` to support multiple LLM providers beyond Cohere, using a unified factory pattern or LangChain's adapter capabilities. Use the latest models for heavy-weight (HW) and medium-weight (MW) tasks, and use older versions for light-weight (LW) tasks.

**Requirements:**

1.  **Abstract Factory**
    *   Support `Provider` enum (Cohere, OpenAI, Anthropic, etc.)
    *   Unified interface for Model parameters (temperature, max_tokens)

#### NRAG-061: OpenAI & Anthropic Integration

**Priority:** P2 - Medium
**Token Estimate:** 12,000 tokens
**Status:** Backlog

**Description:**
Implement specific integration for OpenAI (GPT-5 for HW/MW, GPT-4o for LW) and Anthropic (Claude 4.5 Opus for HW, Sonnet for MW and Haiku for LW)

**Requirements:**

1.  **OpenAI**
    *   Support `langchain-openai`
    *   Map `o1` reasoning capabilities to "Reasoner" agent

2.  **Anthropic**
    *   Support `langchain-anthropic`
    *   Leverage large context windows for "Synthesiser"

#### NRAG-062: Gemini, Qwen & Deepseek Support

**Priority:** P2 - Medium
**Token Estimate:** 15,000 tokens
**Status:** Backlog

**Description:**
Expand support to Google Gemini (Pro/Flash) and open-weights models like Qwen 2.5 and DeepSeek R1 (via API or local hosting).

**Requirements:**

1.  **Google Gemini**
    *   Support `langchain-google-genai`
    
2.  **Open Models (Qwen/Deepseek)**
    *   Support generic OpenAI-compatible endpoints (e.g., OpenRouter, vLLM)
    *   Handle specific prompting requirements for DeepSeek R1 ("Thinking" tags)

---

### Epic: AgentLightning Integration for Prompt Optimization

**Goal:** Use Microsoft's AgentLightning to collect agent trajectories, gather user feedback, and iteratively optimize agent prompts via Automatic Prompt Optimization (APO).

> **Constraint:** All agents use hosted LLM APIs (Cohere, OpenAI, Anthropic, etc.) — RL weight training (PPO/DPO/GRPO) is **not applicable** since we have no access to model weights. AgentLightning's **APO** algorithm works with API-only models: it uses an LLM to critique collected trajectories and rewrite prompts, requiring no weight access.

**Overview:**
Integrate AgentLightning to collect trajectory data from the 4-agent pipeline, gather user feedback as reward signals, and use APO to automatically improve agent system prompts over time.

**Key Benefits:**
- Automatic prompt improvement based on real deployment data (no manual prompt engineering)
- Framework-agnostic integration (works with existing LangChain/LangGraph setup)
- Selective optimization of individual agents or the entire multi-agent system
- Collection of interaction data for analysis and evaluation

**Architecture Approach:**

```
┌────────────────────────────────────────────────────────────┐
│             Existing Agentic RAG System                     │
│   Planner → Researcher → Reasoner → Synthesiser            │
│              (LangChain/LangGraph)                          │
└────────────────────────────────────────────────────────────┘
                          │
                          │ emit_span() / emit_reward()
                          ▼
┌────────────────────────────────────────────────────────────┐
│              AgentLightning Layer                          │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │ Lightning Store  │ ←──→ │ Lightning Client  │          │
│  │ (Trajectories)   │      │ (LLM Proxy)       │          │
│  └──────────────────┘      └──────────────────┘           │
│         │                            │                     │
│         │ Trajectory Data            │ Optimized Prompts  │
│         ▼                            ▼                     │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │ APO Engine       │      │ User Feedback     │          │
│  │ (Prompt Rewrite) │      │ & Rewards         │          │
│  └──────────────────┘      └──────────────────┘           │
└────────────────────────────────────────────────────────────┘
```

**Implementation Strategy:**

| Session | Focus | Token Budget | Key Deliverables |
|---------|-------|--------------|------------------|
| Session 9 | Foundation & Data Collection | ~80,000 | Install AGL, setup emitters, trajectory tracking |
| Session 10 | Feedback & Reward System | ~80,000 | User feedback UI, automatic rewards, signal pipeline |
| Session 11 | APO & Evaluation | ~80,000 | APO prompt optimization, evaluation framework, analytics |

---

### Session 9: AgentLightning Foundation

**Session Token Budget:** ~80,000 tokens  
**Focus:** Install AgentLightning, setup basic infrastructure, and integrate emitters

---

#### NRAG-063: AgentLightning Installation & Configuration

**Priority:** P1 - High  
**Token Estimate:** 8,000 tokens  
**Status:** Backlog

**Description:**  
Install AgentLightning and configure the basic setup for the project.

**Acceptance Criteria:**

- [ ] <cite index="1-6">Install `agentlightning` package via pip</cite>
- [ ] Update `pyproject.toml` with new dependency
- [ ] Create configuration module for AgentLightning settings
- [ ] Update `.env.example` with AgentLightning variables
- [ ] Documentation of setup process

**Implementation:**

```toml
# pyproject.toml additions
[project]
dependencies = [
    # ... existing dependencies
    "agentlightning>=0.2.1",  # Latest stable version
]
```

```python
# config/agl_settings.py
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class AgentLightningConfig(BaseSettings):
    """Configuration for AgentLightning integration."""
    
    # Enable/disable AgentLightning
    enabled: bool = Field(default=False, env="AGL_ENABLED")
    
    # Storage configuration
    store_type: str = Field(default="file", env="AGL_STORE_TYPE")  # file, redis, mongodb
    store_path: str = Field(default="./data/agl_store", env="AGL_STORE_PATH")
    
    # Training configuration
    enable_training: bool = Field(default=False, env="AGL_ENABLE_TRAINING")
    training_algorithm: str = Field(default="ppo", env="AGL_TRAINING_ALGORITHM")  # ppo, dpo, grpo
    
    # Agent selection for optimization
    # Planner & Researcher: Track but don't optimize (use smaller Command-R model)
    # Reasoner & Synthesiser: Optimize (use larger Command-R+/Command-A-Reasoning models)
    optimize_planner: bool = Field(default=False, env="AGL_OPTIMIZE_PLANNER")
    optimize_researcher: bool = Field(default=False, env="AGL_OPTIMIZE_RESEARCHER")
    optimize_reasoner: bool = Field(default=True, env="AGL_OPTIMIZE_REASONER")
    optimize_synthesiser: bool = Field(default=True, env="AGL_OPTIMIZE_SYNTHESISER")
    
    # Reward configuration
    enable_user_feedback: bool = Field(default=True, env="AGL_ENABLE_USER_FEEDBACK")
    automatic_reward: bool = Field(default=True, env="AGL_AUTOMATIC_REWARD")
    
    # LLM Proxy configuration
    proxy_enabled: bool = Field(default=False, env="AGL_PROXY_ENABLED")
    proxy_port: int = Field(default=8000, env="AGL_PROXY_PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

agl_config = AgentLightningConfig()
```

```bash
# .env.example additions
# ===========================================
# AgentLightning Configuration (Optional)
# ===========================================

# Enable AgentLightning integration for agent training
AGL_ENABLED=false

# Storage type: file, redis, mongodb
AGL_STORE_TYPE=file
AGL_STORE_PATH=./data/agl_store

# Training configuration
AGL_ENABLE_TRAINING=false
AGL_TRAINING_ALGORITHM=ppo

# Select which agents to optimize (Reasoner & Synthesiser recommended)
AGL_OPTIMIZE_PLANNER=false
AGL_OPTIMIZE_RESEARCHER=false
AGL_OPTIMIZE_REASONER=true
AGL_OPTIMIZE_SYNTHESISER=true

# Reward configuration
AGL_ENABLE_USER_FEEDBACK=true
AGL_AUTOMATIC_REWARD=true

# LLM Proxy (for training mode)
AGL_PROXY_ENABLED=false
AGL_PROXY_PORT=8000
```

---

#### NRAG-064: AgentLightning Store & Emitter Setup

**Priority:** P1 - High  
**Token Estimate:** 12,000 tokens  
**Status:** Backlog

**Description:**  
<cite index="6-3,6-15">Implement the Lightning Store (unified interface for AgentLightning's core storage) and emitter infrastructure to capture agent interactions.</cite>

**Acceptance Criteria:**

- [ ] <cite index="6-3">Initialize Lightning Store with unified interface</cite>
- [ ] <cite index="6-4,6-16">Create emitter wrapper for agent calls that emits any objects as spans to the store</cite>
- [ ] Implement span tracking for each agent
- [ ] Store metadata (query, response, timing, model used)
- [ ] Basic querying and inspection utilities

**Implementation:**

```python
# src/agl/__init__.py
"""AgentLightning integration for continuous learning."""

from src.agl.store_manager import get_store_manager, AGLStoreManager
from src.agl.emitter import emit_agent_span, emit_user_reward
from src.agl.trajectory import TrajectoryContext

__all__ = [
    "get_store_manager",
    "AGLStoreManager",
    "emit_agent_span",
    "emit_user_reward",
    "TrajectoryContext",
]
```

```python
# src/agl/store_manager.py
import logging
from typing import Optional
from agentlightning.store import LightningStore, FileStore
from config.agl_settings import agl_config

logger = logging.getLogger(__name__)

class AGLStoreManager:
    """
    Manages AgentLightning store for capturing agent interactions.
    """
    
    def __init__(self):
        self.store: Optional[LightningStore] = None
        self.enabled = agl_config.enabled
        
        if self.enabled:
            self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the Lightning Store."""
        try:
            if agl_config.store_type == "file":
                self.store = FileStore(path=agl_config.store_path)
                logger.info(f"Initialized FileStore at {agl_config.store_path}")
            else:
                raise ValueError(f"Unsupported store type: {agl_config.store_type}")
        except Exception as e:
            logger.error(f"Failed to initialize AgentLightning store: {e}")
            self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if AgentLightning is enabled and store is initialized."""
        return self.enabled and self.store is not None
    
    def get_store(self) -> Optional[LightningStore]:
        """Get the Lightning Store instance."""
        return self.store
    
    def get_trajectory_count(self) -> int:
        """Get the number of stored trajectories."""
        if not self.is_enabled():
            return 0
        try:
            return len(self.store.list_trajectories())
        except Exception as e:
            logger.error(f"Error getting trajectory count: {e}")
            return 0

# Global store manager instance
_store_manager: Optional[AGLStoreManager] = None

def get_store_manager() -> AGLStoreManager:
    """Get the singleton store manager."""
    global _store_manager
    if _store_manager is None:
        _store_manager = AGLStoreManager()
    return _store_manager
```

```python
# src/agl/emitter.py
import logging
from typing import Any, Dict, Optional
from functools import wraps
import time
from agentlightning import emit_span, emit_reward

from src.agl.store_manager import get_store_manager
from config.agl_settings import agl_config

logger = logging.getLogger(__name__)

def emit_agent_span(
    agent_name: str,
    model: str,
    should_optimize: bool = False
):
    """
    Decorator to emit AgentLightning spans for agent calls.
    
    Minimal code change required: just wrap the existing agent function.
    
    Args:
        agent_name: Name of the agent (planner, researcher, reasoner, synthesiser)
        model: Model name used by the agent
        should_optimize: Whether this agent should be optimized with RL
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
            store_manager = get_store_manager()
            
            # If AgentLightning is disabled, just call the function normally
            if not store_manager.is_enabled():
                return func(state, *args, **kwargs)
            
            # Extract query for context
            query = state.get("query", "")
            
            # Prepare input for emission
            span_input = {
                "agent": agent_name,
                "query": query,
                "state_keys": list(state.keys()),
            }
            
            # Start timing
            start_time = time.time()
            
            try:
                # Call the agent function
                result = func(state, *args, **kwargs)
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Prepare output for emission
                span_output = {
                    "agent": agent_name,
                    "duration": duration,
                    "success": result.get("error") is None,
                    "output_keys": list(result.keys()),
                }
                
                # Emit the span if optimizable
                if should_optimize:
                    emit_span(
                        name=f"{agent_name}_agent",
                        input=span_input,
                        output=span_output,
                        model=model,
                        metadata={
                            "agent_type": agent_name,
                            "optimizable": True,
                            "duration_seconds": duration,
                        }
                    )
                else:
                    # Just log for non-optimizable agents
                    logger.debug(f"{agent_name} completed in {duration:.2f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Error in {agent_name}: {e}")
                
                # Emit error span
                emit_span(
                    name=f"{agent_name}_agent_error",
                    input=span_input,
                    output={"error": str(e), "duration": duration},
                    model=model,
                    metadata={"agent_type": agent_name, "error": True}
                )
                
                raise
        
        return wrapper
    return decorator


def emit_user_reward(trajectory_id: str, reward: float, feedback: Optional[str] = None):
    """
    Emit user feedback as reward signal.
    
    Args:
        trajectory_id: ID of the trajectory to reward
        reward: Reward value (typically -1 to 1)
        feedback: Optional textual feedback
    """
    store_manager = get_store_manager()
    
    if not store_manager.is_enabled():
        logger.debug("AgentLightning disabled, skipping reward emission")
        return
    
    try:
        emit_reward(
            trajectory_id=trajectory_id,
            reward=reward,
            metadata={"user_feedback": feedback} if feedback else {}
        )
        logger.info(f"Emitted reward {reward} for trajectory {trajectory_id}")
    except Exception as e:
        logger.error(f"Error emitting reward: {e}")
```

---

#### NRAG-065: Agent Integration with Emitters

**Priority:** P0 - Critical  
**Token Estimate:** 10,000 tokens  
**Status:** Backlog

**Description:**  
<cite index="4-25">Integrate AgentLightning emitters into existing agent implementations without changing agent code.</cite>

**Acceptance Criteria:**

- [ ] Wrap existing agent functions with `emit_agent_span` decorator
- [ ] Mark Reasoner and Synthesiser as optimizable (using larger models)
- [ ] Planner and Researcher tracked but not optimized
- [ ] Ensure backward compatibility (works with AGL disabled)
- [ ] Test with AGL enabled and disabled
- [ ] <cite index="1-1">Verify compatibility with existing LangChain/LangGraph setup</cite>

**Example Integration:**

```python
# src/agents/reasoner.py (modified)
from src.agl.emitter import emit_agent_span
from config.agl_settings import agl_config

# ... existing imports and code ...

class ReasonerAgent:
    # ... existing implementation ...
    
    @agent_trace("reasoner", model="command-a-reasoning")
    @emit_agent_span(
        agent_name="reasoner",
        model="command-a-reasoning",
        should_optimize=agl_config.optimize_reasoner
    )
    def __call__(self, state: AgentState) -> AgentState:
        # Existing implementation unchanged
        # ...
        pass
```

---

#### NRAG-066: Trajectory Tracking

**Priority:** P1 - High  
**Token Estimate:** 10,000 tokens  
**Status:** Backlog

**Description:**  
Implement trajectory tracking for complete user interactions from query to response.

**Acceptance Criteria:**

- [ ] Create trajectory context manager
- [ ] Track complete query → response cycles
- [ ] Store intermediate agent outputs
- [ ] Link spans to trajectories
- [ ] Implement trajectory querying utilities

**Implementation:**

```python
# src/agl/trajectory.py
import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from agentlightning import start_trajectory, end_trajectory

from src.agl.store_manager import get_store_manager

logger = logging.getLogger(__name__)

class TrajectoryContext:
    """
    Context manager for AgentLightning trajectories.
    
    Usage:
        with TrajectoryContext(query="What is RAG?") as trajectory:
            result = rag_system.query(query)
    """
    
    def __init__(self, query: str, metadata: Optional[Dict[str, Any]] = None):
        self.trajectory_id = str(uuid.uuid4())
        self.query = query
        self.metadata = metadata or {}
        self.start_time = None
        self.end_time = None
        self.store_manager = get_store_manager()
    
    def __enter__(self):
        if not self.store_manager.is_enabled():
            return self
        
        self.start_time = datetime.now()
        
        try:
            start_trajectory(
                trajectory_id=self.trajectory_id,
                metadata={
                    "query": self.query,
                    "start_time": self.start_time.isoformat(),
                    **self.metadata
                }
            )
            logger.info(f"Started trajectory {self.trajectory_id}")
        except Exception as e:
            logger.error(f"Error starting trajectory: {e}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.store_manager.is_enabled():
            return
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        try:
            end_trajectory(
                trajectory_id=self.trajectory_id,
                metadata={
                    "end_time": self.end_time.isoformat(),
                    "duration_seconds": duration,
                    "success": exc_type is None,
                    "error": str(exc_val) if exc_val else None
                }
            )
            logger.info(f"Ended trajectory {self.trajectory_id} (duration: {duration:.2f}s)")
        except Exception as e:
            logger.error(f"Error ending trajectory: {e}")
```

---

### Session 9: User Feedback & Reward System

**Session Token Budget:** ~80,000 tokens  
**Focus:** Implement user feedback collection and reward signal generation

---

#### NRAG-067: User Feedback Interface in Streamlit

**Priority:** P0 - Critical  
**Token Estimate:** 12,000 tokens  
**Status:** Backlog

**Description:**  
Add user feedback collection UI to the Streamlit interface for reward signal generation.

**Acceptance Criteria:**

- [ ] Thumbs up/down feedback buttons after each response
- [ ] Star rating system (1-5 stars)
- [ ] Optional text feedback field
- [ ] Visual feedback confirmation
- [ ] Link feedback to trajectory ID
- [ ] Store feedback in AgentLightning store

**Implementation:**
Add feedback widget below each response in `app.py` that:
1. Displays rating options
2. Captures user input
3. Converts to reward signal (-1 to 1)
4. Emits reward to AgentLightning

---

#### NRAG-068: Automatic Reward Generation

**Priority:** P1 - High  
**Token Estimate:** 14,000 tokens  
**Status:** Backlog

**Description:**  
<cite index="10-7">Implement automatic reward signal generation using Automatic Intermediate Rewarding that converts runtime signals into dense feedback, reducing sparse rewards in long workflows.</cite>

**Acceptance Criteria:**

- [ ] Success/failure detection based on errors
- [ ] Response quality metrics (length, sources cited, coherence)
- [ ] Retrieval quality (documents found, relevance scores)
- [ ] Latency-based penalties for slow responses
- [ ] Confidence scoring from Reasoner agent
- [ ] Intermediate reward for each agent stage

**Example Reward Signals:**

```python
# Automatic reward calculation examples

# 1. Retrieval Quality (Researcher Agent)
#    - Found relevant documents: +0.5
#    - No documents found: -0.5
#    - High rerank scores: +0.3

# 2. Analysis Quality (Reasoner Agent)  
#    - High confidence analysis: +0.4
#    - Identified contradictions: +0.2
#    - Low confidence: -0.2

# 3. Response Quality (Synthesiser Agent)
#    - Cited sources: +0.5
#    - Complete answer: +0.5
#    - Too short/incomplete: -0.3

# 4. Overall Success
#    - No errors: +0.5
#    - Error occurred: -1.0
```

---

#### NRAG-069: Reward Signal Pipeline Integration

**Priority:** P0 - Critical  
**Token Estimate:** 10,000 tokens  
**Status:** Backlog

**Description:**  
Integrate reward calculation and emission into the main workflow.

**Acceptance Criteria:**

- [ ] Automatic reward calculation after trajectory completion
- [ ] Reward emission to AgentLightning store
- [ ] User feedback override of automatic rewards
- [ ] Intermediate reward emission for each agent
- [ ] Logging and monitoring of reward signals
- [ ] Configurable reward weights and thresholds

---

### Session 11: APO Prompt Optimization & Evaluation

**Session Token Budget:** ~80,000 tokens
**Focus:** Automatic Prompt Optimization and evaluation framework

---

#### NRAG-070: APO Prompt Optimization

**Priority:** P1 - High
**Token Estimate:** 16,000 tokens
**Status:** Backlog

**Description:**
Use AgentLightning's Automatic Prompt Optimization (APO) to iteratively improve agent system prompts using collected trajectories and reward signals. APO works with API-only models — a critique LLM evaluates trajectory rollouts and rewrites prompts to improve performance.

**Acceptance Criteria:**

- [ ] APO configuration with AgentLightning
- [ ] Critique LLM setup (e.g., GPT-4.1-mini or equivalent)
- [ ] Prompt versioning and rollback
- [ ] Optimization metrics and logging (reward improvement per iteration)
- [ ] Integration with existing agent prompts (Planner, Researcher, Reasoner, Synthesiser)
- [ ] Validation dataset to measure prompt quality before/after

**Example APO Script:**

```python
# scripts/optimize_prompts.py
from agentlightning import APOTrainer
from src.agl.store_manager import get_store_manager
from config.agl_settings import agl_config

def optimize_reasoner_prompt():
    """Optimize the Reasoner agent's system prompt using APO."""

    store = get_store_manager().get_store()

    trainer = APOTrainer(
        store=store,
        critique_model="gpt-4.1-mini",   # LLM for trajectory critique
        rewrite_model="gpt-4.1-mini",    # LLM for prompt rewriting
    )

    # Run APO optimization loop
    trainer.optimize(
        agent_name="reasoner_agent",
        num_iterations=10,
        validation_tasks=load_validation_dataset(),
    )

    # Export optimized prompt
    trainer.export_prompt("./prompts/reasoner_v2.txt")
```

---

#### NRAG-071: Evaluation Framework

**Priority:** P1 - High  
**Token Estimate:** 12,000 tokens  
**Status:** Backlog

**Description:**  
Build evaluation framework to measure agent performance before and after training.

**Acceptance Criteria:**

- [ ] Test query dataset covering diverse topics
- [ ] Evaluation metrics (accuracy, F1, BLEU, response quality)
- [ ] Automated evaluation pipeline
- [ ] Comparison between baseline and trained models
- [ ] Visualization of improvement over time
- [ ] Statistical significance testing

**Metrics to Track:**
- Task success rate
- Average response quality score
- Source citation accuracy
- User satisfaction (from feedback)
- Latency (inference time)
- Retrieval precision and recall

---

#### NRAG-072: Analytics & Monitoring Dashboard

**Priority:** P2 - Medium  
**Token Estimate:** 12,000 tokens  
**Status:** Backlog

**Description:**  
Create analytics dashboard for monitoring AgentLightning training and performance.

**Acceptance Criteria:**

- [ ] Trajectory statistics (count, success rate, duration)
- [ ] Reward signal distribution over time
- [ ] Agent performance metrics over time
- [ ] Training progress visualization
- [ ] Integration with existing Streamlit UI
- [ ] Export capabilities for detailed analysis

---

#### NRAG-073: Documentation & Best Practices Guide

**Priority:** P1 - High  
**Token Estimate:** 8,000 tokens  
**Status:** Backlog

**Description:**  
Create comprehensive documentation for AgentLightning integration.

**Acceptance Criteria:**

- [ ] Setup and configuration guide
- [ ] Training workflow documentation
- [ ] Best practices for reward signal design
- [ ] Troubleshooting guide
- [ ] Example training scenarios
- [ ] Performance optimization tips

**Documentation Sections:**
1. Overview and motivation
2. Installation and configuration
3. Data collection and trajectory tracking
4. Reward engineering guidelines
5. Training pipeline setup
6. Model evaluation and deployment
7. Common issues and solutions

---

### AgentLightning Epic Summary

**Total Sessions:** 3 (Sessions 9-11)
**Total Token Budget:** ~240,000 tokens
**Timeline:** Progressive implementation with validation at each stage

**Task Breakdown:**

| Task ID | Task | Tokens | Priority | Session |
|---------|------|--------|----------|----------|
| NRAG-063 | Installation & Configuration | 8,000 | P1 | 9 |
| NRAG-064 | Store & Emitter Setup | 12,000 | P1 | 9 |
| NRAG-065 | Agent Integration | 10,000 | P0 | 9 |
| NRAG-066 | Trajectory Tracking | 10,000 | P1 | 9 |
| NRAG-067 | User Feedback UI | 12,000 | P0 | 10 |
| NRAG-068 | Automatic Rewards | 14,000 | P1 | 10 |
| NRAG-069 | Reward Pipeline | 10,000 | P0 | 10 |
| NRAG-070 | APO Prompt Optimization | 16,000 | P1 | 11 |
| NRAG-071 | Evaluation Framework | 12,000 | P1 | 11 |
| NRAG-072 | Analytics Dashboard | 12,000 | P2 | 11 |
| NRAG-073 | Documentation | 8,000 | P1 | 11 |
| **Total** | | **124,000** | | |

**Key Milestones:**
1. **Foundation (Session 9):** Basic infrastructure and emitter integration (~40,000 tokens)
2. **Feedback System (Session 10):** User feedback and reward generation (~36,000 tokens)
3. **APO & Evaluation (Session 11):** Prompt optimization and analytics (~48,000 tokens)

**Architecture Impact:**
- Minimal changes to existing agent code
- Optional feature (can be disabled via config)
- Leverages existing LangChain/LangGraph infrastructure
- Selective optimization: Reasoner & Synthesiser prioritized for prompt tuning
- **API-compatible:** APO works with hosted LLM APIs (no model weight access required)

**Expected Outcomes:**
- Continuous prompt improvement based on real deployment data
- Data-driven optimization based on real user queries
- Reduced reliance on manual prompt engineering
- Better handling of edge cases through learned prompt patterns
- Measurable quality improvements tracked via evaluation framework

---

### Epic: Advanced Features (Expandable)

| ID | Item | Token Estimate | Priority |
|----|------|----------------|----------|
| NRAG-0?? | AgentLightning Integration | 25,000 | P2 |
| NRAG-0?? | Conversation Memory | 18,000 | P2 |
| NRAG-0?? | Semantic Caching | 15,000 | P3 |
| NRAG-0?? | Multi-Database Support | 18,000 | P2 |
| NRAG-0?? | Automated KB Sync | 15,000 | P2 |
| NRAG-0?? | Feedback Collection | 12,000 | P3 |

### Epic: Deployment

| ID | Item | Token Estimate | Priority |
|----|------|----------------|----------|
| NRAG-0?? | Docker Containerization | 12,000 | P2 |
| NRAG-0?? | Cloud Deployment | 15,000 | P2 |
| NRAG-0?? | REST API Exposure | 15,000 | P2 |

---

## Technical Debt & Improvements

| ID | Issue | Token Estimate | Priority |
|----|-------|----------------|----------|
| TD-001 | Comprehensive error handling | 10,000 | P1 |
| TD-002 | Structured logging | 6,000 | P1 |
| TD-003 | Input validation | 8,000 | P1 |
| TD-004 | Unit test coverage | 20,000 | P2 |
| TD-005 | Integration tests | 15,000 | P2 |
| TD-006 | Chunk size optimization | 10,000 | P2 |
| TD-007 | Type hints completion | 8,000 | P3 |
| TD-008 | API documentation | 15,000 | P3 |

---

## Appendix: Token Budget Summary

### Core Implementation (6 Sessions)

| Session | Focus | Token Budget |
|---------|-------|--------------|
| 1 | Project Foundation | 80,000 |
| 2 | Notion & Arxiv Loaders | 80,000 |
| 3 | Vector Store & Embeddings | 80,000 |
| 4 | Planner & Researcher | 80,000 |
| 5 | Reasoner & Synthesiser | 80,000 |
| 6 | Orchestration & Testing | 80,000 |
| **Total** | | **480,000** |

### Future Features (Estimated)

| Epic | Estimated Tokens |
|------|------------------|
| A2A Protocol | 113,000 |
| User Interface | 51,000 |
| Advanced Features | 103,000 |
| Deployment | 42,000 |
| Technical Debt | 92,000 |
| **Total** | **401,000** |

---

*Last Updated: February 3, 2026*  
*Backlog Owner: Arthur*  
*Execution Model: AI Agent (Session-Based)*
