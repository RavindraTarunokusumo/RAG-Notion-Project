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

- [x] Project directory structure created
- [x] All `__init__.py` files in place
- [x] `.gitignore` configured for Python projects
- [x] Basic `README.md` with project overview

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

- [x] `pyproject.toml` created with standard PEP 621 configuration (uv)
- [x] All core dependencies specified with version ranges
- [x] Development dependencies separated
- [x] Python version constraint: `>=3.11`

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

- [x] `config/settings.py` with Pydantic BaseSettings
- [x] Environment variable loading from `.env`
- [x] Validation for required API keys
- [x] Model configuration presets

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

- [x] All required variables documented
- [x] Placeholder values clearly marked
- [x] Comments explaining each variable

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

- [x] Tracing initialization function
- [x] Custom decorators for agent functions
- [x] Metadata tagging utilities
- [x] Environment variable validation

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

- [x] Document deduplication function
- [x] Arxiv ID extraction from URLs
- [x] JSON parsing utilities with error handling
- [x] Document formatting helpers

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

- [x] `main.py` with basic CLI
- [x] Initialization of all components
- [x] Simple query execution
- [x] Error handling and logging

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

