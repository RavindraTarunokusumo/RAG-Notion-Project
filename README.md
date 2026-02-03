# Notion Agentic RAG

A multi-agent RAG system that answers research questions using your Notion knowledge base and arXiv papers w/ [LangSmith](https://docs.langchain.com/langsmith/home) tracing for observability.

## What It Does

Asks an LLM team to research your question:

- **Planner**: Breaks your question into sub-tasks
- **Researcher**: Finds relevant documents from your knowledge base
- **Reasoner**: Analyzes what was found (uses Cohere's reasoning model)
- **Synthesiser**: Writes a cited answer

## Setup

**Requirements**: Python 3.11+

### Clone and install dependencies:

```bash
git clone <repo-url>
cd RAG-Notion-Project
pip install -r requirements.txt
# Or, using uv
uv sync
```

### Create `.env` from template:

```bash
cp .env.example .env
```

### Add your API keys:

```env
COHERE_API_KEY=your_key
NOTION_TOKEN=your_integration_token
NOTION_DATABASE_ID=your_database_id

# Optional, for tracing
LANGCHAIN_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your_project_name
LANGCHAIN_ENDPOINT=your_region 
# EU: https://eu.api.smith.langchain.com US: https://us.api.smith.langchain.com
```

### Ingest your knowledge base:

```bash
uv run python main.py --ingest
```

## Notion KB Structure

`notion_loader.py` assumes the knowledge base has the following columns:

| Title | Topic | Keywords | URL | Type | Publication Date | Notes |

Otherwise, modify `notion_loader.py:73` and `arxiv_loader.py:96` to match your own.

## Usage

```bash
# Query
uv run python main.py "Tell me about my knowledge base."

# Test connection
uv run python main.py --test-conn

# Rebuild vector store
uv run python main.py --ingest --ingest

# Debugging
uv run python main.py "Tell me about my knowledge base." --verbose
```

The system will plan, retrieve, analyze, and generate an answer with sources.

## Tech Stack

- **LLMs**: Cohere (Command-R, Command-A-Reasoning)
- **Embeddings**: Cohere embed-english-v3.0
- **Vector Store**: ChromaDB
- **Orchestration**: LangGraph
- **Tracing**: LangSmith (optional)

## Project Structure

```
src/
├── agents/          # Planner, Researcher, Reasoner, Synthesiser
├── loaders/         # Notion + arXiv document loading
├── orchestrator/    # LangGraph workflow
└── rag/             # Embeddings, vector store, retrieval
```

## Notes

- Uses a monkeypatch in `llm_factory.py` to handle Cohere's reasoning model responses (see `AGENT.md` for details)
- Rate limits are handled with delays for trial API keys (adjust as you wish if you're in paid tier)
