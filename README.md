# Notion Agentic RAG

A multi-agent RAG system that answers research questions using your Notion knowledge base and arXiv papers. Features dynamic tool agents, real-time streaming, and [LangSmith](https://docs.langchain.com/langsmith/home) tracing.

## What It Does

**Core Pipeline**: Four specialized agents collaborate to answer your questions:

- **Planner**: Decomposes questions into sub-tasks
- **Researcher**: Retrieves relevant documents with reranking
- **Reasoner**: Analyzes evidence using reasoning models
- **Synthesiser**: Generates cited answers

**Tool Agents** (optional, runtime-invoked via A2A protocol):

- **Web Searcher**: DuckDuckGo for current information
- **Code Executor**: Sandboxed Python execution
- **Citation Validator**: Verifies arXiv papers
- **Math Solver**: SymPy symbolic computation
- **Diagram Generator**: Mermaid diagrams via LLM

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

### Web Interface (Recommended)

Launch the Streamlit chat interface:

```bash
streamlit run app.py
```

**Features**:
- 💬 Interactive chat with history
- 📚 Rich citations with source cards
- 🔄 Live agent progress tracking
- ⚡ Real-time streaming responses
- 💾 Session persistence (JSON/Markdown export)
- ⚙️ Dynamic settings (models, retrieval params)
- 🛠️ Tool agent status monitoring

### Command Line Interface

```bash
# Query
uv run python main.py "Tell me about my knowledge base."

# Test connection
uv run python main.py --test-conn

# Rebuild vector store
uv run python main.py --ingest --rebuild

# Debugging
uv run python main.py "Tell me about my knowledge base." --verbose
```

The system will plan, retrieve, analyze, and generate an answer with sources.

## Tech Stack

- **LLMs**: Cohere (Command-R, Command-A-Reasoning)
- **Embeddings**: Cohere embed-english-v3.0
- **Vector Store**: ChromaDB
- **Orchestration**: LangGraph
- **UI**: Streamlit (Web Interface)
- **Tracing**: LangSmith (optional)

## Project Structure

```
src/
├── agents/          # Core 4-agent pipeline
├── tools/           # A2A tool agents (web search, code exec, math, etc.)
├── loaders/         # Notion + arXiv ingestion
├── orchestrator/    # LangGraph workflow + state management
├── rag/             # Embeddings, vector store, retrieval
└── utils/           # Session manager, tracing, helpers
```

## Architecture

**Multi-Agent Pipeline** (LangGraph orchestration):
```
User Query → Planner → Researcher → Reasoner → Synthesiser → Answer
                ↓           ↓           ↓           ↓
            Tool Agents (invoked on-demand via A2A)
```

**Tool Agent Discovery**: Agents can dynamically discover and invoke tool agents via Agent Cards (A2A protocol). Tool results are tracked in `AgentState.tool_results`.

## Configuration

Tool agents can be enabled/disabled in `config/settings.py`:

```python
tool_agents = ToolAgentConfig(
    enabled=True,
    web_searcher_enabled=True,
    code_executor_enabled=True,
    citation_validator_enabled=True,
    math_solver_enabled=True,
    diagram_generator_enabled=True,
)
```

## Notes

- **Cohere Reasoning Patch**: `llm_factory.py` merges Thinking+Text blocks from V2 API (see `AGENTS.md`)
- **Rate Limiting**: Embedding delays configured for free-tier Cohere API
- **Sandboxing**: Code executor uses subprocess isolation with import whitelisting
- **Session Storage**: All sessions saved to `./data/sessions/` with auto-generated names
