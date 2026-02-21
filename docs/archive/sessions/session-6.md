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

- [x] StateGraph with all agent nodes
- [x] Linear flow with proper edges
- [x] Error handling nodes
- [x] Compiled graph ready for execution

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
    Query â†’ Planner â†’ Researcher â†’ Reasoner â†’ Synthesiser â†’ Response
    
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
            query = input("\nðŸ“ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            print("\nâ³ Processing...\n")
            start = datetime.now()
            
            result = rag.query(query)
            
            duration = (datetime.now() - start).total_seconds()
            
            print("="*60)
            print("ðŸ“– RESPONSE")
            print("="*60)
            print(result["answer"])
            
            if result["sources"]:
                print("\nðŸ“š SOURCES")
                print("-"*40)
                for src in result["sources"][:5]:
                    print(f"  [{src['id']}] {src['title']}")
                    if src.get('url'):
                        print(f"      {src['url']}")
            
            print(f"\nâ±ï¸ Response time: {duration:.2f}s")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nâŒ Error: {e}")

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
            print(f"\nâš ï¸ Error: {result['error']}")
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

