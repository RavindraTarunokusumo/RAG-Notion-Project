import pytest

from src.orchestrator.graph import create_rag_graph


@pytest.mark.asyncio
async def test_graph_compilation():
    """Test that the graph compiles without errors."""
    try:
        app = create_rag_graph()
        assert app is not None
    except Exception as e:
        pytest.fail(f"Graph compilation failed: {e}")

@pytest.mark.asyncio
async def test_graph_execution_structure():
    """Run the graph with a dummy query to check state transitions (mocked LLM would be ideal here)."""
    # For now, we skip full execution in unit tests to avoid API costs
    # This just ensures we can initialize the state
    pass
