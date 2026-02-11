"""Tests for the A2A Tool Agent Framework (NRAG-027 through NRAG-033)."""

import asyncio

import pytest

from src.tools.base import AgentCard, ToolAgent, ToolResult
from src.tools.client import A2AToolClient
from src.tools.registry import ToolRegistry

# --- Fixtures ---


class MockToolAgent(ToolAgent):
    """A simple mock tool agent for testing."""

    def __init__(self, name="mock_agent", capabilities=None, confidence=0.5):
        self._name = name
        self._capabilities = capabilities or ["mock_capability"]
        self._confidence = confidence

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name=self._name,
            description=f"Mock agent: {self._name}",
            version="1.0.0",
            capabilities=self._capabilities,
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"output": {"type": "string"}}},
            endpoint=f"tool://{self._name}",
        )

    async def execute(self, task):
        return ToolResult(
            success=True,
            data={"result": f"executed by {self._name}"},
            agent_name=self._name,
        )

    def can_handle(self, task_description):
        return self._confidence


@pytest.fixture
def registry():
    """Fresh registry for each test."""
    reg = ToolRegistry()
    return reg


@pytest.fixture
def mock_agent():
    return MockToolAgent()


@pytest.fixture
def client(registry):
    return A2AToolClient(registry)


# --- AgentCard Tests ---


class TestAgentCard:
    def test_create_agent_card(self):
        card = AgentCard(
            name="test",
            description="Test agent",
            version="1.0.0",
            capabilities=["cap_a", "cap_b"],
            input_schema={},
            output_schema={},
            endpoint="tool://test",
        )
        assert card.name == "test"
        assert card.version == "1.0.0"
        assert len(card.capabilities) == 2

    def test_matches_capability(self):
        card = AgentCard(
            name="test",
            description="Test",
            version="1.0.0",
            capabilities=["web_search", "news_search"],
            input_schema={},
            output_schema={},
            endpoint="tool://test",
        )
        assert card.matches_capability("web_search") is True
        assert card.matches_capability("code_execution") is False


# --- ToolResult Tests ---


class TestToolResult:
    def test_success_result(self):
        result = ToolResult(success=True, data={"key": "value"}, agent_name="test")
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_error_result(self):
        result = ToolResult(success=False, data=None, error="Something failed")
        assert result.success is False
        assert result.error == "Something failed"

    def test_serialization(self):
        result = ToolResult(
            success=True,
            data=[1, 2, 3],
            metadata={"count": 3},
            agent_name="test",
        )
        d = result.model_dump()
        assert d["success"] is True
        assert d["data"] == [1, 2, 3]
        assert d["metadata"] == {"count": 3}


# --- ToolAgent Tests ---


class TestToolAgent:
    def test_default_can_handle(self):
        agent = MockToolAgent(confidence=0.0)
        # Override to test base default
        assert ToolAgent.can_handle(agent, "anything") == 0.0

    def test_default_health_check(self):
        agent = MockToolAgent()
        assert agent.health_check() is True

    def test_name_property(self):
        agent = MockToolAgent(name="my_agent")
        assert agent.name == "my_agent"


# --- Registry Tests ---


class TestToolRegistry:
    def test_register_and_discover(self, registry, mock_agent):
        registry.register(mock_agent)
        cards = registry.discover()
        assert len(cards) == 1
        assert cards[0].name == "mock_agent"

    def test_discover_by_capability(self, registry):
        agent_a = MockToolAgent("agent_a", ["web_search"])
        agent_b = MockToolAgent("agent_b", ["code_execution"])
        registry.register(agent_a)
        registry.register(agent_b)

        web_cards = registry.discover("web_search")
        assert len(web_cards) == 1
        assert web_cards[0].name == "agent_a"

        code_cards = registry.discover("code_execution")
        assert len(code_cards) == 1
        assert code_cards[0].name == "agent_b"

    def test_discover_all(self, registry):
        registry.register(MockToolAgent("a", ["cap_1"]))
        registry.register(MockToolAgent("b", ["cap_2"]))
        all_cards = registry.discover()
        assert len(all_cards) == 2

    def test_unregister(self, registry, mock_agent):
        registry.register(mock_agent)
        assert len(registry.discover()) == 1
        registry.unregister("mock_agent")
        assert len(registry.discover()) == 0

    def test_get_agent(self, registry, mock_agent):
        registry.register(mock_agent)
        agent = registry.get_agent("mock_agent")
        assert agent is mock_agent

    def test_get_agent_not_found(self, registry):
        assert registry.get_agent("nonexistent") is None

    def test_health_check_all(self, registry):
        registry.register(MockToolAgent("a"))
        registry.register(MockToolAgent("b"))
        results = registry.health_check_all()
        assert results == {"a": True, "b": True}

    def test_singleton_pattern(self):
        ToolRegistry.reset_instance()
        r1 = ToolRegistry.get_instance()
        r2 = ToolRegistry.get_instance()
        assert r1 is r2
        ToolRegistry.reset_instance()


# --- Client Tests ---


class TestA2AToolClient:
    @pytest.mark.asyncio
    async def test_discover_agents(self, client, registry):
        registry.register(MockToolAgent("a", ["web_search"]))
        cards = await client.discover_agents("web_search")
        assert len(cards) == 1
        assert cards[0].name == "a"

    @pytest.mark.asyncio
    async def test_invoke_tool_success(self, client, registry):
        registry.register(MockToolAgent("test_agent"))
        result = await client.invoke_tool("test_agent", {"input": "hello"})
        assert result.success is True
        assert result.agent_name == "test_agent"

    @pytest.mark.asyncio
    async def test_invoke_tool_not_found(self, client):
        result = await client.invoke_tool("nonexistent", {})
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_invoke_tool_timeout(self, client, registry):
        class SlowAgent(MockToolAgent):
            async def execute(self, task):
                await asyncio.sleep(10)
                return ToolResult(success=True, data=None)

        registry.register(SlowAgent("slow"))
        result = await client.invoke_tool("slow", {}, timeout=0.1)
        assert result.success is False
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_invoke_tool_exception(self, client, registry):
        class FailAgent(MockToolAgent):
            async def execute(self, task):
                raise RuntimeError("Boom")

        registry.register(FailAgent("fail"))
        result = await client.invoke_tool("fail", {})
        assert result.success is False
        assert "Boom" in result.error

    def test_select_best_agent(self, client, registry):
        low = MockToolAgent("low", ["cap"], confidence=0.2)
        high = MockToolAgent("high", ["cap"], confidence=0.9)
        registry.register(low)
        registry.register(high)

        cards = registry.discover()
        best = client.select_best_agent("some task", cards)
        assert best is not None
        assert best.name == "high"

    def test_select_best_agent_none(self, client, registry):
        zero = MockToolAgent("zero", ["cap"], confidence=0.0)
        registry.register(zero)
        cards = registry.discover()
        best = client.select_best_agent("some task", cards)
        assert best is None


# --- Individual Agent Card Tests ---


class TestBuiltInAgentCards:
    def test_web_searcher_card(self):
        from src.tools.web_searcher import WebSearcherAgent

        agent = WebSearcherAgent()
        card = agent.get_agent_card()
        assert card.name == "web_searcher"
        assert "web_search" in card.capabilities
        assert card.version == "1.0.0"

    def test_code_executor_card(self):
        from src.tools.code_executor import CodeExecutorAgent

        agent = CodeExecutorAgent()
        card = agent.get_agent_card()
        assert card.name == "code_executor"
        assert "code_execution" in card.capabilities

    def test_citation_validator_card(self):
        from src.tools.citation_validator import CitationValidatorAgent

        agent = CitationValidatorAgent()
        card = agent.get_agent_card()
        assert card.name == "citation_validator"
        assert "citation_validation" in card.capabilities

    def test_math_solver_card(self):
        from src.tools.math_solver import MathSolverAgent

        agent = MathSolverAgent()
        card = agent.get_agent_card()
        assert card.name == "math_solver"
        assert "math_solving" in card.capabilities

    def test_diagram_generator_card(self):
        from src.tools.diagram_generator import DiagramGeneratorAgent

        agent = DiagramGeneratorAgent()
        card = agent.get_agent_card()
        assert card.name == "diagram_generator"
        assert "diagram_generation" in card.capabilities


# --- can_handle Confidence Tests ---


class TestCanHandle:
    def test_web_searcher_keywords(self):
        from src.tools.web_searcher import WebSearcherAgent

        agent = WebSearcherAgent()
        assert agent.can_handle("latest news about AI") > 0.0
        assert agent.can_handle("irrelevant query") == 0.0

    def test_code_executor_keywords(self):
        from src.tools.code_executor import CodeExecutorAgent

        agent = CodeExecutorAgent()
        assert agent.can_handle("calculate the sum of numbers") > 0.0
        assert agent.can_handle("irrelevant query") == 0.0

    def test_citation_validator_keywords(self):
        from src.tools.citation_validator import CitationValidatorAgent

        agent = CitationValidatorAgent()
        assert agent.can_handle("verify arxiv citation 2301.07041") > 0.0
        assert agent.can_handle("irrelevant query") == 0.0

    def test_math_solver_keywords(self):
        from src.tools.math_solver import MathSolverAgent

        agent = MathSolverAgent()
        assert agent.can_handle("solve the equation x^2 + 1 = 0") > 0.0
        assert agent.can_handle("irrelevant query") == 0.0

    def test_diagram_generator_keywords(self):
        from src.tools.diagram_generator import DiagramGeneratorAgent

        agent = DiagramGeneratorAgent()
        assert agent.can_handle("create a flowchart diagram") > 0.0
        assert agent.can_handle("irrelevant query") == 0.0


# --- Integration Test ---


class TestIntegration:
    def test_register_all_discover_by_capability(self):
        """Register all built-in agents, discover by capability."""
        from src.tools.citation_validator import CitationValidatorAgent
        from src.tools.code_executor import CodeExecutorAgent
        from src.tools.diagram_generator import DiagramGeneratorAgent
        from src.tools.math_solver import MathSolverAgent
        from src.tools.web_searcher import WebSearcherAgent

        registry = ToolRegistry()
        registry.register(WebSearcherAgent())
        registry.register(CodeExecutorAgent())
        registry.register(CitationValidatorAgent())
        registry.register(MathSolverAgent())
        registry.register(DiagramGeneratorAgent())

        all_cards = registry.discover()
        assert len(all_cards) == 5

        web = registry.discover("web_search")
        assert len(web) == 1
        assert web[0].name == "web_searcher"

        math = registry.discover("math_solving")
        assert len(math) == 1
        assert math[0].name == "math_solver"

        mermaid = registry.discover("mermaid")
        assert len(mermaid) == 1
        assert mermaid[0].name == "diagram_generator"

        health = registry.health_check_all()
        assert all(health.values())
