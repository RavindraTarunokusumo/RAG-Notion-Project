import json

import pytest

from config.settings import settings
from src.orchestrator.graph import _instrument_node
from src.orchestrator.state import build_initial_state
from src.tools.base import AgentCard, ToolAgent, ToolResult
from src.tools.client import A2AToolClient
from src.tools.registry import ToolRegistry
from src.utils.debugging import debug_run, log_trace_event


def _read_trace_records(log_dir):
    traces = list(log_dir.glob("trace-*.jsonl"))
    assert len(traces) == 1
    return [json.loads(line) for line in traces[0].read_text().splitlines()]


def _set_debug_config(tmp_path, enabled=True):
    settings.debug.enabled = enabled
    settings.debug.log_dir = str(tmp_path / "logs")
    settings.debug.log_level = "INFO"


def test_debug_run_writes_start_and_end_events(tmp_path):
    _set_debug_config(tmp_path, enabled=True)
    initial_state = build_initial_state("test query")

    with debug_run(
        query="test query",
        initial_state=initial_state,
        mode="invoke",
    ) as session:
        assert session is not None
        log_trace_event("custom_event", {"ok": True})
        session.record_run_end(initial_state)

    records = _read_trace_records(tmp_path / "logs")
    event_types = [item["event_type"] for item in records]
    assert "run_start" in event_types
    assert "custom_event" in event_types
    assert "run_end" in event_types


def test_debug_run_disabled_writes_nothing(tmp_path):
    _set_debug_config(tmp_path, enabled=False)
    initial_state = build_initial_state("test query")

    with debug_run(
        query="test query",
        initial_state=initial_state,
        mode="invoke",
    ) as session:
        assert session is None

    assert not list((tmp_path / "logs").glob("trace-*.jsonl"))


def test_instrument_node_records_state_delta(tmp_path):
    _set_debug_config(tmp_path, enabled=True)
    initial_state = build_initial_state("delta query")

    def fake_node(_state):
        return {
            "planning_reasoning": "done",
            "current_agent": "planner",
        }

    wrapped = _instrument_node("planner", fake_node)
    with debug_run(
        query="delta query",
        initial_state=initial_state,
        mode="invoke",
    ) as session:
        updated = wrapped(initial_state)
        final_state = dict(initial_state)
        final_state.update(updated)
        session.record_run_end(final_state)

    records = _read_trace_records(tmp_path / "logs")
    node_end = [item for item in records if item["event_type"] == "node_end"]
    assert len(node_end) == 1
    payload = node_end[0]["payload"]
    assert payload["node"] == "planner"
    assert "planning_reasoning" in payload["state_delta"]
    assert payload["state_delta"]["planning_reasoning"]["after"] == "done"


class _TraceToolAgent(ToolAgent):
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="trace_tool",
            description="Trace tool",
            version="1.0.0",
            capabilities=["debug"],
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            endpoint="tool://trace_tool",
        )

    async def execute(self, task):
        return ToolResult(
            success=True,
            data={"echo": task},
            metadata={"size": len(task)},
            agent_name="trace_tool",
        )

    def can_handle(self, _task_description):
        return 0.9


@pytest.mark.asyncio
async def test_tool_client_events_are_traced(tmp_path):
    _set_debug_config(tmp_path, enabled=True)
    registry = ToolRegistry()
    registry.register(_TraceToolAgent())
    client = A2AToolClient(registry)
    initial_state = build_initial_state("tool trace")

    with debug_run(
        query="tool trace",
        initial_state=initial_state,
        mode="invoke",
    ) as session:
        cards = await client.discover_agents("debug")
        selected = client.select_best_agent("needs debug", cards)
        assert selected is not None
        result = await client.invoke_tool(
            selected.name,
            {"task": "ping"},
            timeout=5,
        )
        assert result.success is True
        session.record_run_end(initial_state)

    records = _read_trace_records(tmp_path / "logs")
    event_types = [item["event_type"] for item in records]
    assert "tool_discovery" in event_types
    assert "tool_selection" in event_types
    assert "tool_invoke_start" in event_types
    assert "tool_invoke_end" in event_types
