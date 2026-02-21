from src.agents.reasoner import reasoner_node


def test_reasoner_returns_gap_analysis_without_documents(monkeypatch):
    def _fail_if_called(_agent_type):
        raise AssertionError("get_agent_llm should not be called when no docs are retrieved")

    monkeypatch.setattr("src.agents.reasoner.get_agent_llm", _fail_if_called)

    state = {
        "sub_tasks": [
            {"id": 1, "task": "Describe the KB structure", "priority": "high", "keywords": []},
            {"id": 2, "task": "List core topics", "priority": "medium", "keywords": []},
        ],
        "retrieved_docs": [],
    }

    result = reasoner_node(state)

    assert result["current_agent"] == "reasoner"
    assert "No supporting documents were retrieved" in result["overall_assessment"]
    assert len(result["analysis"]) == 2
    assert result["analysis"][0]["sub_task_id"] == 1
    assert result["analysis"][1]["sub_task_id"] == 2
    assert result["analysis"][0]["confidence"] == 0.0
    assert result["analysis"][1]["supporting_evidence"] == []


def test_reasoner_gap_analysis_falls_back_to_position_for_missing_task_id(monkeypatch):
    def _fail_if_called(_agent_type):
        raise AssertionError("get_agent_llm should not be called when no docs are retrieved")

    monkeypatch.setattr("src.agents.reasoner.get_agent_llm", _fail_if_called)

    state = {
        "sub_tasks": [
            {"task": "Task with missing id", "priority": "high", "keywords": []},
        ],
        "retrieved_docs": [],
    }

    result = reasoner_node(state)

    assert result["analysis"][0]["sub_task_id"] == 1
    assert "Task with missing id" in result["analysis"][0]["gaps"][0]
