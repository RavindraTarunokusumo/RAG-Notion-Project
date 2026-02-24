from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.agents.reasoner import reasoner_node
from src.agents.researcher import researcher_node
from src.agents.synthesiser import synthesiser_node


class _FakeQueryChain:
    def __or__(self, _other):
        return self

    def invoke(self, _input):
        return {"optimized_queries": ["query-a"]}


def test_researcher_aborts_on_fatal_quota_error(monkeypatch):
    class FatalRetriever:
        def __init__(self):
            self.calls = 0

        def invoke(self, _query):
            self.calls += 1
            raise RuntimeError(
                "status_code: 429 body: Trial key quota exceeded "
                "(limited to 1000 api calls)"
            )

    retriever = FatalRetriever()
    monkeypatch.setattr("src.agents.researcher.time.sleep", lambda *_: None)
    monkeypatch.setattr("src.agents.researcher.get_agent_llm", lambda *_: object())
    monkeypatch.setattr(
        "src.agents.researcher.get_query_optimizer_prompt",
        lambda _parser: _FakeQueryChain(),
    )
    monkeypatch.setattr("src.agents.researcher.get_retriever", lambda **_: retriever)

    state = {
        "sub_tasks": [
            {"id": 1, "task": "t1", "keywords": []},
            {"id": 2, "task": "t2", "keywords": []},
        ]
    }
    result = researcher_node(state)

    assert retriever.calls == 1
    assert "error" in result
    assert result["retrieval_metadata"]["fatal"] is True
    assert result["retrieval_metadata"]["aborted"] is True
    assert result["retrieval_metadata"]["error_type"] == "quota_exhausted"
    assert result["retrieval_metadata"]["failed_tasks"] == 1


def test_researcher_continues_on_transient_error(monkeypatch):
    class FlakyRetriever:
        def __init__(self):
            self.calls = 0

        def invoke(self, _query):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary network error")
            return [Document(page_content="doc-1", metadata={"source": "test"})]

    retriever = FlakyRetriever()
    monkeypatch.setattr("src.agents.researcher.time.sleep", lambda *_: None)
    monkeypatch.setattr("src.agents.researcher.get_agent_llm", lambda *_: object())
    monkeypatch.setattr(
        "src.agents.researcher.get_query_optimizer_prompt",
        lambda _parser: _FakeQueryChain(),
    )
    monkeypatch.setattr("src.agents.researcher.get_retriever", lambda **_: retriever)

    state = {
        "sub_tasks": [
            {"id": 1, "task": "t1", "keywords": []},
            {"id": 2, "task": "t2", "keywords": []},
        ]
    }
    result = researcher_node(state)

    assert "error" not in result
    assert retriever.calls == 2
    assert result["retrieval_metadata"]["fatal"] is False
    assert result["retrieval_metadata"]["failed_tasks"] == 1
    assert result["retrieval_metadata"]["total_docs"] == 1


def test_reasoner_parses_json_with_thinking_wrapper(monkeypatch):
    raw = """
<THINKING>
hidden reasoning
</THINKING>
{
  "analysis": [
    {
      "sub_task_id": 1,
      "key_findings": ["k1"],
      "supporting_evidence": ["e1"],
      "contradictions": [],
      "confidence": 0.8,
      "gaps": []
    }
  ],
  "overall_assessment": "ok"
}
"""
    monkeypatch.setattr(
        "src.agents.reasoner.get_agent_llm",
        lambda *_: RunnableLambda(lambda _input: AIMessage(content=raw)),
    )

    state = {
        "error": None,
        "sub_tasks": [{"id": 1, "task": "x"}],
        "retrieved_docs": [Document(page_content="doc", metadata={"source": "t"})],
    }
    result = reasoner_node(state)

    assert "error" not in result
    assert result["overall_assessment"] == "ok"
    assert result["analysis"][0]["sub_task_id"] == 1


def test_reasoner_returns_deterministic_gap_analysis_when_no_docs(monkeypatch):
    def _should_not_call(_agent_type):
        raise AssertionError("LLM should not be called for no-doc fallback")

    monkeypatch.setattr("src.agents.reasoner.get_agent_llm", _should_not_call)

    state = {
        "error": None,
        "sub_tasks": [{"id": 7, "task": "Find architecture"}],
        "retrieved_docs": [],
    }
    result = reasoner_node(state)

    assert "error" not in result
    assert result["current_agent"] == "reasoner"
    assert len(result["analysis"]) == 1
    assert result["analysis"][0]["confidence"] == 0.0
    assert "No documents retrieved" in result["analysis"][0]["gaps"][0]


def test_synthesiser_skips_llm_when_upstream_error_has_no_docs(monkeypatch):
    def _should_not_call(_agent_type):
        raise AssertionError("LLM should not be called when upstream error exists")

    monkeypatch.setattr("src.agents.synthesiser.get_agent_llm", _should_not_call)

    state = {
        "query": "Tell me about my knowledge base",
        "error": "Researcher Error: Retrieval aborted",
        "retrieved_docs": [],
        "analysis": [],
    }
    result = synthesiser_node(state)

    assert result["current_agent"] == "synthesiser"
    assert result["sources"] == []
    assert "could not retrieve supporting evidence" in result["final_answer"].lower()
