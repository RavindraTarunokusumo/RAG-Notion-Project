import pytest

from config.settings import settings
from src.agents.llm_factory import get_agent_llm


def test_qwen_provider_selected_for_planner(monkeypatch):
    captured: dict = {}

    class DummyTongyi:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "src.agents.providers.qwen.ChatTongyi",
        DummyTongyi,
    )
    monkeypatch.setattr(settings.models.planner, "provider", "qwen")
    monkeypatch.setattr(
        settings.models.planner, "model", "qwen-flash-2025-07-28"
    )

    llm = get_agent_llm("planner")

    assert isinstance(llm, DummyTongyi)
    assert captured["model"] == "qwen-flash-2025-07-28"


def test_openai_provider_selected_for_planner(monkeypatch):
    captured: dict = {}

    class DummyOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "src.agents.providers.openai.ChatOpenAI",
        DummyOpenAI,
    )
    monkeypatch.setattr(settings.models.planner, "provider", "openai")
    monkeypatch.setattr(settings.models.planner, "model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "openai_api_key", "test-openai-key")

    llm = get_agent_llm("planner")

    assert isinstance(llm, DummyOpenAI)
    assert captured["model"] == "gpt-4o-mini"
    assert captured["api_key"] == "test-openai-key"


def test_missing_openai_api_key_raises(monkeypatch):
    monkeypatch.setattr(settings.models.planner, "provider", "openai")
    monkeypatch.setattr(settings.models.planner, "model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "openai_api_key", None)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_agent_llm("planner")


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(settings.models.planner, "provider", "unknown")

    with pytest.raises(ValueError, match="Unsupported chat provider"):
        get_agent_llm("planner")
