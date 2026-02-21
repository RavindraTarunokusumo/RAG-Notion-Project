from config.settings import settings
from src.rag.embeddings import get_embeddings
from src.rag.retriever import get_retriever


def test_get_embeddings_uses_dashscope(monkeypatch):
    captured: dict = {}

    class DummyEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "src.rag.embeddings.DashScopeEmbeddings",
        DummyEmbeddings,
    )
    monkeypatch.setattr(settings.models, "embedding_provider", "qwen")
    monkeypatch.setattr(settings.models, "embedding_model", "text-embedding-v4")
    monkeypatch.setattr(settings, "dashscope_api_key", "test-dashscope-key")

    emb = get_embeddings()

    assert isinstance(emb, DummyEmbeddings)
    assert captured["model"] == "text-embedding-v4"
    assert captured["dashscope_api_key"] == "test-dashscope-key"


def test_get_retriever_uses_dashscope_rerank(monkeypatch):
    captured: dict = {}
    base_retriever = object()

    class DummyVectorStore:
        def as_retriever(self, **_kwargs):
            return base_retriever

    class DummyRerank:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class DummyCompressionRetriever:
        def __init__(self, base_compressor, base_retriever):
            self.base_compressor = base_compressor
            self.base_retriever = base_retriever

    monkeypatch.setattr("src.rag.retriever.get_vector_store", DummyVectorStore)
    monkeypatch.setattr("src.rag.retriever.DashScopeRerank", DummyRerank)
    monkeypatch.setattr(
        "src.rag.retriever.ContextualCompressionRetriever",
        DummyCompressionRetriever,
    )
    monkeypatch.setattr(settings.models, "rerank_provider", "qwen")
    monkeypatch.setattr(settings.models, "rerank_model", "qwen3-rerank")
    monkeypatch.setattr(settings, "dashscope_api_key", "test-dashscope-key")
    monkeypatch.setattr(settings, "rerank_top_n", 5)

    retriever = get_retriever(use_rerank=True)

    assert isinstance(retriever, DummyCompressionRetriever)
    assert captured["model"] == "qwen3-rerank"
    assert captured["api_key"] == "test-dashscope-key"
    assert captured["top_n"] == 5
    assert retriever.base_retriever is base_retriever


def test_rerank_init_failure_falls_back_to_base_retriever(monkeypatch):
    base_retriever = object()

    class DummyVectorStore:
        def as_retriever(self, **_kwargs):
            return base_retriever

    class ExplodingRerank:
        def __init__(self, **_kwargs):
            raise RuntimeError("rerank init failed")

    monkeypatch.setattr("src.rag.retriever.get_vector_store", DummyVectorStore)
    monkeypatch.setattr("src.rag.retriever.DashScopeRerank", ExplodingRerank)
    monkeypatch.setattr(settings.models, "rerank_provider", "qwen")
    monkeypatch.setattr(settings.models, "rerank_model", "qwen3-rerank")
    monkeypatch.setattr(settings, "dashscope_api_key", "test-dashscope-key")

    retriever = get_retriever(use_rerank=True)

    assert retriever is base_retriever
