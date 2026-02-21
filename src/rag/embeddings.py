import logging

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Wrapper for embedding provider service.
    """

    def __init__(self):
        if settings.models.embedding_provider != "qwen":
            raise ValueError(
                "Only 'qwen' embedding provider is supported in this build."
            )

        self._model = DashScopeEmbeddings(
            dashscope_api_key=settings.dashscope_api_key,
            model=settings.models.embedding_model,
        )

    def get_embeddings_model(self) -> Embeddings:
        return self._model


def get_embeddings() -> Embeddings:
    """Factory function for embeddings."""
    return EmbeddingService().get_embeddings_model()
