import logging

from langchain_cohere import CohereEmbeddings

from config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Wrapper for Cohere Embeddings service.
    """
    def __init__(self):
        self._model = CohereEmbeddings(
            cohere_api_key=settings.cohere_api_key,
            model="embed-english-v3.0",
            user_agent="notion-agentic-rag"
        )

    def get_embeddings_model(self) -> CohereEmbeddings:
        return self._model

def get_embeddings() -> CohereEmbeddings:
    """Factory function for embeddings."""
    return EmbeddingService().get_embeddings_model()
