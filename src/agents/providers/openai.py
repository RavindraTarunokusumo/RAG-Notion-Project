from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config.settings import settings
from src.agents.providers.base import ChatModelRequest


class OpenAIChatProvider:
    name = "openai"

    def create_chat_model(self, request: ChatModelRequest) -> BaseChatModel:
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAI chat provider."
            )

        kwargs = {
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "api_key": settings.openai_api_key,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url

        return ChatOpenAI(**kwargs)
