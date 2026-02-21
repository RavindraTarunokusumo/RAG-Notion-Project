from langchain_community.chat_models import ChatTongyi
from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import settings
from src.agents.providers.base import ChatModelRequest


class QwenChatProvider:
    name = "qwen"

    def create_chat_model(self, request: ChatModelRequest) -> BaseChatModel:
        if not settings.dashscope_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is required for Qwen chat provider."
            )

        return ChatTongyi(
            model=request.model,
            api_key=settings.dashscope_api_key,
            model_kwargs={
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
        )
