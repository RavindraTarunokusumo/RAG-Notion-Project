from dataclasses import dataclass
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass(frozen=True)
class ChatModelRequest:
    model: str
    temperature: float
    max_tokens: int


class ChatProvider(Protocol):
    name: str

    def create_chat_model(self, request: ChatModelRequest) -> BaseChatModel:
        ...
