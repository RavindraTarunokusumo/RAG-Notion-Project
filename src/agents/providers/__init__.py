from src.agents.providers.base import ChatProvider
from src.agents.providers.openai import OpenAIChatProvider
from src.agents.providers.qwen import QwenChatProvider

_CHAT_PROVIDER_REGISTRY: dict[str, ChatProvider] = {
    QwenChatProvider.name: QwenChatProvider(),
    OpenAIChatProvider.name: OpenAIChatProvider(),
}


def get_chat_provider(name: str) -> ChatProvider:
    provider = _CHAT_PROVIDER_REGISTRY.get(name)
    if provider is None:
        supported = ", ".join(sorted(_CHAT_PROVIDER_REGISTRY))
        raise ValueError(
            f"Unsupported chat provider '{name}'. Supported: {supported}."
        )
    return provider


def list_chat_providers() -> list[str]:
    return sorted(_CHAT_PROVIDER_REGISTRY)
