from typing import Literal

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ToolAgentConfig(BaseModel):
    enabled: bool = True
    timeout: float = 30.0
    web_searcher_enabled: bool = True
    code_executor_enabled: bool = True
    citation_validator_enabled: bool = True
    math_solver_enabled: bool = True
    diagram_generator_enabled: bool = True


class DebugConfig(BaseModel):
    enabled: bool = True
    log_dir: str = "./logs"
    log_level: str = "INFO"


ProviderName = Literal["qwen", "openai"]


class AgentModelProfile(BaseModel):
    provider: ProviderName = "qwen"
    model: str = "qwen-flash-2025-07-28"
    temperature: float = 0.0
    max_tokens: int = 2048


class ModelConfig(BaseModel):
    planner: AgentModelProfile = AgentModelProfile(
        provider="qwen",
        model="qwen-flash-2025-07-28",
        temperature=0.0,
        max_tokens=1024,
    )
    researcher: AgentModelProfile = AgentModelProfile(
        provider="qwen",
        model="qwen-flash-2025-07-28",
        temperature=0.0,
        max_tokens=2048,
    )
    reasoner: AgentModelProfile = AgentModelProfile(
        provider="qwen",
        model="qwen-flash-2025-07-28",
        temperature=0.1,
        max_tokens=4096,
    )
    synthesiser: AgentModelProfile = AgentModelProfile(
        provider="qwen",
        model="qwen-flash-2025-07-28",
        temperature=0.3,
        max_tokens=4096,
    )
    tool_agent: AgentModelProfile = AgentModelProfile(
        provider="qwen",
        model="qwen-flash-2025-07-28",
        temperature=0.2,
        max_tokens=2048,
    )
    embedding_provider: ProviderName = "qwen"
    embedding_model: str = "text-embedding-v4"
    rerank_provider: ProviderName = "qwen"
    rerank_model: str = "qwen3-rerank"

class Settings(BaseSettings):
    # API Keys
    dashscope_api_key: str = Field(..., description="DashScope API Key")
    openai_api_key: str | None = Field(None, description="OpenAI API Key")
    openai_base_url: str | None = Field(
        None,
        description="Optional OpenAI-compatible base URL for custom endpoints",
    )
    notion_token: str = Field(..., description="Notion Integration Token")
    notion_database_id: str = Field(..., description="Notion Database ID")
    langsmith_api_key: str = Field(
        ...,
        validation_alias=AliasChoices("langsmith_api_key", "langchain_api_key"),
        description="LangSmith API Key",
    )
    
    # LangSmith
    langsmith_tracing: bool = Field(
        True,
        validation_alias=AliasChoices(
            "langsmith_tracing",
            "langchain_tracing_v2",
        ),
    )
    langsmith_project: str = Field(
        "notion-agentic-rag",
        validation_alias=AliasChoices("langsmith_project", "langchain_project"),
    )
    langsmith_endpoint: str = Field(
        "https://eu.api.smith.langchain.com",
        validation_alias=AliasChoices("langsmith_endpoint", "langchain_endpoint"),
    )
    
    # Vector Store
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "notion_knowledge_base"
    embedding_batch_size: int = 16
    embedding_delay: float = 1.0  # Prevent rate limiting (free tier)

    model_config = SettingsConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8")
    
    # RAG Settings
    chunk_size: int = 2048
    chunk_overlap: int = 200
    retrieval_k: int = 10
    rerank_top_n: int = 5
    
    # Model Config
    models: ModelConfig = ModelConfig()

    # Tool Agents
    tool_agents: ToolAgentConfig = ToolAgentConfig()

    # Debugging & observability
    debug: DebugConfig = DebugConfig()

settings = Settings()
