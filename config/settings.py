from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CohereModelConfig(BaseModel):
    planner_model: str = "command-r-08-2024"
    researcher_model: str = "command-r-08-2024"
    reasoner_model: str = "command-a-reasoning-08-2025"
    synthesiser_model: str = "command-r-plus-08-2024"
    
    planner_temperature: float = 0.0
    researcher_temperature: float = 0.0
    reasoner_temperature: float = 0.1
    synthesiser_temperature: float = 0.3

class Settings(BaseSettings):
    # API Keys
    cohere_api_key: str = Field(..., description="Cohere API Key")
    notion_token: str = Field(..., description="Notion Integration Token")
    notion_database_id: str = Field(..., description="Notion Database ID")
    langsmith_api_key: str = Field(..., validation_alias=AliasChoices("langsmith_api_key", "langchain_api_key"), description="LangSmith API Key")
    
    # LangSmith
    langsmith_tracing: bool = Field(True, validation_alias=AliasChoices("langsmith_tracing", "langchain_tracing_v2"))
    langsmith_project: str = Field("notion-agentic-rag", validation_alias=AliasChoices("langsmith_project", "langchain_project"))
    langsmith_endpoint: str = Field("https://eu.api.smith.langchain.com", validation_alias=AliasChoices("langsmith_endpoint", "langchain_endpoint"))
    
    # Vector Store
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "notion_knowledge_base"
    embedding_batch_size: int = 10
    embedding_delay: float = 10.0 # Prevent rate limiting (free tier)

    model_config = SettingsConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8")
    
    # RAG Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 10
    rerank_top_n: int = 5
    
    # Model Config
    models: CohereModelConfig = CohereModelConfig()

settings = Settings()
