from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class CohereModelConfig(BaseModel):
    planner_model: str = "command-r-08-2024"
    researcher_model: str = "command-r-08-2024"
    reasoner_model: str = "command-r-plus-08-2024"
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
    langchain_api_key: str = Field(..., description="LangSmith API Key")
    
    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_project: str = "notion-agentic-rag"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    
    # Vector Store
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "notion_knowledge_base"
    
    # RAG Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 10
    rerank_top_n: int = 5
    
    # Model Config
    models: CohereModelConfig = CohereModelConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
