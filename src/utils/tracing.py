import os
from functools import wraps
from typing import Callable
from langsmith import traceable
from config.settings import settings

def initialize_tracing():
    """Initialize LangSmith tracing with project settings."""
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

def agent_trace(
    agent_name: str,
    model: str = None,
    tags: list[str] = None
) -> Callable:
    """Decorator for tracing agent function calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @traceable(
            run_type="chain",
            name=agent_name,
            tags=tags or [],
            metadata={"model": model} if model else {}
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
