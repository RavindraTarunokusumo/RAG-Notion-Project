import os
from collections.abc import Callable
from functools import wraps

from langsmith import traceable

from config.settings import settings


def initialize_tracing():
    """Initialize LangSmith tracing with project settings."""
    os.environ["LANGSMITH_TRACING"] = str(settings.langsmith_tracing).lower()
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

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
