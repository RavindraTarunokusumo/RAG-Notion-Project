import os

# Ensure required settings exist for import-time BaseSettings validation.
os.environ.setdefault("DASHSCOPE_API_KEY", "test-dashscope-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion-token")
os.environ.setdefault("NOTION_DATABASE_ID", "test-notion-db")
os.environ.setdefault("LANGCHAIN_API_KEY", "test-langsmith-key")
