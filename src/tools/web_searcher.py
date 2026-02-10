"""NRAG-028: Web Searcher Tool Agent."""

import logging
from typing import Any

from src.tools.base import AgentCard, ToolAgent, ToolResult

logger = logging.getLogger(__name__)

_TRIGGER_KEYWORDS = {"latest", "current", "recent", "2024", "2025", "2026", "news", "today", "now"}


class WebSearcherAgent(ToolAgent):
    """Tool agent that performs web searches using DuckDuckGo."""

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="web_searcher",
            description="Searches the web for current information using DuckDuckGo",
            version="1.0.0",
            capabilities=["web_search", "news_search"],
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            output_schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "snippet": {"type": "string"},
                    },
                },
            },
            endpoint="tool://web_searcher",
        )

    def can_handle(self, task_description: str) -> float:
        words = set(task_description.lower().split())
        matches = words & _TRIGGER_KEYWORDS
        if matches:
            return min(0.3 + 0.2 * len(matches), 1.0)
        return 0.0

    async def execute(self, task: dict[str, Any]) -> ToolResult:
        query = task.get("query", "")
        max_results = task.get("max_results", 5)

        if not query:
            return ToolResult(success=False, data=None, error="No query provided")

        try:
            from langchain_community.utilities import (
                DuckDuckGoSearchAPIWrapper,
            )

            search = DuckDuckGoSearchAPIWrapper(max_results=max_results)
            raw_results = search.results(query, max_results=max_results)

            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "snippet": r.get("snippet", ""),
                }
                for r in raw_results
            ]

            return ToolResult(
                success=True,
                data=results,
                metadata={"query": query, "result_count": len(results)},
            )
        except Exception as e:
            logger.exception("Web search failed")
            return ToolResult(success=False, data=None, error=str(e))
