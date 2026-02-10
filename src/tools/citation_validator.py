"""NRAG-030: Citation Validator Tool Agent."""

import logging
import re
from typing import Any

from src.tools.base import AgentCard, ToolAgent, ToolResult

logger = logging.getLogger(__name__)

_ARXIV_ID_PATTERN = re.compile(r"\d{4}\.\d{4,5}(v\d+)?")


class CitationValidatorAgent(ToolAgent):
    """Tool agent that validates academic citations using the arXiv API."""

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="citation_validator",
            description="Validates academic citations and verifies arXiv paper metadata",
            version="1.0.0",
            capabilities=["citation_validation", "arxiv_verification"],
            input_schema={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv paper ID (e.g., 2301.07041)"},
                    "expected_title": {"type": "string", "description": "Expected paper title for matching"},
                    "expected_authors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Expected author names",
                    },
                },
                "required": ["arxiv_id"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "valid": {"type": "boolean"},
                    "paper_title": {"type": "string"},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "title_match": {"type": "boolean"},
                    "author_match": {"type": "boolean"},
                },
            },
            endpoint="tool://citation_validator",
        )

    def can_handle(self, task_description: str) -> float:
        lower = task_description.lower()
        keywords = {"citation", "arxiv", "paper", "reference", "verify", "validate"}
        matches = sum(1 for kw in keywords if kw in lower)
        # Also check for arXiv ID pattern
        if _ARXIV_ID_PATTERN.search(task_description):
            matches += 2
        if matches:
            return min(0.3 + 0.15 * matches, 1.0)
        return 0.0

    async def execute(self, task: dict[str, Any]) -> ToolResult:
        arxiv_id = task.get("arxiv_id", "")
        expected_title = task.get("expected_title", "")
        expected_authors = task.get("expected_authors", [])

        if not arxiv_id:
            return ToolResult(success=False, data=None, error="No arxiv_id provided")

        # Clean arXiv ID
        arxiv_id = arxiv_id.strip()
        if not _ARXIV_ID_PATTERN.match(arxiv_id):
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid arXiv ID format: {arxiv_id}",
            )

        try:
            import arxiv

            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(client.results(search))

            if not results:
                return ToolResult(
                    success=True,
                    data={"valid": False, "reason": "Paper not found on arXiv"},
                    metadata={"arxiv_id": arxiv_id},
                )

            paper = results[0]
            actual_title = paper.title
            actual_authors = [a.name for a in paper.authors]

            title_match = (
                expected_title.lower().strip() in actual_title.lower().strip()
                if expected_title
                else None
            )
            author_match = (
                any(
                    exp.lower() in auth.lower()
                    for exp in expected_authors
                    for auth in actual_authors
                )
                if expected_authors
                else None
            )

            return ToolResult(
                success=True,
                data={
                    "valid": True,
                    "paper_title": actual_title,
                    "authors": actual_authors[:10],  # Limit to first 10
                    "title_match": title_match,
                    "author_match": author_match,
                    "published": paper.published.isoformat() if paper.published else None,
                    "pdf_url": paper.pdf_url,
                },
                metadata={"arxiv_id": arxiv_id},
            )
        except Exception as e:
            logger.exception("Citation validation failed")
            return ToolResult(success=False, data=None, error=str(e))
