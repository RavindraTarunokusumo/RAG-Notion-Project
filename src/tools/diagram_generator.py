"""NRAG-032: Diagram Generator Tool Agent."""

import logging
from typing import Any

from src.tools.base import AgentCard, ToolAgent, ToolResult

logger = logging.getLogger(__name__)

_TRIGGER_KEYWORDS = {
    "diagram", "flowchart", "graph", "chart", "mermaid",
    "sequence", "class diagram", "visualize", "architecture",
}


class DiagramGeneratorAgent(ToolAgent):
    """Tool agent that generates Mermaid diagram syntax from textual descriptions."""

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="diagram_generator",
            description="Generates Mermaid diagram syntax from textual descriptions using LLM",
            version="1.0.0",
            capabilities=["diagram_generation", "mermaid"],
            input_schema={
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Textual description of the diagram"},
                    "diagram_type": {
                        "type": "string",
                        "enum": ["flowchart", "sequence", "classDiagram", "stateDiagram", "erDiagram", "gantt"],
                        "default": "flowchart",
                    },
                },
                "required": ["description"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "mermaid_syntax": {"type": "string"},
                    "diagram_type": {"type": "string"},
                },
            },
            endpoint="tool://diagram_generator",
        )

    def can_handle(self, task_description: str) -> float:
        lower = task_description.lower()
        matches = sum(1 for kw in _TRIGGER_KEYWORDS if kw in lower)
        if matches:
            return min(0.3 + 0.2 * matches, 1.0)
        return 0.0

    async def execute(self, task: dict[str, Any]) -> ToolResult:
        description = task.get("description", "")
        diagram_type = task.get("diagram_type", "flowchart")

        if not description:
            return ToolResult(success=False, data=None, error="No description provided")

        try:
            from src.agents.llm_factory import get_agent_llm

            llm = get_agent_llm("tool_agent")

            prompt = (
                f"Generate a Mermaid {diagram_type} diagram for the following description. "
                f"Output ONLY the Mermaid syntax, no explanation or markdown fences.\n\n"
                f"Description: {description}"
            )

            response = llm.invoke(prompt)
            mermaid_syntax = response.content.strip()

            # Clean up if wrapped in markdown fences
            if mermaid_syntax.startswith("```"):
                lines = mermaid_syntax.split("\n")
                # Remove first and last fence lines
                lines = [line for line in lines if not line.strip().startswith("```")]
                mermaid_syntax = "\n".join(lines).strip()

            return ToolResult(
                success=True,
                data={
                    "mermaid_syntax": mermaid_syntax,
                    "diagram_type": diagram_type,
                },
                metadata={"description": description},
            )
        except Exception as e:
            logger.exception("Diagram generation failed")
            return ToolResult(success=False, data=None, error=str(e))
