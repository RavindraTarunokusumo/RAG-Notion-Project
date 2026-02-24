import logging
import re
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.llm_factory import get_agent_llm
from src.orchestrator.state import AgentState, Analysis
from src.utils.tracing import agent_trace

logger = logging.getLogger(__name__)

class ReasonerOutput(BaseModel):
    """Structured output for search query reformulation."""
    analysis: list[Analysis] = Field(description="List of analysis for each sub-task")
    overall_assessment: str = Field(description="High-level summary of what was found and what is missing")


def _content_to_text(raw_output: Any) -> str:
    """Normalize model output content to plain text."""
    content = getattr(raw_output, "content", raw_output)

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("thinking") or str(item)
            elif hasattr(item, "text"):
                text = str(item.text)
            elif hasattr(item, "thinking"):
                text = str(item.thinking)
            else:
                text = str(item)
            parts.append(text)
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def _strip_thinking_blocks(text: str) -> str:
    return re.sub(
        r"<thinking>.*?</thinking>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def _extract_json_candidate(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1].strip()


def _parse_reasoner_output(raw_output: Any, parser: JsonOutputParser) -> dict:
    """Parse reasoner output with resilient fallbacks."""
    raw_text = _content_to_text(raw_output)
    stripped = _strip_thinking_blocks(raw_text)
    json_candidate = _extract_json_candidate(stripped) or _extract_json_candidate(
        raw_text
    )

    attempts = [raw_text]
    if stripped and stripped != raw_text:
        attempts.append(stripped)
    if json_candidate and json_candidate not in attempts:
        attempts.append(json_candidate)

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            parsed = parser.parse(attempt)
            if isinstance(parsed, dict):
                return parsed
        except Exception as error:
            last_error = error

    raise ValueError("Failed to parse structured reasoner output.") from last_error


def _build_no_doc_analysis(sub_tasks: list[dict]) -> tuple[list[Analysis], str]:
    """Build deterministic analysis when retrieval produced no documents."""
    analysis: list[Analysis] = []
    for task in sub_tasks:
        task_id = int(task.get("id", -1))
        task_text = str(task.get("task", "Unknown task"))
        analysis.append(
            {
                "sub_task_id": task_id,
                "key_findings": [],
                "supporting_evidence": [],
                "contradictions": [],
                "confidence": 0.0,
                "gaps": [f"No documents retrieved for sub-task {task_id}: {task_text}"],
            }
        )

    overall_assessment = (
        "No documents were retrieved, so evidence-based analysis could not be "
        "performed for the planned sub-tasks."
    )
    return analysis, overall_assessment

def get_reasoner_prompt(parser):
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert Research Analyst. Your goal is to analyze retrieved documents against the planned sub-tasks.

        Reflect on the gathered information and determine if it satisfies the requirements of each sub-task.
        
        GUIDELINES:
        1. For each sub-task, cross-reference the retrieved documents.
        2. Extract key findings, supporting evidence, and contradictions.
        3. Identify any gaps where information is missing.
        4. Assign a confidence score (0.0 to 1.0) based on the quality of evidence.
        
        {format_instructions}
        """),
        ("user", """
        Sub-Tasks: {sub_tasks}
        
        Retrieved Documents:
        {documents}
        """)
    ]).partial(format_instructions=parser.get_format_instructions())

def format_docs(docs):
    return "\n\n".join(
        [f"Document {i+1} (Source: {d.metadata.get('source', 'Unknown')}):\n{d.page_content}" 
         for i, d in enumerate(docs)]
    )

@agent_trace("reasoner", tags=["reasoning"])
def reasoner_node(state: AgentState) -> dict:
    """
    Reasoner Agent: Analyzes retrieved documents to answer sub-tasks.
    """
    logger.info("Reasoner starting analysis...")

    if state.get("error"):
        logger.warning("Skipping reasoner due to upstream error in state.")
        return {
            "analysis": [],
            "overall_assessment": "Reasoner skipped due to upstream retrieval error.",
            "current_agent": "reasoner",
        }
    
    sub_tasks = state.get("sub_tasks", [])
    docs = state.get("retrieved_docs", [])
    
    if not sub_tasks:
        logger.warning("No sub-tasks to analyze")
        return {
            "analysis": [],
            "overall_assessment": "No tasks to analyze",
            "current_agent": "reasoner",
        }

    if not docs:
        logger.warning("No documents retrieved; returning deterministic gap analysis.")
        analysis, overall_assessment = _build_no_doc_analysis(sub_tasks)
        return {
            "analysis": analysis,
            "overall_assessment": overall_assessment,
            "current_agent": "reasoner",
        }
        
    doc_str = format_docs(docs)
    
    try:
        llm = get_agent_llm("reasoner")
        parser = JsonOutputParser(pydantic_object=ReasonerOutput)
        chain = get_reasoner_prompt(parser) | llm
        
        raw_result = chain.invoke({
            "sub_tasks": str(sub_tasks),
            "documents": doc_str
        })
        result = _parse_reasoner_output(raw_result, parser)
        
        logger.info(f"Reasoner completed analysis with {len(result.get('analysis', []))} items")
        
        return {
            "analysis": result.get("analysis", []),
            "overall_assessment": result.get("overall_assessment", "No overall assessment provided by the model."),
            "current_agent": "reasoner"
        }
        
    except Exception as e:
        logger.error(f"Reasoner failed: {e}")
        details = str(e)
        if len(details) > 300:
            details = details[:300] + "..."
        return {
            "error": f"Reasoner Error: {details}",
            "current_agent": "reasoner"
        }
