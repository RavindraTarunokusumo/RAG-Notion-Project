import logging

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
    
    sub_tasks = state.get("sub_tasks", [])
    docs = state.get("retrieved_docs", [])
    
    if not sub_tasks:
        logger.warning("No sub-tasks to analyze")
        return {"analysis": [], "overall_assessment": "No tasks to analyze"}
        
    doc_str = format_docs(docs) if docs else "No documents retrieved."
    
    try:
        llm = get_agent_llm("reasoner")
        parser = JsonOutputParser(pydantic_object=ReasonerOutput)
        chain = get_reasoner_prompt(parser) | llm | parser
        
        result = chain.invoke({
            "sub_tasks": str(sub_tasks),
            "documents": doc_str
        })
        
        logger.info(f"Reasoner completed analysis with {len(result['analysis'])} items")
        
        return {
            "analysis": result["analysis"],
            "overall_assessment": result["overall_assessment"],
            "current_agent": "reasoner"
        }
        
    except Exception as e:
        logger.error(f"Reasoner failed: {e}")
        return {
            "error": f"Reasoner Error: {str(e)}",
            "current_agent": "reasoner"
        }
