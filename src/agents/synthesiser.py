import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agents.llm_factory import get_agent_llm
from src.orchestrator.state import AgentState
from src.utils.tracing import agent_trace

logger = logging.getLogger(__name__)

def get_synthesiser_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert Technical Writer. Your goal is to synthesize a final answer based on the analysis provided.

        GUIDELINES:
        1. Answer the user's original query directly and comprehensively.
        2. Use the provided analysis to structure your response.
        3. Cite your sources implicitly (e.g., "According to Notion notes...", "Research paper X suggests...").
        4. If there are conflicting findings, highlight them.
        5. If information is missing, clearly state what could not be found.
        6. Format your response in clean Markdown.
        """),
        ("user", """
        Original Query: {query}
        
        Planner Reasoning: {reasoning}
        
        Detailed Analysis:
        {analysis}
        
        Overall Assessment: {assessment}
        """)
    ])

@agent_trace("synthesiser", tags=["synthesis"])
def synthesiser_node(state: AgentState) -> dict:
    """
    Synthesiser Agent: Generates the final answer.
    """
    logger.info("Synthesiser generating final answer...")
    
    query = state.get("query", "")
    analysis = state.get("analysis", [])
    
    try:
        llm = get_agent_llm("synthesiser")
        chain = get_synthesiser_prompt() | llm | StrOutputParser()
        
        # Format analysis for the prompt
        analysis_str = "\n".join([
            f"- Task {item.get('sub_task_id')}: {item.get('key_findings')}" 
            for item in analysis
        ])
        
        final_answer = chain.invoke({
            "query": query,
            "reasoning": state.get("planning_reasoning", ""),
            "analysis": analysis_str,
            "assessment": state.get("overall_assessment", "")
        })
        
        logger.info("Synthesiser completed generation")
        
        # Extract sources from retrieved docs
        sources = [
            {"source": d.metadata.get("source"), "title": d.metadata.get("title", "Unknown")}
            for d in state.get("retrieved_docs", [])
        ]
        
        # Deduplicate sources based on source string
        unique_sources = []
        seen = set()
        for s in sources:
            if s["source"] not in seen:
                seen.add(s["source"])
                unique_sources.append(s)
        
        return {
            "final_answer": final_answer,
            "sources": unique_sources,
            "current_agent": "synthesiser"
        }
        
    except Exception as e:
        logger.error(f"Synthesiser failed: {e}")
        return {
            "error": f"Synthesiser Error: {str(e)}",
            "current_agent": "synthesiser"
        }
