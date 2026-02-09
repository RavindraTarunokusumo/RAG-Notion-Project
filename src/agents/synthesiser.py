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
        3. **IMPORTANT: Cite your sources using numbered citations [1], [2], etc. Place the citation number immediately after the relevant statement. If a statement is supported by multiple sources, list all relevant citation numbers in separate brackets (e.g., [1][3])**.
        4. Each source should be referenced by its index number from the sources list provided.
        5. If there are conflicting findings, highlight them with appropriate citations.
        6. If information is missing, clearly state what could not be found.
        7. Format your response in clean Markdown.
        
        Example:
        "The transformer architecture revolutionized NLP [1]. Recent work shows attention mechanisms are key [2]."
        """),
        ("user", """
        Original Query: {query}
        
        Planner Reasoning: {reasoning}
        
        Detailed Analysis:
        {analysis}
        
        Overall Assessment: {assessment}
        
        Available Sources:
        {sources_list}
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
        
        # Extract and enrich sources from retrieved docs
        sources = []
        for d in state.get("retrieved_docs", []):
            # Extract metadata
            meta = d.metadata
            title = meta.get("title") or meta.get("Title") or "Unknown"
            source_type = meta.get("source", "unknown")
            
            # Build enriched source object
            source = {
                "source": source_type,
                "title": title,
                "url": meta.get("url") or meta.get("notion_url"),
                "snippet": d.page_content[:300] if d.page_content else None,
            }
            
            # Add source-specific metadata
            if source_type == "arxiv":
                source.update({
                    "arxiv_url": f"https://arxiv.org/abs/{meta.get('arxiv_id')}" if meta.get('arxiv_id') else None,
                    "arxiv_id": meta.get("arxiv_id"),
                    "authors": meta.get("Authors", []) if isinstance(meta.get("Authors"), list) else None,
                    "published": meta.get("Published") or meta.get("publication_date"),
                    "abstract": meta.get("Summary"),
                })
            elif source_type == "notion":
                source.update({
                    "notion_url": meta.get("url"),
                    "topic": meta.get("topic"),
                    "keywords": meta.get("keywords"),
                    "published": meta.get("publication_date"),
                    "category": meta.get("category"),
                })
            
            sources.append(source)
        
        # Deduplicate sources based on title
        unique_sources = []
        seen = set()
        for s in sources:
            unique_key = f"{s['source']}:{s['title']}"
            if unique_key not in seen:
                seen.add(unique_key)
                unique_sources.append(s)
        
        # Create sources list for prompt
        sources_list = "\n".join([
            f"[{i+1}] {s['title']} ({s['source']})"
            for i, s in enumerate(unique_sources)
        ])
        
        final_answer = chain.invoke({
            "query": query,
            "reasoning": state.get("planning_reasoning", ""),
            "analysis": analysis_str,
            "assessment": state.get("overall_assessment", ""),
            "sources_list": sources_list
        })
        
        logger.info("Synthesiser completed generation")
        
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
