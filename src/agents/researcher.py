import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.llm_factory import get_agent_llm
from src.orchestrator.state import AgentState
from src.rag.retriever import get_retriever
from src.utils.tracing import agent_trace

logger = logging.getLogger(__name__)

class SearchQueryOutput(BaseModel):
    """Structured output for search query reformulation."""
    optimized_queries: list[str] = Field(description="List of optimized search queries")
    filters: dict[str, str] = Field(description="Optional metadata filters", default_factory=dict)

def get_query_optimizer_prompt(parser):
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert Research Assistant. Your goal is to formulate optimal search queries for a vector database.
        
        Input: A sub-task from the planner.
        Output: A list of 2-3 specific, keyword-rich search queries.
        
        The database contains:
        1. Notes from a Notion Knowledge Base
        2. Academic papers from Arxiv
        
        Avoid generic terms. Focus on technical concepts mentioned in the task.
        
        {format_instructions}
        """),
        ("user", "Task: {task}\nKeywords: {keywords}")
    ]).partial(format_instructions=parser.get_format_instructions())

@agent_trace("researcher", tags=["retrieval"])
def researcher_node(state: AgentState) -> dict:
    """
    Researcher Agent: Executes retrieval for each sub-task.
    """
    logger.info("Researcher starting retrieval...")
    
    sub_tasks = state.get("sub_tasks", [])
    if not sub_tasks:
        logger.warning("No sub-tasks found for researcher")
        return {"retrieved_docs": []}
        
    llm = get_agent_llm("researcher")
    retriever = get_retriever(use_rerank=True)
    parser = JsonOutputParser(pydantic_object=SearchQueryOutput)
    chain = get_query_optimizer_prompt(parser) | llm | parser
    
    all_docs: list[Document] = []
    seen_content = set()
    
    for task in sub_tasks:
        try:
            # 1. Optimize query
            logger.info(f"Processing task: {task['task']}")
            query_result = chain.invoke({
                "task": task['task'],
                "keywords": ", ".join(task.get('keywords', []))
            })
            
            # 2. Execute Search
            for query in query_result['optimized_queries']:
                docs = retriever.invoke(query)
                logger.debug(f"Query '{query}' returned {len(docs)} docs")
                
                # 3. Deduplicate and collect
                for doc in docs:
                    # Create a hash of the content to check uniqueness
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        # Add metadata about which task found this doc
                        doc.metadata["retrieval_task_id"] = task["id"]
                        doc.metadata["retrieval_query"] = query
                        all_docs.append(doc)
                        
        except Exception as e:
            logger.error(f"Error processing task {task['id']}: {e}")
            continue
            
    logger.info(f"Researcher retrieved {len(all_docs)} unique documents total")
    
    return {
        "retrieved_docs": all_docs,
        "retrieval_metadata": {"total_docs": len(all_docs)},
        "current_agent": "researcher"
    }
