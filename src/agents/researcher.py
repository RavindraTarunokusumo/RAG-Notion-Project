import logging
import time

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


def _classify_retrieval_error(error: Exception) -> tuple[str, bool]:
    """Classify retrieval errors into fatal vs. recoverable categories."""
    message = str(error).lower()

    if (
        "status_code: 429" in message
        and (
            "trial key" in message
            or "quota" in message
            or "rate limit" in message
            or "too many requests" in message
            or "limited to 1000 api calls" in message
        )
    ):
        return "quota_exhausted", True

    if (
        "status_code: 401" in message
        or "status_code: 403" in message
        or "unauthorized" in message
        or "forbidden" in message
    ):
        return "auth_failed", True

    return "transient", False

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
        return {
            "retrieved_docs": [],
            "retrieval_metadata": {
                "total_docs": 0,
                "processed_tasks": 0,
                "failed_tasks": 0,
                "fatal": False,
                "error_type": None,
                "aborted": False,
            },
            "current_agent": "researcher",
        }
        
    llm = get_agent_llm("researcher")
    retriever = get_retriever(use_rerank=True)
    parser = JsonOutputParser(pydantic_object=SearchQueryOutput)
    chain = get_query_optimizer_prompt(parser) | llm | parser
    
    all_docs: list[Document] = []
    seen_content = set()
    failed_tasks = 0
    processed_tasks = 0
    fatal_error: str | None = None
    fatal_error_type: str | None = None

    for task in sub_tasks:
        processed_tasks += 1
        try:
            # Rate limit mitigation for Trial Tier (10 calls/min)
            time.sleep(6) 
            
            # 1. Optimize query
            logger.info(f"Processing task: {task['task']}")
            query_result = chain.invoke({
                "task": task['task'],
                "keywords": ", ".join(task.get('keywords', []))
            })
            
            # 2. Execute Search
            for query in query_result['optimized_queries']:
                time.sleep(6) # Rate limit mitigation
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
            failed_tasks += 1
            error_type, is_fatal = _classify_retrieval_error(e)
            task_id = task.get("id", "unknown")

            if is_fatal:
                fatal_error_type = error_type
                fatal_error = str(e)
                logger.error(
                    "Fatal retrieval error on task %s (%s): %s",
                    task_id,
                    error_type,
                    e,
                )
                break

            logger.error(
                "Error processing task %s (%s): %s",
                task_id,
                error_type,
                e,
            )
            continue

    if fatal_error is not None:
        logger.error(
            "Researcher aborted after fatal retrieval error (%s).",
            fatal_error_type,
        )
        return {
            "retrieved_docs": all_docs,
            "retrieval_metadata": {
                "total_docs": len(all_docs),
                "processed_tasks": processed_tasks,
                "failed_tasks": failed_tasks,
                "fatal": True,
                "error_type": fatal_error_type,
                "aborted": True,
            },
            "error": (
                "Researcher Error: Retrieval aborted due to "
                f"{fatal_error_type}. Check API quota/credentials and retry. "
                f"Details: {fatal_error}"
            ),
            "current_agent": "researcher",
        }

    logger.info(f"Researcher retrieved {len(all_docs)} unique documents total")

    return {
        "retrieved_docs": all_docs,
        "retrieval_metadata": {
            "total_docs": len(all_docs),
            "processed_tasks": processed_tasks,
            "failed_tasks": failed_tasks,
            "fatal": False,
            "error_type": None,
            "aborted": False,
        },
        "current_agent": "researcher"
    }
