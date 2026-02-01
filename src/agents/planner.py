import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.llm_factory import get_agent_llm
from src.orchestrator.state import AgentState, SubTask

logger = logging.getLogger(__name__)

class PlanOutput(BaseModel):
    """Structured output for the planning step."""
    reasoning: str = Field(description="Reasoning behind the breakdown")
    sub_tasks: list[SubTask] = Field(description="List of sub-tasks to execute")

def get_planner_prompt(parser):
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert Research Planner. Your goal is to break down complex queries into specific, actionable retrieval sub-tasks.

        GUIDELINES:
        1. Analyze the user's query carefully.
        2. Break it down into 3-5 distinct sub-tasks.
        3. Assign a priority (high/medium/low) to each task.
        4. Identify 3-5 specific search keywords for each task.
        5. DO NOT try to answer the query yourself.
        
        Focus on retrieving factual information from a Notion knowledge base and academic papers.
        
        {format_instructions}
        """),
        ("user", "Query: {query}")
    ]).partial(format_instructions=parser.get_format_instructions())

def planner_node(state: AgentState) -> dict:
    """
    Planner Agent: Decomposes the query into sub-tasks.
    """
    logger.info(f"Planner processing query: {state['query']}")
    
    try:
        llm = get_agent_llm("planner")
        parser = JsonOutputParser(pydantic_object=PlanOutput)
        
        chain = get_planner_prompt(parser) | llm | parser
        
        result = chain.invoke({"query": state["query"]})
        
        logger.info(f"Planner generated {len(result['sub_tasks'])} sub-tasks")
        
        return {
            "sub_tasks": result["sub_tasks"],
            "planning_reasoning": result["reasoning"],
            "current_agent": "planner"
        }
        
    except Exception as e:
        logger.error(f"Planner failed: {e}")
        return {
            "error": str(e),
            "current_agent": "planner"
        }
