from langchain_core.documents import Document
from typing_extensions import TypedDict


class SubTask(TypedDict):
    """A single sub-task created by the Planner."""
    id: int
    task: str
    priority: str  # "high", "medium", "low"
    keywords: list[str]

class Analysis(TypedDict):
    """Analysis result from the Reasoner."""
    sub_task_id: int
    key_findings: list[str]
    supporting_evidence: list[str]
    contradictions: list[str]
    confidence: float  # 0.0 to 1.0
    gaps: list[str]

class AgentState(TypedDict):
    """
    Shared state passed between all agents in the workflow.
    
    Flow:
    1. User provides `query`
    2. Planner populates `sub_tasks`
    3. Researcher populates `retrieved_docs`
    4. Reasoner populates `analysis`
    5. Synthesiser populates `final_answer`
    """
    # Input
    query: str
    
    # Planner output
    sub_tasks: list[SubTask]
    planning_reasoning: str
    
    # Researcher output
    retrieved_docs: list[Document]
    retrieval_metadata: dict
    
    # Reasoner output
    analysis: list[Analysis]
    overall_assessment: str
    
    # Synthesiser output
    final_answer: str
    sources: list[dict]
    
    # Error handling
    error: str | None
    current_agent: str
