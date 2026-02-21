## Session 4: Planner & Researcher Agents

**Session Token Budget:** ~80,000 tokens  
**Focus:** Implement the first two agents in the pipeline

---

#### NRAG-016: Agent State Schema

**Priority:** P0 - Critical  
**Token Estimate:** 5,000 tokens  
**Status:** To Do

**Description:**  
Define the shared state schema for all agents in the LangGraph workflow.

**Acceptance Criteria:**

- [x] TypedDict for agent state
- [x] All required fields defined
- [x] Clear documentation of each field
- [x] Support for error states

**Implementation:**

```python
# src/orchestrator/state.py
from typing import TypedDict, List, Any, Optional
from langchain_core.documents import Document

class SubTask(TypedDict):
    """A single sub-task created by the Planner."""
    id: int
    task: str
    priority: str  # "high", "medium", "low"
    keywords: List[str]

class Analysis(TypedDict):
    """Analysis result from the Reasoner."""
    sub_task_id: int
    key_findings: List[str]
    supporting_evidence: List[str]
    contradictions: List[str]
    confidence: float  # 0.0 to 1.0
    gaps: List[str]

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
    sub_tasks: List[SubTask]
    planning_reasoning: str
    
    # Researcher output
    retrieved_docs: List[Document]
    retrieval_metadata: dict
    
    # Reasoner output
    analysis: List[Analysis]
    overall_assessment: str
    
    # Synthesiser output
    final_answer: str
    sources: List[dict]
    
    # Error handling
    error: Optional[str]
    current_agent: str
```

---

#### NRAG-017: LLM Factory

**Priority:** P0 - Critical  
**Token Estimate:** 6,000 tokens  
**Status:** To Do

**Description:**  
Create factory for instantiating Cohere LLMs with agent-specific configurations.

**Acceptance Criteria:**

- [x] Centralized LLM creation
- [x] Agent-specific model/temperature settings
- [x] Retry logic configuration
- [x] Consistent error handling

**Implementation:**

```python
# src/agents/llm_factory.py
import logging
from typing import Literal
from langchain_cohere import ChatCohere
from config.settings import settings

logger = logging.getLogger(__name__)

AgentType = Literal["planner", "researcher", "reasoner", "synthesiser"]

# Model assignments based on task complexity
AGENT_CONFIGS = {
    "planner": {
        "model": settings.models.planner_model,
        "temperature": settings.models.planner_temperature,
        "max_tokens": 1024,
        "description": "Task decomposition - fast, focused"
    },
    "researcher": {
        "model": settings.models.researcher_model,
        "temperature": settings.models.researcher_temperature,
        "max_tokens": 2048,
        "description": "Query formulation - precise, systematic"
    },
    "reasoner": {
        "model": settings.models.reasoner_model,
        "temperature": settings.models.reasoner_temperature,
        "max_tokens": 4096,
        "description": "Complex analysis - powerful, nuanced"
    },
    "synthesiser": {
        "model": settings.models.synthesiser_model,
        "temperature": settings.models.synthesiser_temperature,
        "max_tokens": 4096,
        "description": "Response generation - creative, coherent"
    }
}

def get_agent_llm(agent_type: AgentType) -> ChatCohere:
    """
    Get a configured LLM for the specified agent type.
    
    Model sizes:
    - Planner: command-r-08-2024 (35B) - Fast task decomposition
    - Researcher: command-r-08-2024 (35B) - Efficient query handling
    - Reasoner: command-r-plus-08-2024 (104B) - Deep analysis
    - Synthesiser: command-r-plus-08-2024 (104B) - Quality generation
    """
    if agent_type not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    config = AGENT_CONFIGS[agent_type]
    
    llm = ChatCohere(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        cohere_api_key=settings.cohere_api_key
    )
    
    logger.debug(f"Created LLM for {agent_type}: {config['model']} ({config['description']})")
    
    return llm

def get_model_info(agent_type: AgentType) -> dict:
    """Get information about the model used for an agent."""
    return AGENT_CONFIGS.get(agent_type, {})
```

---

#### NRAG-018: Planner Agent Implementation

**Priority:** P0 - Critical  
**Token Estimate:** 12,000 tokens  
**Status:** To Do

**Description:**  
Implement the Planner agent that decomposes queries into sub-tasks.

**Acceptance Criteria:**

- [x] Query decomposition into 2-5 sub-tasks
- [x] Priority assignment for each sub-task
- [x] Keyword extraction for retrieval
- [x] JSON output parsing with fallback
- [x] LangSmith tracing integration

**Implementation:**

```python
# src/agents/planner.py
import logging
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.orchestrator.state import AgentState, SubTask
from src.agents.llm_factory import get_agent_llm
from src.utils.tracing import agent_trace
from src.utils.helpers import safe_json_parse

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a Planning Agent specialized in breaking down complex questions into actionable research tasks.

Your role is to analyze the user's query and decompose it into 2-5 specific sub-tasks that can be researched independently from a knowledge base about AI, machine learning, and related topics.

Guidelines:
1. Each sub-task should be a focused, specific question
2. Sub-tasks should cover different aspects of the query
3. Assign priority based on importance to answering the main query
4. Extract keywords that will help with document retrieval
5. Consider what information would be most valuable

Output your response as a JSON object with this exact structure:
{{
    "original_query": "the user's original question",
    "sub_tasks": [
        {{
            "id": 1,
            "task": "Specific question or research task",
            "priority": "high|medium|low",
            "keywords": ["keyword1", "keyword2"]
        }}
    ],
    "reasoning": "Brief explanation of why you decomposed the query this way"
}}

Important: Output ONLY the JSON object, no additional text."""

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT),
    ("human", "{query}")
])

class PlannerAgent:
    """
    Decomposes user queries into actionable sub-tasks.
    
    Uses Command R (smaller model) for fast, focused task decomposition.
    """
    
    def __init__(self):
        self.llm = get_agent_llm("planner")
        self.chain = PLANNER_PROMPT | self.llm
    
    @agent_trace("planner", model="command-r-08-2024")
    def __call__(self, state: AgentState) -> AgentState:
        """
        Process state and generate sub-tasks.
        
        Args:
            state: Current agent state with query
        
        Returns:
            Updated state with sub_tasks populated
        """
        logger.info(f"Planner processing query: {state['query'][:100]}...")
        
        try:
            response = self.chain.invoke({"query": state["query"]})
            content = response.content
            
            # Parse JSON output
            parsed = safe_json_parse(content)
            
            if not parsed:
                logger.warning("Failed to parse planner output, using fallback")
                parsed = self._create_fallback_tasks(state["query"])
            
            sub_tasks = parsed.get("sub_tasks", [])
            reasoning = parsed.get("reasoning", "")
            
            logger.info(f"Planner created {len(sub_tasks)} sub-tasks")
            
            return {
                **state,
                "sub_tasks": sub_tasks,
                "planning_reasoning": reasoning,
                "current_agent": "planner"
            }
            
        except Exception as e:
            logger.error(f"Planner error: {e}")
            return {
                **state,
                "sub_tasks": self._create_fallback_tasks(state["query"])["sub_tasks"],
                "planning_reasoning": f"Fallback due to error: {e}",
                "error": str(e),
                "current_agent": "planner"
            }
    
    def _create_fallback_tasks(self, query: str) -> dict:
        """Create simple fallback tasks when parsing fails."""
        return {
            "original_query": query,
            "sub_tasks": [
                {
                    "id": 1,
                    "task": query,
                    "priority": "high",
                    "keywords": query.lower().split()[:5]
                }
            ],
            "reasoning": "Fallback: using original query as single task"
        }


def planner_agent(state: AgentState) -> AgentState:
    """Functional interface for LangGraph."""
    agent = PlannerAgent()
    return agent(state)
```

---

#### NRAG-019: Researcher Agent Implementation

**Priority:** P0 - Critical  
**Token Estimate:** 14,000 tokens  
**Status:** To Do

**Description:**  
Implement the Researcher agent that retrieves relevant documents.

**Acceptance Criteria:**

- [x] Process each sub-task from Planner
- [x] Execute retrieval with reranking
- [x] Deduplicate across sub-tasks
- [x] Capture retrieval metadata
- [x] Handle empty results gracefully

**Implementation:**

```python
# src/agents/researcher.py
import logging
from typing import List
from langchain_core.documents import Document

from src.orchestrator.state import AgentState
from src.rag.retriever import RAGRetriever
from src.utils.tracing import agent_trace
from src.utils.helpers import deduplicate_documents

logger = logging.getLogger(__name__)

class ResearcherAgent:
    """
    Retrieves relevant documents for each sub-task.
    
    Uses Cohere embeddings + reranking for high-quality retrieval.
    """
    
    def __init__(self, use_rerank: bool = True):
        self.retriever = RAGRetriever(use_rerank=use_rerank)
    
    @agent_trace("researcher", model="command-r-08-2024")
    def __call__(self, state: AgentState) -> AgentState:
        """
        Retrieve documents for all sub-tasks.
        
        Args:
            state: Current state with sub_tasks
        
        Returns:
            Updated state with retrieved_docs
        """
        sub_tasks = state.get("sub_tasks", [])
        
        if not sub_tasks:
            logger.warning("No sub-tasks provided to researcher")
            return {
                **state,
                "retrieved_docs": [],
                "retrieval_metadata": {"error": "No sub-tasks"},
                "current_agent": "researcher"
            }
        
        logger.info(f"Researcher processing {len(sub_tasks)} sub-tasks")
        
        all_docs = []
        task_results = {}
        
        for task in sub_tasks:
            task_id = task.get("id", 0)
            task_query = task.get("task", "")
            keywords = task.get("keywords", [])
            
            # Combine task with keywords for better retrieval
            search_query = f"{task_query} {' '.join(keywords)}"
            
            logger.debug(f"Searching for task {task_id}: {search_query[:100]}...")
            
            try:
                docs = self.retriever.retrieve(search_query)
                all_docs.extend(docs)
                task_results[task_id] = {
                    "query": search_query,
                    "docs_found": len(docs)
                }
            except Exception as e:
                logger.error(f"Retrieval error for task {task_id}: {e}")
                task_results[task_id] = {
                    "query": search_query,
                    "docs_found": 0,
                    "error": str(e)
                }
        
        # Deduplicate documents
        unique_docs = deduplicate_documents(all_docs)
        
        logger.info(f"Retrieved {len(unique_docs)} unique documents (from {len(all_docs)} total)")
        
        return {
            **state,
            "retrieved_docs": unique_docs,
            "retrieval_metadata": {
                "total_retrieved": len(all_docs),
                "unique_docs": len(unique_docs),
                "task_results": task_results
            },
            "current_agent": "researcher"
        }


def researcher_agent(state: AgentState) -> AgentState:
    """Functional interface for LangGraph."""
    agent = ResearcherAgent()
    return agent(state)
```

---

### Session 4 Summary

| Item | Token Estimate | Cumulative |
|------|----------------|------------|
| NRAG-016: State Schema | 5,000 | 5,000 |
| NRAG-017: LLM Factory | 6,000 | 11,000 |
| NRAG-018: Planner Agent | 12,000 | 23,000 |
| NRAG-019: Researcher Agent | 14,000 | 37,000 |
| **Session Buffer** | ~43,000 | 80,000 |

**Buffer Use:** Testing agents, prompt iteration, debugging

---

