## Session 5: Reasoner & Synthesiser Agents

**Session Token Budget:** ~80,000 tokens  
**Focus:** Implement the analysis and synthesis agents

---

#### NRAG-020: Reasoner Agent Implementation

**Priority:** P0 - Critical  
**Token Estimate:** 16,000 tokens  
**Status:** To Do

**Description:**  
Implement the Reasoner agent that applies logical analysis to retrieved documents.

**Acceptance Criteria:**

- [x] Analyze documents against each sub-task
- [x] Identify key findings and evidence
- [x] Detect contradictions and gaps
- [x] Assign confidence scores
- [x] Use larger model (Command R+) for complex reasoning

**Implementation:**

```python
# src/agents/reasoner.py
import logging
from langchain_core.prompts import ChatPromptTemplate

from src.orchestrator.state import AgentState, Analysis
from src.agents.llm_factory import get_agent_llm
from src.utils.tracing import agent_trace
from src.utils.helpers import safe_json_parse, format_documents_for_prompt

logger = logging.getLogger(__name__)

REASONER_SYSTEM_PROMPT = """You are a Reasoning Agent specialized in logical analysis and critical evaluation.

Your role is to analyze retrieved documents against the research sub-tasks and provide structured analysis.

For each sub-task, you must:
1. Identify key findings relevant to the task
2. Extract supporting evidence with specific citations
3. Note any contradictions or conflicting information
4. Assess your confidence in answering the task (0.0 to 1.0)
5. Identify gaps where information is missing or insufficient

Guidelines:
- Be thorough but concise
- Support findings with specific document references
- Be honest about uncertainty
- Consider multiple perspectives when documents conflict

Output your analysis as a JSON object:
{{
    "analyses": [
        {{
            "sub_task_id": 1,
            "key_findings": ["Finding 1", "Finding 2"],
            "supporting_evidence": ["Document X states...", "According to Document Y..."],
            "contradictions": ["Document A says X while Document B says Y"],
            "confidence": 0.85,
            "gaps": ["No information found about..."]
        }}
    ],
    "overall_assessment": "Summary of the analysis quality and completeness",
    "synthesis_recommendations": "Suggestions for how to synthesize the final answer"
}}

Important: Output ONLY the JSON object."""

REASONER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REASONER_SYSTEM_PROMPT),
    ("human", """Original Query: {query}

Sub-tasks to analyze:
{sub_tasks}

Retrieved Documents:
{documents}

Please analyze the documents against each sub-task.""")
])

class ReasonerAgent:
    """
    Applies logical analysis to retrieved documents.
    
    Uses Command R+ (larger model) for nuanced reasoning and analysis.
    """
    
    def __init__(self):
        self.llm = get_agent_llm("reasoner")
        self.chain = REASONER_PROMPT | self.llm
    
    @agent_trace("reasoner", model="command-r-plus-08-2024")
    def __call__(self, state: AgentState) -> AgentState:
        """
        Analyze retrieved documents.
        
        Args:
            state: Current state with sub_tasks and retrieved_docs
        
        Returns:
            Updated state with analysis
        """
        docs = state.get("retrieved_docs", [])
        sub_tasks = state.get("sub_tasks", [])
        
        if not docs:
            logger.warning("No documents to analyze")
            return {
                **state,
                "analysis": [],
                "overall_assessment": "No documents available for analysis",
                "current_agent": "reasoner"
            }
        
        logger.info(f"Reasoner analyzing {len(docs)} documents for {len(sub_tasks)} tasks")
        
        try:
            # Format inputs
            docs_text = format_documents_for_prompt(docs, max_chars=12000)
            tasks_text = self._format_tasks(sub_tasks)
            
            response = self.chain.invoke({
                "query": state["query"],
                "sub_tasks": tasks_text,
                "documents": docs_text
            })
            
            parsed = safe_json_parse(response.content)
            
            if not parsed:
                logger.warning("Failed to parse reasoner output")
                parsed = self._create_fallback_analysis(sub_tasks)
            
            analyses = parsed.get("analyses", [])
            overall = parsed.get("overall_assessment", "Analysis completed")
            
            logger.info(f"Reasoner completed analysis with {len(analyses)} task analyses")
            
            return {
                **state,
                "analysis": analyses,
                "overall_assessment": overall,
                "current_agent": "reasoner"
            }
            
        except Exception as e:
            logger.error(f"Reasoner error: {e}")
            return {
                **state,
                "analysis": [],
                "overall_assessment": f"Analysis failed: {e}",
                "error": str(e),
                "current_agent": "reasoner"
            }
    
    def _format_tasks(self, tasks: list) -> str:
        """Format sub-tasks for the prompt."""
        lines = []
        for task in tasks:
            lines.append(f"Task {task['id']} [{task['priority']}]: {task['task']}")
        return "\n".join(lines)
    
    def _create_fallback_analysis(self, sub_tasks: list) -> dict:
        """Create fallback analysis when parsing fails."""
        return {
            "analyses": [
                {
                    "sub_task_id": t["id"],
                    "key_findings": ["Analysis parsing failed"],
                    "supporting_evidence": [],
                    "contradictions": [],
                    "confidence": 0.3,
                    "gaps": ["Unable to perform detailed analysis"]
                }
                for t in sub_tasks
            ],
            "overall_assessment": "Fallback analysis due to parsing error"
        }


def reasoner_agent(state: AgentState) -> AgentState:
    """Functional interface for LangGraph."""
    agent = ReasonerAgent()
    return agent(state)
```

---

#### NRAG-021: Synthesiser Agent Implementation

**Priority:** P0 - Critical  
**Token Estimate:** 16,000 tokens  
**Status:** To Do

**Description:**  
Implement the Synthesiser agent that creates the final coherent response.

**Acceptance Criteria:**

- [x] Combine all findings into coherent answer
- [x] Include citations for factual claims
- [x] Acknowledge uncertainty appropriately
- [x] Adapt format to query type
- [x] Include sources section

**Implementation:**

```python
# src/agents/synthesiser.py
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.orchestrator.state import AgentState
from src.agents.llm_factory import get_agent_llm
from src.utils.tracing import agent_trace
from src.utils.helpers import format_documents_for_prompt

logger = logging.getLogger(__name__)

SYNTHESISER_SYSTEM_PROMPT = """You are a Synthesis Agent specialized in creating comprehensive, well-structured answers.

Your role is to combine analysis results and source documents into a coherent, informative response that directly answers the user's question.

Guidelines:
1. Start with a direct answer to the main question
2. Organize information logically (use headers if helpful)
3. Include inline citations [Source N] for factual claims
4. Acknowledge areas of uncertainty or conflicting information
5. Be comprehensive but avoid unnecessary repetition
6. End with a "Sources" section listing referenced documents

Response Formatting:
- For explanatory questions: Structured explanation with examples
- For comparison questions: Clear comparison with key differences
- For how-to questions: Step-by-step guidance
- For exploratory questions: Comprehensive overview with multiple perspectives

Confidence Guidelines:
- High confidence (>0.8): State findings directly
- Medium confidence (0.5-0.8): Use phrases like "evidence suggests" or "likely"
- Low confidence (<0.5): Clearly note limitations and gaps

Remember: Quality over quantity. Be thorough but concise."""

SYNTHESISER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYNTHESISER_SYSTEM_PROMPT),
    ("human", """Original Question: {query}

Analysis Summary:
{analysis_summary}

Key Findings by Task:
{findings}

Source Documents:
{documents}

Please synthesize a comprehensive response to the original question.""")
])

class SynthesiserAgent:
    """
    Creates the final coherent response.
    
    Uses Command R+ (larger model) for high-quality text generation.
    """
    
    def __init__(self):
        self.llm = get_agent_llm("synthesiser")
        self.chain = SYNTHESISER_PROMPT | self.llm | StrOutputParser()
    
    @agent_trace("synthesiser", model="command-r-plus-08-2024")
    def __call__(self, state: AgentState) -> AgentState:
        """
        Generate final response.
        
        Args:
            state: Current state with all prior agent outputs
        
        Returns:
            Updated state with final_answer
        """
        analysis = state.get("analysis", [])
        docs = state.get("retrieved_docs", [])
        overall = state.get("overall_assessment", "")
        
        logger.info("Synthesiser generating final response")
        
        try:
            # Format analysis for prompt
            analysis_summary = overall
            findings = self._format_findings(analysis)
            docs_text = format_documents_for_prompt(docs, max_chars=10000)
            
            response = self.chain.invoke({
                "query": state["query"],
                "analysis_summary": analysis_summary,
                "findings": findings,
                "documents": docs_text
            })
            
            # Extract sources for metadata
            sources = self._extract_sources(docs)
            
            logger.info(f"Synthesiser generated response ({len(response)} chars)")
            
            return {
                **state,
                "final_answer": response,
                "sources": sources,
                "current_agent": "synthesiser"
            }
            
        except Exception as e:
            logger.error(f"Synthesiser error: {e}")
            return {
                **state,
                "final_answer": f"I apologize, but I encountered an error while generating the response: {e}",
                "sources": [],
                "error": str(e),
                "current_agent": "synthesiser"
            }
    
    def _format_findings(self, analyses: list) -> str:
        """Format analysis findings for the prompt."""
        if not analyses:
            return "No analysis available."
        
        lines = []
        for analysis in analyses:
            task_id = analysis.get("sub_task_id", "?")
            confidence = analysis.get("confidence", 0)
            findings = analysis.get("key_findings", [])
            gaps = analysis.get("gaps", [])
            
            lines.append(f"\n### Task {task_id} (Confidence: {confidence:.0%})")
            
            if findings:
                lines.append("Findings:")
                for f in findings:
                    lines.append(f"  - {f}")
            
            if gaps:
                lines.append("Gaps:")
                for g in gaps:
                    lines.append(f"  - {g}")
        
        return "\n".join(lines)
    
    def _extract_sources(self, docs: list) -> list:
        """Extract source metadata from documents."""
        sources = []
        seen = set()
        
        for i, doc in enumerate(docs, 1):
            source_id = doc.metadata.get("arxiv_id") or doc.metadata.get("notion_id") or f"doc_{i}"
            
            if source_id in seen:
                continue
            seen.add(source_id)
            
            sources.append({
                "id": i,
                "title": doc.metadata.get("title", "Untitled"),
                "source": doc.metadata.get("source", "unknown"),
                "url": doc.metadata.get("arxiv_url") or doc.metadata.get("source_url", ""),
                "arxiv_id": doc.metadata.get("arxiv_id", "")
            })
        
        return sources


def synthesiser_agent(state: AgentState) -> AgentState:
    """Functional interface for LangGraph."""
    agent = SynthesiserAgent()
    return agent(state)
```

---

#### NRAG-022: Agent Module Exports

**Priority:** P1 - High  
**Token Estimate:** 3,000 tokens  
**Status:** To Do

**Description:**  
Create clean module exports for all agents.

**Implementation:**

```python
# src/agents/__init__.py
"""
Agentic RAG Agent Implementations

Agent Pipeline:
1. Planner (Command R) - Decomposes queries into sub-tasks
2. Researcher (Command R) - Retrieves relevant documents  
3. Reasoner (Command R+) - Analyzes and evaluates findings
4. Synthesiser (Command R+) - Creates final coherent response
"""

from src.agents.planner import PlannerAgent, planner_agent
from src.agents.researcher import ResearcherAgent, researcher_agent
from src.agents.reasoner import ReasonerAgent, reasoner_agent
from src.agents.synthesiser import SynthesiserAgent, synthesiser_agent
from src.agents.llm_factory import get_agent_llm, AGENT_CONFIGS

__all__ = [
    # Classes
    "PlannerAgent",
    "ResearcherAgent", 
    "ReasonerAgent",
    "SynthesiserAgent",
    # Functional interfaces (for LangGraph)
    "planner_agent",
    "researcher_agent",
    "reasoner_agent",
    "synthesiser_agent",
    # Utilities
    "get_agent_llm",
    "AGENT_CONFIGS",
]
```

---

### Session 5 Summary

| Item | Token Estimate | Cumulative |
|------|----------------|------------|
| NRAG-020: Reasoner Agent | 16,000 | 16,000 |
| NRAG-021: Synthesiser Agent | 16,000 | 32,000 |
| NRAG-022: Module Exports | 3,000 | 35,000 |
| **Session Buffer** | ~45,000 | 80,000 |

**Buffer Use:** Prompt refinement, output quality testing, integration

---

