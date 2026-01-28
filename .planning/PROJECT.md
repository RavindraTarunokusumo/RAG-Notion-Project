# Notion Agentic RAG System

## What This Is

A multi-agent RAG system that retrieves and synthesizes information from a Notion knowledge base containing research papers, articles, and personal notes. Built with LangChain/LangGraph and Cohere APIs, the system demonstrates advanced agentic architectures through a 4-agent design (Research, Retrieval, Reasoning, Response) coordinated via Agent-to-Agent (A2A) protocol. Serves as both a personal research assistant and a portfolio piece showcasing modern AI engineering patterns.

## Core Value

Demonstrate proper multi-agent orchestration with A2A protocol - agents must communicate effectively, delegate appropriately, and maintain context across handoffs while delivering coherent research synthesis.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] 4-agent architecture with A2A protocol (Research, Retrieval, Reasoning, Response)
- [ ] Notion API integration to access structured knowledge base
- [ ] ArxivLoader integration for research paper content retrieval
- [ ] Vector search over Notion knowledge base and paper content
- [ ] Cohere API integration with appropriate model selection (Command Light for lightweight agents, Command for heavy agents)
- [ ] LangSmith observability for all agent interactions
- [ ] Citation and source attribution in responses
- [ ] Support for Notion's structured properties (Topics, Type, URL, Date Added)
- [ ] Focus on "Research Paper" type entries initially
- [ ] Cost tracking and performance monitoring
- [ ] Comprehensive testing (unit, integration, end-to-end)
- [ ] Production deployment setup with health checks and error handling
- [ ] Technical documentation covering architecture, APIs, and operational procedures

### Out of Scope

- Google Scholar metadata enrichment — deferred to v2 (use ArxivLoader for v1)
- Multi-language support — v2 enhancement
- Fine-tuning custom Cohere models — v2 enhancement
- Multi-turn conversation memory — v2 enhancement
- Integration with additional academic sources (PubMed, SSRN) — v2 enhancement
- Citation network visualization — v2 enhancement
- Automatic literature review generation — v2 enhancement
- Multi-modal paper analysis (figures, tables) — v2 enhancement

## Context

**Notion Knowledge Base Structure:**
- Entries organized by Topics (AI Infra, AI Safety, Agent Development, etc.)
- Entry properties: Title, Topics, URL, Type, Date Added, Status, Notes, AI Insights, Related Resources
- URLs point to various sources: arXiv papers, articles, GitHub repos, X posts
- Initial focus on Type="Research Paper" entries
- Notes, AI Insights, and Related Resources fields currently empty (opportunities for agent-generated content)
- Status field is UI-only, not used by RAG agents

**Technical Environment:**
- Python-based project using LangChain and LangGraph for agent framework
- Cohere APIs for LLM capabilities (Command and Command Light models)
- LangSmith for observability and debugging
- Notion API access already configured
- Vector database TBD during implementation (Chroma, Pinecone, or Weaviate)

**Learning Objectives:**
- Master multi-agent system design with proper delegation and communication patterns
- Understand RAG pipeline optimization (embedding, retrieval, context injection)
- Gain production experience with observability, cost tracking, and performance monitoring
- Build portfolio-worthy demonstration of modern AI engineering practices

## Constraints

- **Model Costs**: Optimize model selection (Command Light for simple tasks, Command for complex reasoning) to keep cost per query under $0.10
- **API Rate Limits**: Notion API and Cohere API have rate limits - implement intelligent caching and backoff strategies
- **Notion Access**: Must work within Notion's API capabilities and database schema constraints
- **Paper Sources**: v1 limited to arXiv papers accessible via ArxivLoader; other sources deferred to v2
- **Development Timeline**: Dedicated work sessions with systematic phase progression - need clear resumption points
- **Portfolio Quality**: Code quality, documentation, and architecture must demonstrate professional engineering standards

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 4-agent architecture (Research, Retrieval, Reasoning, Response) | Core learning objective - demonstrates A2A protocol and proper separation of concerns | — Pending |
| ArxivLoader for v1, defer Google Scholar to v2 | Reduces complexity while still enabling paper retrieval; metadata enrichment is enhancement not requirement | — Pending |
| Cohere APIs (Command Light + Command) | Backlog specifies Cohere; two-tier approach optimizes cost vs capability tradeoff | — Pending |
| LangSmith for observability | Industry-standard tool for agent debugging and performance monitoring | — Pending |
| Focus on Research Paper type initially | Largest value, clearest use case; other content types can be added incrementally | — Pending |

---
*Last updated: 2026-01-28 after initialization*
