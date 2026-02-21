## Project Vision

Build an Agentic RAG system that retrieves and synthesizes information from a Notion knowledge base using four specialized AI agents (Planner, Researcher, Reasoner, Synthesiser) communicating via the A2A protocol for tool calling, powered by Cohere's Command R/R+ models through LangChain/LangGraph with Agent Lightning integration for continual learning and optimization.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                       │
│                    (LangSmith Tracing)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Planner     │───>│  Researcher   │───>│   Reasoner    │
│ (Command R)   │    │ (Command R)   │    │(Command R+)   │
└───────────────┘    └───────────────┘    └───────────────┘
                              │                     │
                              ▼                     ▼
                     ┌───────────────┐    ┌───────────────┐
                     │  Vector Store │    │  Synthesiser  │
                     │  (ChromaDB)   │    │ (Command R+)  │
                     └───────────────┘    └───────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────┐                          ┌───────────────┐
│ Notion Loader │                          │ Arxiv Loader  │
│ (Metadata &   │─────────────────────────>│ (Full Papers) │
│  Arxiv Links) │                          │               │
└───────────────┘                          └───────────────┘
```

---

### Document Pipeline Flow

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Notion KB       │     │  Extract Arxiv   │     │  ArxivLoader     │
│  (Links + Meta)  │────>│  IDs from Links  │────>│  (Full Papers)   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                                                  │
         │ Metadata:                                        │ Content:
         │ - Title                                          │ - Full Abstract
         │ - Categories                                     │ - Paper Content
         │ - Source URL                                     │ - Authors
         │ - User Tags                                      │ - Publication Date
         │                                                  │
         └──────────────────────┬───────────────────────────┘
                                ▼
                    ┌──────────────────────┐
                    │  Merged Documents    │
                    │  (Content + Meta)    │
                    └──────────────────────┘
```

---

### Tool Calling

```
┌─────────────────────────────────────────────────────────────────┐
│                    Fixed LangGraph Pipeline                      │
│  Planner ──> Researcher ──> Reasoner ──> Synthesiser            │
└─────────────────────────────────────────────────────────────────┘
                    │              │            │
                    ▼              ▼            ▼
         ┌─────────────────────────────────────────────┐
         │     Dynamic A2A Tool Agents (Optional)       │
         │  ┌───────────┐ ┌───────────┐ ┌────────────┐ │
         │  │ Web       │ │ Code      │ │ Citation   │ │
         │  │ Searcher  │ │ Executor  │ │ Validator  │ │
         │  └───────────┘ └───────────┘ └────────────┘ │
         │  ┌───────────┐ ┌───────────┐ ┌────────────┐ │
         │  │ Math      │ │ Diagram   │ │ Fact       │ │
         │  │ Solver    │ │ Generator │ │ Checker    │ │
         │  └───────────┘ └───────────┘ └────────────┘ │
         └─────────────────────────────────────────────┘
                         ▲
                         │
              A2A Agent Cards for Discovery
              (Capabilities, Input/Output Schemas)
```

---

### Agent Lightning Integration

```
┌────────────────────────────────────────────────────────────┐
│             Existing Agentic RAG System                     │
│   Planner → Researcher → Reasoner → Synthesiser            │
│              (LangChain/LangGraph)                          │
└────────────────────────────────────────────────────────────┘
                          │
                          │ emit_span() / emit_reward()
                          ▼
┌────────────────────────────────────────────────────────────┐
│              AgentLightning Layer                          │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │ Lightning Store  │ ←──→ │ Lightning Client  │          │
│  │ (Trajectories)   │      │ (LLM Proxy)       │          │
│  └──────────────────┘      └──────────────────┘           │
│         │                            │                     │
│         │ Trajectory Data            │ Optimized Prompts  │
│         ▼                            ▼                     │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │ APO Engine       │      │ User Feedback     │          │
│  │ (Prompt Rewrite) │      │ & Rewards         │          │
│  └──────────────────┘      └──────────────────┘           │
└────────────────────────────────────────────────────────────┘
```