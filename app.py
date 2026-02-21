"""
Streamlit Chat Interface for Notion Agentic RAG

Launch with: streamlit run app.py
"""

import logging
import re
import time
from datetime import datetime

import streamlit as st

from config.settings import settings
from src.orchestrator.graph import create_rag_graph
from src.orchestrator.state import build_initial_state
from src.utils.debugging import configure_logging, debug_run, merge_state
from src.utils.session_manager import get_session_manager
from src.utils.tracing import initialize_tracing

configure_logging(app_name="app")
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Notion Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .stChatMessage {
        padding: 1rem;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
    .session-card {
        background-color: #ffffff;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .citation {
        color: #0066cc;
        font-weight: 500;
        text-decoration: none;
        cursor: pointer;
    }
    .citation:hover {
        text-decoration: underline;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    if "session_name" not in st.session_state:
        st.session_state.session_name = None

    if "streaming_enabled" not in st.session_state:
        st.session_state.streaming_enabled = True

    if "graph" not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            try:
                initialize_tracing()
                st.session_state.graph = create_rag_graph()
                st.session_state.graph_initialized = True
                logger.info("Graph initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize graph: {e}")
                st.session_state.graph_initialized = False
                st.session_state.init_error = str(e)

    if "session_manager" not in st.session_state:
        st.session_state.session_manager = get_session_manager()


def enhance_citations(text: str) -> str:
    """
    Enhance citation markers in text with clickable styling.
    Converts [1], [2], etc. into styled, anchor-linked citations.
    """
    # Find all citation patterns like [1], [2], etc.
    pattern = r'\[(\d+)\]'
    
    def replacer(match):
        num = match.group(1)
        # Create an HTML anchor link
        return f'<a href="#source-{num}" class="citation">[{num}]</a>'
    
    return re.sub(pattern, replacer, text)


def display_source_card(source: dict, index: int):
    """Display a rich source card with enhanced metadata."""
    title = source.get("title", "Untitled")
    source_type = source.get("source", "unknown")
    url = (
        source.get("url")
        or source.get("arxiv_url")
        or source.get("notion_url")
    )
    
    # Source type badge with color
    badge_map = {
        "arxiv": ("🔬 ArXiv", "#FFE5E5"),
        "notion": ("📝 Notion", "#E8F4FF"),
        "pdf": ("📄 PDF", "#F0F0F0"),
        "web": ("🌐 Web", "#E8FFE8")
    }
    badge_text, badge_color = badge_map.get(source_type.lower(), ("📄 Document", "#F5F5F5"))

    # Container with styled card and anchor
    with st.container():
        # Add anchor ID for citation linking
        st.markdown(f'<div id="source-{index+1}"></div>', unsafe_allow_html=True)
        
        # Citation number and title
        citation_col, content_col = st.columns([1, 20])
        
        with citation_col:
            st.markdown(f"### [{index+1}]")
        
        with content_col:
            # Title with link
            if url:
                st.markdown(f"**[{title}]({url})**")
            else:
                st.markdown(f"**{title}**")
            
            # Metadata row
            metadata_parts = []
            
            # Authors (for papers)
            if (authors := source.get("authors")) and isinstance(authors, list):
                author_text = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_text += " et al."
                metadata_parts.append(f"👤 {author_text}")
            
            # Publication date
            if pub_date := source.get("published"):
                # Clean up date formatting
                if isinstance(pub_date, str):
                    # Handle datetime strings
                    try:
                        from datetime import datetime
                        if "T" in pub_date:
                            pub_date = pub_date.split("T")[0]
                        # Try to parse and format nicely
                        dt = datetime.fromisoformat(pub_date.replace("Z", ""))
                        pub_date = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        pass
                metadata_parts.append(f"📅 {pub_date}")
            
            # Topic (for Notion)
            if topic := source.get("topic"):
                metadata_parts.append(f"🏷️ {topic}")
            
            # Category
            if category := source.get("category"):
                metadata_parts.append(f"📑 {category}")
            
            # Source type badge
            st.markdown(
                f'<span style="background-color: {badge_color}; padding: 2px 8px; '
                f'border-radius: 4px; font-size: 0.85em;">{badge_text}</span>',
                unsafe_allow_html=True
            )
            
            # Display metadata
            if metadata_parts:
                st.caption(" • ".join(metadata_parts))
            
            # Snippet/Abstract
            snippet = source.get("snippet") or source.get("abstract")
            if snippet:
                # Clean up snippet
                snippet = snippet.strip()
                # Truncate if too long
                if len(snippet) > 300:
                    snippet = snippet[:297] + "..."
                # Remove extra whitespace
                snippet = " ".join(snippet.split())
                
                with st.expander("📝 Preview", expanded=False):
                    st.caption(snippet)
            
            # ArXiv ID (if available)
            if arxiv_id := source.get("arxiv_id"):
                st.caption(f"🔗 ArXiv ID: `{arxiv_id}`")
            
            # Keywords (for Notion)
            if (keywords := source.get("keywords")) and isinstance(keywords, list) and keywords:
                keyword_text = ", ".join(keywords[:5])
                if len(keywords) > 5:
                    keyword_text += f" (+{len(keywords)-5} more)"
                st.caption(f"🔖 Keywords: {keyword_text}")


def render_workflow_progress(current_agent: str, completed_agents: list, agent_outcomes: dict):
    """
    Render a visual progress tracking sidebar.
    Should be called within a st.sidebar context or container.
    """
    st.markdown("### 🚦 Workflow Progress")
    
    agents = [
        {"id": "planner", "label": "Planning", "icon": "🎯"},
        {"id": "researcher", "label": "Researching", "icon": "🔍"},
        {"id": "reasoner", "label": "Reasoning", "icon": "🧠"},
        {"id": "synthesiser", "label": "Synthesising", "icon": "✍️"}
    ]
    
    for agent in agents:
        agent_id = agent["id"]
        is_completed = agent_id in completed_agents
        is_current = agent_id == current_agent
        
        # Determine visuals
        if is_completed:
            status_icon = "✅"
            font_weight = "normal"
            opacity = "1.0"
            color = "green"
        elif is_current:
            status_icon = "🔄"
            font_weight = "bold"
            opacity = "1.0"
            color = "blue"
        else:
            status_icon = "⚪"
            font_weight = "normal"
            opacity = "0.5"
            color = "gray"
            
        # Outcome text (if available)
        outcome = agent_outcomes.get(agent_id, "")
        
        # Render row
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 8px; opacity: {opacity}">
                <div style="font-size: 1.2em; margin-right: 8px;">{status_icon}</div>
                <div style="flex-grow: 1;">
                    <div style="font-weight: {font_weight}; color: {color}">
                        {agent['icon']} {agent['label']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if outcome:
             st.caption(f"└ {outcome}")
             
    if "synthesiser" in completed_agents:
        st.success("✨ Process Complete!")


def _get_agent_emoji(agent: str) -> str:
    emojis = {
        "planner": "🎯",
        "researcher": "🔍",
        "reasoner": "🧠",
        "synthesiser": "✍️",
        "end": "🏁"
    }
    return emojis.get(agent, "🤖")


def process_query_streaming(prompt: str):
    """
    Process a user query through the RAG pipeline with streaming.

    Yields status updates and final result.
    """
    initial_state = build_initial_state(prompt)

    # Track progress locally
    completed_agents = []
    agent_outcomes = {}

    # 1. Start with Planner
    yield {
        "type": "progress",
        "current_agent": "planner",
        "completed_agents": [],
        "agent_outcomes": {},
        "message": "Planner is analyzing your query...",
    }

    try:
        final_result = None
        accumulated_state = dict(initial_state)

        with debug_run(
            query=prompt,
            initial_state=initial_state,
            mode="stream",
        ) as trace_session:
            # Use stream mode to get intermediate updates
            for event in st.session_state.graph.stream(initial_state):
                # Event is a dict with node name as key
                for node_name, node_output in event.items():
                    accumulated_state = merge_state(accumulated_state, node_output)

                    # Mark current as completed
                    if node_name not in completed_agents:
                        completed_agents.append(node_name)

                    # Extract interesting stats/outcomes
                    outcome = ""
                    if node_name == "planner":
                        sub_tasks = node_output.get("sub_tasks", [])
                        outcome = f"Created {len(sub_tasks)} sub-tasks"
                    elif node_name == "researcher":
                        docs = node_output.get("retrieved_docs", [])
                        outcome = f"Found {len(docs)} documents"
                    elif node_name == "reasoner":
                        analysis = node_output.get("analysis", [])
                        outcome = f"Analyzed {len(analysis)} items"
                    elif node_name == "synthesiser":
                        outcome = "Drafted response"

                    agent_outcomes[node_name] = outcome

                    # Determine next agent
                    next_agent = "end"
                    if node_name == "planner":
                        next_agent = "researcher"
                    elif node_name == "researcher":
                        next_agent = "reasoner"
                    elif node_name == "reasoner":
                        next_agent = "synthesiser"

                    # Yield progress update
                    yield {
                        "type": "progress",
                        "current_agent": next_agent,
                        "completed_agents": completed_agents.copy(),
                        "agent_outcomes": agent_outcomes.copy(),
                        "message": (
                            f"{_get_agent_emoji(next_agent)} "
                            f"{next_agent.title()} running..."
                            if next_agent != "end"
                            else "Finalizing..."
                        ),
                    }

                    # Store merged final state
                    final_result = accumulated_state

            if trace_session is not None:
                trace_session.record_run_end(final_result or accumulated_state)

        # Check for errors in final result
        if final_result and final_result.get("error"):
            yield {
                "type": "error",
                "message": f"Pipeline Error: {final_result['error']}",
                "sources": [],
            }
        elif final_result:
            # Return the final answer
            yield {
                "type": "complete",
                "message": final_result.get("final_answer", ""),
                "sources": final_result.get("sources", []),
            }
        else:
            yield {
                "type": "error",
                "message": "No response from pipeline",
                "sources": [],
            }

    except Exception as e:
        logger.exception("Error processing query")
        yield {
            "type": "error",
            "message": f"System Error: {str(e)}",
            "sources": [],
        }


def process_query(prompt: str):
    """Process a user query through the RAG pipeline (non-streaming)."""
    initial_state = build_initial_state(prompt)

    try:
        # Execute graph
        with debug_run(
            query=prompt,
            initial_state=initial_state,
            mode="invoke",
        ) as trace_session:
            result = st.session_state.graph.invoke(initial_state)
            if trace_session is not None:
                trace_session.record_run_end(result)

        if result.get("error"):
            return {
                "success": False,
                "message": f"Pipeline Error: {result['error']}",
                "sources": [],
            }

        return {
            "success": True,
            "message": result["final_answer"],
            "sources": result.get("sources", []),
        }

    except Exception as e:
        logger.exception("Error processing query")
        return {
            "success": False,
            "message": f"System Error: {str(e)}",
            "sources": [],
        }

def main():
    """Main application."""
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("🤖 Notion RAG")
        st.markdown("**Multi-Agent Research Assistant**")
        st.divider()

        # System status
        st.markdown("### System Status")
        if st.session_state.get("graph_initialized"):
            st.success("✓ Graph Initialized")
            st.info(f"📊 Project: {settings.langsmith_project}")
        else:
            st.error("✗ Initialization Failed")
            if error := st.session_state.get("init_error"):
                st.error(error)
                st.stop()

        st.divider()

        # Session Management
        st.markdown("### 💾 Session Management")

        session_mgr = st.session_state.session_manager

        # Current session info
        if st.session_state.current_session_id:
            st.info(
                f"📝 Current: {st.session_state.session_name or 'Unnamed Session'}"
            )
        else:
            st.caption("No session loaded")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ New", use_container_width=True):
                st.session_state.messages = []
                st.session_state.current_session_id = None
                st.session_state.session_name = None
                st.rerun()

        with col2:
            if st.button(
                "💾 Save",
                use_container_width=True,
                disabled=len(st.session_state.messages) == 0,
            ):
                if st.session_state.current_session_id is None:
                    # New session
                    session_id = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    st.session_state.current_session_id = session_id
                else:
                    session_id = st.session_state.current_session_id

                session_data = session_mgr.save_session(
                    session_id,
                    st.session_state.messages,
                    st.session_state.session_name,
                )
                
                # Update session name in state if it generated one
                if session_data.get("name"):
                    st.session_state.session_name = session_data["name"]

                st.success("✓ Session saved!")
                st.rerun()

        # Load saved sessions
        sessions = session_mgr.list_sessions()
        if sessions:
            st.markdown(f"**Saved Sessions ({len(sessions)})**")

            # Show last 5 sessions
            for session in sessions[:5]:
                session_col1, session_col2 = st.columns([3, 1])

                with session_col1:
                    if st.button(
                        f"📄 {session['name']}",
                        key=f"load_{session['id']}",
                        use_container_width=True,
                    ):
                        # Load session
                        session_data = session_mgr.load_session(
                            session["id"]
                        )
                        if session_data:
                            st.session_state.messages = session_data[
                                "messages"
                            ]
                            st.session_state.current_session_id = session[
                                "id"
                            ]
                            st.session_state.session_name = session["name"]
                            st.rerun()

                with session_col2:
                    if st.button(
                        "🗑️",
                        key=f"delete_{session['id']}",
                        use_container_width=True,
                    ):
                        session_mgr.delete_session(session["id"])
                        if (
                            st.session_state.current_session_id
                            == session["id"]
                        ):
                            st.session_state.messages = []
                            st.session_state.current_session_id = None
                            st.session_state.session_name = None
                        st.rerun()

                st.caption(
                    f"💬 {session['message_count']} msgs • {session['updated_at'][:10]}"
                )

            if len(sessions) > 5:
                st.caption(f"...and {len(sessions) - 5} more")

        # Export options
        if st.session_state.current_session_id:
            with st.expander("📤 Export"):
                export_col1, export_col2 = st.columns(2)

                with export_col1:
                    if st.button("JSON", use_container_width=True):
                        json_str = session_mgr.export_session_json(
                            st.session_state.current_session_id
                        )
                        if json_str:
                            st.download_button(
                                label="⬇️ Download JSON",
                                data=json_str,
                                file_name=f"session_{st.session_state.current_session_id}.json",
                                mime="application/json",
                                use_container_width=True,
                            )

                with export_col2:
                    if st.button("Markdown", use_container_width=True):
                        md_str = session_mgr.export_session_markdown(
                            st.session_state.current_session_id
                        )
                        if md_str:
                            st.download_button(
                                label="⬇️ Download MD",
                                data=md_str,
                                file_name=f"session_{st.session_state.current_session_id}.md",
                                mime="text/markdown",
                                use_container_width=True,
                            )

        st.divider()

        # Agent pipeline info
        with st.expander("🔄 Agent Pipeline"):
            st.markdown(
                """
            1. **🎯 Planner**: Breaks down your query
            2. **🔍 Researcher**: Finds relevant documents
            3. **🧠 Reasoner**: Analyzes findings
            4. **✍️ Synthesiser**: Writes the answer
            """
            )

        st.divider()

        # Settings
        with st.expander("⚙️ Settings"):
            st.markdown("#### Models")
            
            # Planner model
            model_options = [
                "qwen-flash-2025-07-28",
                "gpt-4o-mini",
            ]
            current_planner = settings.models.planner.model
            planner_index = (
                model_options.index(current_planner)
                if current_planner in model_options
                else 0
            )
            
            planner_model = st.selectbox(
                "Planner Model",
                model_options,
                index=planner_index,
                help="Model used for breaking down tasks"
            )

            st.markdown("#### Retrieval")
            
            # Retrieval K
            retrieval_k = st.slider(
                "Documents to Retrieve", 
                5, 20, 
                value=settings.retrieval_k
            )
            
            # Rerank Top N
            rerank_top_n = st.slider(
                "Rerank Top N", 
                3, 10, 
                value=settings.rerank_top_n
            )

            st.markdown("#### Display")
            
            # Streaming (Existing)
            streaming = st.checkbox(
                "Enable Streaming",
                value=st.session_state.streaming_enabled,
                help="Stream responses character by character with real-time agent status",
            )
            
            # Start fresh session handling for settings in session state if needed
            if "show_intermediate" not in st.session_state:
                st.session_state.show_intermediate = False
            
            if "verbose_mode" not in st.session_state:
                st.session_state.verbose_mode = False

            show_intermediate = st.checkbox(
                "Show Intermediate Results", 
                value=st.session_state.show_intermediate,
                help="Display intermediate agent outputs like search queries and reasoning"
            )
            
            verbose_mode = st.checkbox(
                "Verbose Logging", 
                value=st.session_state.verbose_mode
            )
            
            # Application Button
            if st.button("Apply Settings", use_container_width=True):
                # Update Session State
                st.session_state.streaming_enabled = streaming
                st.session_state.show_intermediate = show_intermediate
                st.session_state.verbose_mode = verbose_mode
                
                # Update Global Settings
                settings.models.planner.model = planner_model
                settings.models.planner.provider = (
                    "qwen"
                    if planner_model.startswith("qwen")
                    else "openai"
                )
                settings.retrieval_k = retrieval_k
                settings.rerank_top_n = rerank_top_n
                settings.debug.log_level = (
                    "DEBUG" if verbose_mode else "INFO"
                )
                logging.getLogger().setLevel(
                    logging.DEBUG if verbose_mode else logging.INFO
                )
                
                # Force graph re-init if model changed (though simple update is fine for this architecture)
                # But retriever uses settings at runtime, so it's fine.
                # Planner uses get_agent_llm at runtime? 
                # Let's check planner.py. Node calls get_agent_llm inside?
                # Actually planner_node calls get_agent_llm inside the function => dynamic => good.
                
                st.success("✓ Settings updated!")
                
                # Rerun to reflect changes immediately
                time.sleep(0.5) 
                st.rerun()

        st.divider()

        # Chat controls
        st.markdown("### Chat Controls")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Stats
        if st.session_state.messages:
            msg_count = len(st.session_state.messages)
            st.caption(f"💬 {msg_count} message(s) in history")

    # Main chat interface
    st.title("Research Assistant")
    st.markdown(
        "Ask questions about your knowledge base and get cited answers from your Notion database and arXiv papers."
    )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Enhance citations in the message content
            content = message["content"]
            if message.get("sources"):
                # Only enhance citations if there are sources
                content_html = enhance_citations(content)
                st.markdown(content_html, unsafe_allow_html=True)
            else:
                st.markdown(content)

            # Display sources if available
            if message.get("sources"):
                sources = message["sources"]
                
                with st.expander(
                    f"📚 Sources ({len(sources)})", expanded=False
                ):
                    # Source filtering options
                    filter_col1, filter_col2 = st.columns([1, 1])
                    
                    with filter_col1:
                        # Get unique source types
                        source_types = list({s.get("source", "unknown").lower() for s in sources})
                        source_types.insert(0, "All")
                        
                        selected_type = st.selectbox(
                            "Filter by type:",
                            source_types,
                            key=f"filter_type_{message.get('timestamp', id(message))}"
                        )
                    
                    with filter_col2:
                        # Search within sources
                        search_term = st.text_input(
                            "Search sources:",
                            key=f"search_{message.get('timestamp', id(message))}"
                        )
                    
                    st.divider()
                    
                    # Filter sources
                    filtered_sources = sources
                    if selected_type != "All":
                        filtered_sources = [
                            s for s in filtered_sources 
                            if s.get("source", "").lower() == selected_type
                        ]
                    
                    if search_term:
                        search_lower = search_term.lower()
                        filtered_sources = [
                            s for s in filtered_sources
                            if search_lower in s.get("title", "").lower()
                            or search_lower in str(s.get("keywords", "")).lower()
                            or search_lower in str(s.get("topic", "")).lower()
                        ]
                    
                    # Display filtered sources
                    if filtered_sources:
                        for i, source in enumerate(filtered_sources):
                            # Find original index for citation numbering
                            original_idx = sources.index(source)
                            display_source_card(source, original_idx)
                            if i < len(filtered_sources) - 1:
                                st.divider()
                    else:
                        st.info("No sources match your filter criteria.")

    # Chat input
    if prompt := st.chat_input(
        "Ask a question about your knowledge base..."
    ):
        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": []}
        )

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process and display assistant response
        with st.chat_message("assistant"):
            if st.session_state.streaming_enabled:
                # Streaming mode with real-time updates
                status_container = st.container()
                response_placeholder = st.empty()
                sources_container = st.container()
                
                # Sidebar placeholder for progress
                progress_placeholder = st.sidebar.empty()

                final_message = ""
                final_sources = []

                try:
                    for update in process_query_streaming(prompt):
                        if update["type"] == "progress":
                            # Update sidebar
                            with progress_placeholder.container():
                                render_workflow_progress(
                                    update["current_agent"], 
                                    update["completed_agents"],
                                    update["agent_outcomes"]
                                )
                            
                            # Update status message
                            with status_container:
                                st.info(update["message"])

                        elif update["type"] == "complete":
                            # Clear status and show final answer
                            status_container.empty()
                            # Do not clear sidebar progress so user can see what happened
                            
                            final_message = update["message"]
                            final_sources = update["sources"]

                            # Display response with streaming effect
                            response_text = ""
                            for i, char in enumerate(final_message):
                                response_text += char
                                # Update every 3 characters for smoother streaming
                                if i % 3 == 0 or i == len(final_message) - 1:
                                    response_placeholder.markdown(
                                        response_text + "▌"
                                    )

                            # Remove cursor and show final text
                            response_placeholder.markdown(final_message)

                        elif update["type"] == "error":
                            status_container.empty()
                            final_message = update["message"]
                            final_sources = []
                            response_placeholder.error(final_message)

                    # Display sources
                    if final_sources:
                        with sources_container, st.expander(
                            f"📚 Sources ({len(final_sources)})", expanded=True
                        ):
                            # Source filtering
                            filter_col1, filter_col2 = st.columns([1, 1])
                            
                            with filter_col1:
                                source_types = list({s.get("source", "unknown").lower() for s in final_sources})
                                source_types.insert(0, "All")
                                selected_type = st.selectbox(
                                    "Filter by type:",
                                    source_types,
                                    key="stream_filter_type"
                                )
                            
                            with filter_col2:
                                search_term = st.text_input(
                                    "Search sources:",
                                    key="stream_search"
                                )
                            
                            st.divider()
                                
                            # Apply filters
                            filtered = final_sources
                            if selected_type != "All":
                                filtered = [s for s in filtered if s.get("source", "").lower() == selected_type]
                            if search_term:
                                search_lower = search_term.lower()
                                filtered = [
                                    s for s in filtered
                                    if search_lower in s.get("title", "").lower()
                                    or search_lower in str(s.get("keywords", "")).lower()
                                    or search_lower in str(s.get("topic", "")).lower()
                                ]
                            
                            if filtered:
                                for i, source in enumerate(filtered):
                                    original_idx = final_sources.index(source)
                                    display_source_card(source, original_idx)
                                    if i < len(filtered) - 1:
                                        st.divider()
                            else:
                                st.info("No sources match your filter criteria.")

                except Exception as e:
                    logger.exception("Error during streaming")
                    status_container.empty()
                    final_message = f"❌ Streaming Error: {str(e)}"
                    final_sources = []
                    response_placeholder.error(final_message)

            else:
                # Non-streaming mode (original behavior)
                status_placeholder = st.empty()
                response_placeholder = st.empty()

                status_placeholder.info(
                    "🔍 Processing your query through the agent pipeline..."
                )

                # Process query
                result = process_query(prompt)

                # Clear status
                status_placeholder.empty()

                # Display response
                if result["success"]:
                    final_message = result["message"]
                    final_sources = result["sources"]
                    response_placeholder.markdown(final_message)
                else:
                    final_message = result["message"]
                    final_sources = []
                    response_placeholder.error(final_message)

                # Display sources
                if final_sources:
                    with st.expander(
                        f"📚 Sources ({len(final_sources)})", expanded=True
                    ):
                        # Source filtering
                        filter_col1, filter_col2 = st.columns([1, 1])
                        
                        with filter_col1:
                            source_types = list({s.get("source", "unknown").lower() for s in final_sources})
                            source_types.insert(0, "All")
                            selected_type = st.selectbox(
                                "Filter by type:",
                                source_types,
                                key="nonstream_filter_type"
                            )
                        
                        with filter_col2:
                            search_term = st.text_input(
                                "Search sources:",
                                key="nonstream_search"
                            )
                        
                        st.divider()
                        
                        # Apply filters
                        filtered = final_sources
                        if selected_type != "All":
                            filtered = [s for s in filtered if s.get("source", "").lower() == selected_type]
                        if search_term:
                            search_lower = search_term.lower()
                            filtered = [
                                s for s in filtered
                                if search_lower in s.get("title", "").lower()
                                or search_lower in str(s.get("keywords", "")).lower()
                                or search_lower in str(s.get("topic", "")).lower()
                            ]
                        
                        if filtered:
                            for i, source in enumerate(filtered):
                                original_idx = final_sources.index(source)
                                display_source_card(source, original_idx)
                                if i < len(filtered) - 1:
                                    st.divider()
                        else:
                            st.info("No sources match your filter criteria.")

            # Add to message history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_message,
                    "sources": final_sources,
                }
            )


if __name__ == "__main__":
    main()
