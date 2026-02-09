"""
Streamlit Chat Interface for Notion Agentic RAG

Launch with: streamlit run app.py
"""

import logging
from datetime import datetime

import streamlit as st

from config.settings import settings
from src.orchestrator.graph import create_rag_graph
from src.utils.session_manager import get_session_manager
from src.utils.tracing import initialize_tracing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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


def display_source_card(source: dict, index: int):
    """Display a rich source card."""
    title = source.get("title", "Untitled")
    source_type = source.get("source", "unknown")
    url = (
        source.get("url")
        or source.get("arxiv_url")
        or source.get("notion_url")
    )

    # Source icon
    icon = {"arxiv": "🔬", "notion": "📝"}.get(source_type, "📄")

    # Build display text
    display_text = f"{icon} **{title}**"
    if url:
        display_text = f"{icon} **[{title}]({url})**"

    st.markdown(display_text)

    # Additional metadata
    metadata_parts = []
    if authors := source.get("authors"):
        if isinstance(authors, list):
            author_text = ", ".join(authors[:3])
        author_text += " et al." if len(authors) > 3 else ""
        metadata_parts.append(f"👤 {author_text}")

    if pub_date := source.get("published"):
        metadata_parts.append(f"📅 {pub_date}")

    if metadata_parts:
        st.caption(" • ".join(metadata_parts))


def process_query_streaming(prompt: str):
    """
    Process a user query through the RAG pipeline with streaming.
    
    Yields status updates and final result.
    """
    initial_state = {
        "query": prompt,
        "sub_tasks": [],
        "planning_reasoning": "",
        "retrieved_docs": [],
        "retrieval_metadata": {},
        "analysis": [],
        "overall_assessment": "",
        "final_answer": "",
        "sources": [],
        "error": None,
        "current_agent": "start",
    }

    try:
        # Stream events from the graph
        agent_emojis = {
            "planner": "🎯",
            "researcher": "🔍",
            "reasoner": "🧠",
            "synthesiser": "✍️",
        }

        final_result = None

        # Use stream mode to get intermediate updates
        for event in st.session_state.graph.stream(initial_state):
            # Event is a dict with node name as key
            for node_name, node_output in event.items():
                if node_name in agent_emojis:
                    emoji = agent_emojis[node_name]
                    yield {
                        "type": "status",
                        "agent": node_name,
                        "emoji": emoji,
                        "message": f"{emoji} {node_name.title()} working...",
                    }

                # Store final state
                final_result = node_output

        # Check for errors in final result
        if final_result and final_result.get("error"):
            yield {
                "type": "error",
                "message": f"❌ Pipeline Error: {final_result['error']}",
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
                "message": "❌ No response from pipeline",
                "sources": [],
            }

    except Exception as e:
        logger.exception("Error processing query")
        yield {
            "type": "error",
            "message": f"❌ System Error: {str(e)}",
            "sources": [],
        }


def process_query(prompt: str):
    """Process a user query through the RAG pipeline (non-streaming)."""
    initial_state = {
        "query": prompt,
        "sub_tasks": [],
        "planning_reasoning": "",
        "retrieved_docs": [],
        "retrieval_metadata": {},
        "analysis": [],
        "overall_assessment": "",
        "final_answer": "",
        "sources": [],
        "error": None,
        "current_agent": "start",
    }

    try:
        # Execute graph
        result = st.session_state.graph.invoke(initial_state)

        if result.get("error"):
            return {
                "success": False,
                "message": f"❌ Pipeline Error: {result['error']}",
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
            "message": f"❌ System Error: {str(e)}",
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

                session_mgr.save_session(
                    session_id,
                    st.session_state.messages,
                    st.session_state.session_name,
                )
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
            streaming = st.checkbox(
                "Enable Streaming",
                value=st.session_state.streaming_enabled,
                help="Stream responses character by character with real-time agent status",
            )
            if streaming != st.session_state.streaming_enabled:
                st.session_state.streaming_enabled = streaming
                st.success("✓ Setting updated!")

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
            st.markdown(message["content"])

            # Display sources if available
            if message.get("sources"):
                with st.expander(
                    f"📚 Sources ({len(message['sources'])})"
                ):
                    for i, source in enumerate(message["sources"]):
                        display_source_card(source, i)
                        if i < len(message["sources"]) - 1:
                            st.divider()

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

                final_message = ""
                final_sources = []

                try:
                    for update in process_query_streaming(prompt):
                        if update["type"] == "status":
                            # Show which agent is currently working
                            with status_container:
                                st.caption(update["message"])

                        elif update["type"] == "complete":
                            # Clear status and show final answer
                            status_container.empty()
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
                                f"📚 Sources ({len(final_sources)})"
                            ):
                                for i, source in enumerate(final_sources):
                                    display_source_card(source, i)
                                    if i < len(final_sources) - 1:
                                        st.divider()

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
                    with sources_container, st.expander(
                        f"📚 Sources ({len(final_sources)})"
                    ):
                        for i, source in enumerate(final_sources):
                            display_source_card(source, i)
                            if i < len(final_sources) - 1:
                                st.divider()

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
