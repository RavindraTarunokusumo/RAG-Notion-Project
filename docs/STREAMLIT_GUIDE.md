# Streamlit Chat Interface - Quick Start

## Installation

Install the streamlit dependency:

```bash
uv sync
```

## Running the Application

Launch the Streamlit chat interface:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Features

### Chat Interface
- **Message History**: All conversations are stored in the session
- **User-Friendly UI**: Clean, modern chat interface
- **Rich Responses**: Markdown-formatted answers with proper citations
- **Streaming Responses (NEW!)**: Real-time text streaming with agent status updates

### Session Management
- **Save Sessions**: Persist chat conversations to disk
- **Load Sessions**: Resume previous conversations
- **Auto-Save**: Sessions automatically named from first message
- **Delete Sessions**: Remove unwanted conversations
- **Export Options**: Export sessions as JSON or Markdown

### Source Display
- **Expandable Source Cards**: Click to view sources for each response
- **Rich Metadata**: View authors, publication dates, topics, keywords, and source types
- **Clickable Links**: Direct links to arXiv papers and Notion pages
- **Numbered Citations**: Sources referenced with [1], [2] format in answers
- **Content Snippets**: Preview relevant content from each source
- **Source Filtering**: Filter sources by type (ArXiv, Notion)
- **Search Sources**: Search within sources by title, keywords, or topic
- **Source Badges**: Visual distinction between different source types

### Sidebar
- **System Status**: See if the RAG graph is initialized
- **Session Management Panel**: Save, load, and manage conversations
- **Settings Panel**: Toggle streaming on/off
- **Agent Pipeline Info**: Understand the 4-agent workflow
- **Clear Chat**: Reset conversation history
- **Message Counter**: Track conversation length

## Streaming Feature

### What is Streaming?
Streaming mode provides real-time feedback during query processing:
- **Agent Status Updates**: See which agent is currently working (Planner → Researcher → Reasoner → Synthesiser)
- **Character-by-Character Display**: Watch the response appear in real-time
- **Better UX**: More engaging and responsive experience

### Using Streaming
1. Open Settings panel in sidebar (⚙️ Settings)
2. Check/uncheck "Enable Streaming"
3. Setting is saved in your session
4. Submit a query to see the difference

**With Streaming ON:**
- See live agent updates: "🎯 Planner working...", "🔍 Researcher working..."
- Response appears character by character with cursor
- More interactive feel

**With Streaming OFF:**
- Simple "Processing..." message
- Complete response appears at once
- Faster perceived completion for short answers

## Session Management

### Saving a Session
1. Start a conversation by asking questions
2. Click the **💾 Save** button in the sidebar
3. Session is saved with an auto-generated name
4. Session ID format: `YYYYMMDD_HHMMSS`

### Loading a Session
1. View saved sessions in the "Saved Sessions" list
2. Click on a session name to load it
3. Chat history will be restored
4. Continue the conversation or start a new one

### Exporting Sessions
1. Load the session you want to export
2. Click the **📤 Export** expander in the sidebar
3. Choose format (JSON or Markdown)
4. Click download button to save

### Session Storage
- Sessions stored in `./data/sessions/`
- Each file named `{session_id}.json`
- Persists across application restarts
- Not tracked in git

## Agent Pipeline

The system uses 4 specialized agents:

1. **🎯 Planner**: Decomposes your question into sub-tasks
2. **🔍 Researcher**: Retrieves relevant documents from the vector store
3. **🧠 Reasoner**: Analyzes the retrieved information using Cohere's reasoning model
4. **✍️ Synthesiser**: Generates a comprehensive answer with numbered citations

## Source Citations

The synthesiser includes numbered citations in responses:
- Citations appear as **[1]**, **[2]**, etc.
- Click on a citation number to jump to the source details
- Each source card shows the citation number for easy reference
- Sources include rich metadata:
  - **Authors** (for research papers)
  - **Publication dates**
  - **Topics and keywords** (for Notion entries)
  - **Content snippets** for preview
  - **Direct links** to original sources

### Filtering Sources
Within the sources expander:
1. Use the **Filter by type** dropdown to show only ArXiv or Notion sources
2. Use the **Search sources** box to find sources by title, keywords, or topic
3. Original citation numbers are preserved when filtering

## Troubleshooting

### "Failed to initialize graph"
- Check that your `.env` file has all required API keys
- Ensure the vector store is populated: `uv run python main.py --ingest`
- Check that all dependencies are installed: `uv sync`

### "No response from agent"
- Check your Cohere API key and quota
- Enable verbose logging in `main.py` to see detailed errors
- Check LangSmith traces if tracing is enabled

### Port already in use
If port 8501 is already in use, specify a different port:

```bash
streamlit run app.py --server.port 8502
```

## Next Steps

See `docs/PROJECT.md` for additional UI features planned:
- ~~NRAG-051: Session management (save/load conversations)~~ ✅ Completed
- ~~NRAG-052: Enhanced source display with filtering~~ ✅ Completed
- NRAG-053: Real-time agent progress visualization
- NRAG-054: Runtime settings configuration
