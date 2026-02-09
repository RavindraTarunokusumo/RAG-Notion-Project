"""
Session Management for Streamlit Chat Interface

Handles saving, loading, and managing chat sessions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages chat session persistence and operations.

    Sessions are stored as JSON files in the data/sessions directory.
    Each session contains:
    - id: Unique identifier
    - name: User-friendly name
    - messages: List of chat messages
    - created_at: Creation timestamp
    - updated_at: Last update timestamp
    - message_count: Number of messages
    """

    def __init__(self, sessions_dir: str = "./data/sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Session manager initialized at {self.sessions_dir}")

    def save_session(
        self, session_id: str, messages: list, name: str | None = None
    ) -> dict:
        """
        Save a chat session to disk.

        Args:
            session_id: Unique session identifier
            messages: List of message dictionaries
            name: Optional custom name for the session

        Returns:
            Session metadata dictionary
        """
        session_file = self.sessions_dir / f"{session_id}.json"

        # Load existing session if it exists to preserve metadata
        if session_file.exists():
            try:
                with open(session_file, encoding="utf-8") as f:
                    existing_data = json.load(f)
                    created_at = existing_data.get(
                        "created_at", datetime.now().isoformat()
                    )
                    existing_name = existing_data.get("name")
            except Exception as e:
                logger.warning(f"Error reading existing session: {e}")
                created_at = datetime.now().isoformat()
                existing_name = None
        else:
            created_at = datetime.now().isoformat()
            existing_name = None

        # Use provided name, or existing name, or generate default
        if name is None:
            name = existing_name or self._generate_session_name(
                session_id, messages
            )

        session_data = {
            "id": session_id,
            "name": name,
            "messages": messages,
            "created_at": created_at,
            "updated_at": datetime.now().isoformat(),
            "message_count": len(messages),
        }

        try:
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved session {session_id} with {len(messages)} messages")
            return session_data
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            raise

    def load_session(self, session_id: str) -> dict | None:
        """
        Load a session from disk.

        Args:
            session_id: Session identifier

        Returns:
            Session data dictionary or None if not found
        """
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            logger.warning(f"Session {session_id} not found")
            return None

        try:
            with open(session_file, encoding="utf-8") as f:
                session_data = json.load(f)
            logger.info(f"Loaded session {session_id}")
            return session_data
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from disk.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False otherwise
        """
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            logger.warning(f"Cannot delete - session {session_id} not found")
            return False

        try:
            session_file.unlink()
            logger.info(f"Deleted session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def list_sessions(self) -> list[dict]:
        """
        List all available sessions.

        Returns:
            List of session metadata dictionaries sorted by updated_at (newest first)
        """
        sessions = []

        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, encoding="utf-8") as f:
                    session_data = json.load(f)

                # Include only metadata, not full messages
                sessions.append(
                    {
                        "id": session_data.get("id"),
                        "name": session_data.get("name", session_file.stem),
                        "created_at": session_data.get("created_at"),
                        "updated_at": session_data.get("updated_at"),
                        "message_count": session_data.get(
                            "message_count", 0
                        ),
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Failed to read session {session_file.name}: {e}"
                )
                continue

        # Sort by updated_at (newest first)
        sessions.sort(
            key=lambda x: x.get("updated_at", ""), reverse=True
        )

        return sessions

    def rename_session(self, session_id: str, new_name: str) -> bool:
        """
        Rename a session.

        Args:
            session_id: Session identifier
            new_name: New name for the session

        Returns:
            True if renamed, False otherwise
        """
        session_data = self.load_session(session_id)

        if not session_data:
            return False

        try:
            self.save_session(
                session_id, session_data["messages"], name=new_name
            )
            logger.info(f"Renamed session {session_id} to '{new_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to rename session {session_id}: {e}")
            return False

    def export_session_json(self, session_id: str) -> str | None:
        """
        Export session as JSON string.

        Args:
            session_id: Session identifier

        Returns:
            JSON string or None if session not found
        """
        session_data = self.load_session(session_id)

        if not session_data:
            return None

        try:
            return json.dumps(session_data, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to export session {session_id}: {e}")
            return None

    def export_session_markdown(self, session_id: str) -> str | None:
        """
        Export session as Markdown string.

        Args:
            session_id: Session identifier

        Returns:
            Markdown string or None if session not found
        """
        session_data = self.load_session(session_id)

        if not session_data:
            return None

        try:
            lines = [
                f"# Chat Session: {session_data.get('name', session_id)}",
                "",
                f"**Created:** {session_data.get('created_at', 'Unknown')}",
                f"**Updated:** {session_data.get('updated_at', 'Unknown')}",
                f"**Messages:** {session_data.get('message_count', 0)}",
                "",
                "---",
                "",
            ]

            for msg in session_data.get("messages", []):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role == "user":
                    lines.append("## 👤 User")
                else:
                    lines.append("## 🤖 Assistant")

                lines.append("")
                lines.append(content)
                lines.append("")

                # Add sources if available
                if msg.get("sources"):
                    lines.append("### Sources")
                    lines.append("")
                    for source in msg["sources"]:
                        title = source.get("title", "Untitled")
                        url = (
                            source.get("url")
                            or source.get("arxiv_url")
                            or source.get("notion_url")
                        )
                        if url:
                            lines.append(f"- [{title}]({url})")
                        else:
                            lines.append(f"- {title}")
                    lines.append("")

                lines.append("---")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error(
                f"Failed to export session {session_id} to markdown: {e}"
            )
            return None

    def _generate_session_name(
        self, session_id: str, messages: list
    ) -> str:
        """
        Generate a default session name based on first user message.

        Args:
            session_id: Session identifier
            messages: List of messages

        Returns:
            Generated session name
        """
        # Try to use first user message as name
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Truncate to first 50 chars
                name = content[:50].strip()
                if name:
                    # Add ellipsis if truncated
                    if len(content) > 50:
                        name += "..."
                    return name

        # Fallback to timestamp-based name
        return f"Session {session_id[:8]}"

    def get_session_count(self) -> int:
        """Get total number of saved sessions."""
        return len(list(self.sessions_dir.glob("*.json")))


# Singleton instance
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the singleton session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
