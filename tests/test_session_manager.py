"""
Tests for Session Manager

Run with: pytest tests/test_session_manager.py -v
"""

import json
import tempfile

import pytest

from src.utils.session_manager import SessionManager


@pytest.fixture
def temp_sessions_dir():
    """Create a temporary directory for sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def session_manager(temp_sessions_dir):
    """Create a session manager with temporary directory."""
    return SessionManager(sessions_dir=temp_sessions_dir)


@pytest.fixture
def sample_messages():
    """Sample message list for testing."""
    return [
        {"role": "user", "content": "What is RAG?", "sources": []},
        {
            "role": "assistant",
            "content": "RAG stands for Retrieval Augmented Generation...",
            "sources": [
                {
                    "title": "RAG Paper",
                    "source": "arxiv",
                    "url": "https://arxiv.org/abs/2005.11401",
                }
            ],
        },
    ]


def test_save_session(session_manager, sample_messages):
    """Test saving a session."""
    session_id = "test_session_001"

    result = session_manager.save_session(
        session_id, sample_messages, name="Test Session"
    )

    assert result["id"] == session_id
    assert result["name"] == "Test Session"
    assert result["message_count"] == 2
    assert "created_at" in result
    assert "updated_at" in result


def test_load_session(session_manager, sample_messages):
    """Test loading a saved session."""
    session_id = "test_session_002"

    # Save first
    session_manager.save_session(session_id, sample_messages)

    # Load
    loaded = session_manager.load_session(session_id)

    assert loaded is not None
    assert loaded["id"] == session_id
    assert len(loaded["messages"]) == 2
    assert loaded["messages"][0]["content"] == "What is RAG?"


def test_load_nonexistent_session(session_manager):
    """Test loading a session that doesn't exist."""
    loaded = session_manager.load_session("nonexistent")
    assert loaded is None


def test_delete_session(session_manager, sample_messages):
    """Test deleting a session."""
    session_id = "test_session_003"

    # Save first
    session_manager.save_session(session_id, sample_messages)

    # Verify it exists
    assert session_manager.load_session(session_id) is not None

    # Delete
    result = session_manager.delete_session(session_id)
    assert result is True

    # Verify it's gone
    assert session_manager.load_session(session_id) is None


def test_list_sessions(session_manager, sample_messages):
    """Test listing all sessions."""
    # Create multiple sessions
    session_manager.save_session("session_1", sample_messages, "First")
    session_manager.save_session("session_2", sample_messages, "Second")
    session_manager.save_session("session_3", sample_messages, "Third")

    sessions = session_manager.list_sessions()

    assert len(sessions) == 3
    assert all("id" in s for s in sessions)
    assert all("name" in s for s in sessions)
    assert all("message_count" in s for s in sessions)


def test_rename_session(session_manager, sample_messages):
    """Test renaming a session."""
    session_id = "test_session_004"

    # Save with initial name
    session_manager.save_session(
        session_id, sample_messages, name="Original Name"
    )

    # Rename
    result = session_manager.rename_session(session_id, "New Name")
    assert result is True

    # Verify
    loaded = session_manager.load_session(session_id)
    assert loaded["name"] == "New Name"


def test_export_session_json(session_manager, sample_messages):
    """Test exporting session as JSON."""
    session_id = "test_session_005"

    session_manager.save_session(session_id, sample_messages)

    json_str = session_manager.export_session_json(session_id)

    assert json_str is not None
    # Verify it's valid JSON
    data = json.loads(json_str)
    assert data["id"] == session_id
    assert len(data["messages"]) == 2


def test_export_session_markdown(session_manager, sample_messages):
    """Test exporting session as Markdown."""
    session_id = "test_session_006"

    session_manager.save_session(
        session_id, sample_messages, name="Test Session"
    )

    md_str = session_manager.export_session_markdown(session_id)

    assert md_str is not None
    assert "# Chat Session: Test Session" in md_str
    assert "## 👤 User" in md_str
    assert "## 🤖 Assistant" in md_str
    assert "What is RAG?" in md_str
    assert "RAG stands for" in md_str


def test_generate_session_name(session_manager, sample_messages):
    """Test automatic session name generation."""
    session_id = "test_session_007"

    # Save without providing a name
    result = session_manager.save_session(session_id, sample_messages)

    # Should generate name from first user message
    assert "What is RAG" in result["name"]


def test_update_existing_session(session_manager, sample_messages):
    """Test updating an existing session."""
    session_id = "test_session_008"

    # Save initial
    session_manager.save_session(
        session_id, sample_messages, name="Original"
    )
    original_created = session_manager.load_session(session_id)[
        "created_at"
    ]

    # Update with more messages
    updated_messages = sample_messages + [
        {"role": "user", "content": "Tell me more", "sources": []}
    ]
    session_manager.save_session(session_id, updated_messages)

    # Load and verify
    loaded = session_manager.load_session(session_id)
    assert loaded["message_count"] == 3
    assert loaded["created_at"] == original_created  # Should preserve
    assert (
        loaded["updated_at"] != original_created
    )  # Should be different


def test_get_session_count(session_manager, sample_messages):
    """Test getting session count."""
    assert session_manager.get_session_count() == 0

    session_manager.save_session("s1", sample_messages)
    session_manager.save_session("s2", sample_messages)

    assert session_manager.get_session_count() == 2


def test_empty_messages(session_manager):
    """Test saving a session with no messages."""
    session_id = "test_empty"

    result = session_manager.save_session(session_id, [])

    assert result["message_count"] == 0
    loaded = session_manager.load_session(session_id)
    assert len(loaded["messages"]) == 0
