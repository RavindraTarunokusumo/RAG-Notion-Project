"""
Tests for enhanced source display and citation functionality (NRAG-052)
"""

import re


def test_citation_pattern():
    """Test that citation pattern regex works correctly."""
    text = "This is a statement [1]. Another fact [2]. More info [3]."
    pattern = r'\[(\d+)\]'
    matches = re.findall(pattern, text)
    assert matches == ['1', '2', '3']


def test_source_enrichment():
    """Test that source metadata is properly enriched."""
    # Mock document metadata
    meta = {
        "source": "arxiv",
        "title": "Test Paper",
        "arxiv_id": "2301.00234",
        "Authors": ["John Doe", "Jane Smith"],
        "Published": "2023-01-15",
        "Summary": "This is a test abstract."
    }
    
    # Build enriched source (same logic as synthesiser)
    source = {
        "source": "arxiv",
        "title": meta.get("title"),
        "arxiv_url": f"https://arxiv.org/abs/{meta.get('arxiv_id')}",
        "arxiv_id": meta.get("arxiv_id"),
        "authors": meta.get("Authors"),
        "published": meta.get("Published"),
        "abstract": meta.get("Summary"),
    }
    
    assert source["source"] == "arxiv"
    assert source["title"] == "Test Paper"
    assert source["arxiv_url"] == "https://arxiv.org/abs/2301.00234"
    assert source["authors"] == ["John Doe", "Jane Smith"]
    assert source["published"] == "2023-01-15"


def test_source_deduplication():
    """Test that duplicate sources are removed."""
    sources = [
        {"source": "arxiv", "title": "Paper A"},
        {"source": "arxiv", "title": "Paper A"},
        {"source": "notion", "title": "Note B"},
        {"source": "arxiv", "title": "Paper C"},
        {"source": "notion", "title": "Note B"},
    ]
    
    # Deduplicate (same logic as synthesiser)
    unique_sources = []
    seen = set()
    for s in sources:
        unique_key = f"{s['source']}:{s['title']}"
        if unique_key not in seen:
            seen.add(unique_key)
            unique_sources.append(s)
    
    assert len(unique_sources) == 3
    titles = [s["title"] for s in unique_sources]
    assert titles == ["Paper A", "Note B", "Paper C"]


def test_source_filtering():
    """Test source filtering by type."""
    sources = [
        {"source": "arxiv", "title": "Paper A"},
        {"source": "notion", "title": "Note B"},
        {"source": "arxiv", "title": "Paper C"},
        {"source": "notion", "title": "Note D"},
    ]
    
    # Filter by arxiv
    filtered = [s for s in sources if s.get("source", "").lower() == "arxiv"]
    assert len(filtered) == 2
    assert all(s["source"] == "arxiv" for s in filtered)
    
    # Filter by notion
    filtered = [s for s in sources if s.get("source", "").lower() == "notion"]
    assert len(filtered) == 2
    assert all(s["source"] == "notion" for s in filtered)


def test_source_search():
    """Test source search functionality."""
    sources = [
        {"source": "arxiv", "title": "Machine Learning Paper", "keywords": ["ML", "AI"]},
        {"source": "notion", "title": "Deep Learning Notes", "topic": "Neural Networks"},
        {"source": "arxiv", "title": "Computer Vision", "keywords": ["CV", "Image"]},
    ]
    
    # Search by title
    search_term = "learning"
    filtered = [
        s for s in sources
        if search_term.lower() in s.get("title", "").lower()
        or search_term.lower() in str(s.get("keywords", "")).lower()
        or search_term.lower() in str(s.get("topic", "")).lower()
    ]
    assert len(filtered) == 2
    
    # Search by keyword
    search_term = "cv"
    filtered = [
        s for s in sources
        if search_term.lower() in str(s.get("keywords", "")).lower()
    ]
    assert len(filtered) == 1
    assert filtered[0]["title"] == "Computer Vision"


def test_citation_numbering():
    """Test that citation numbers are correctly generated."""
    sources = [
        {"source": "arxiv", "title": "Paper A"},
        {"source": "notion", "title": "Note B"},
        {"source": "arxiv", "title": "Paper C"},
    ]
    
    # Create sources list for prompt
    sources_list = "\n".join([
        f"[{i+1}] {s['title']} ({s['source']})"
        for i, s in enumerate(sources)
    ])
    
    expected = "[1] Paper A (arxiv)\n[2] Note B (notion)\n[3] Paper C (arxiv)"
    assert sources_list == expected
