from src.loaders.notion_loader import NotionKnowledgeBaseLoader

def test_notion_knowledge_base_loader():
    loader = NotionKnowledgeBaseLoader()
    entries = loader.load_entries(use_cache=False)
    
    assert isinstance(entries, list)
    assert len(entries) > 0
    
    for entry in entries:
        assert hasattr(entry, "notion_id")
        assert hasattr(entry, "title")
        assert hasattr(entry, "topic")
        assert hasattr(entry, "keywords")
        assert hasattr(entry, "source_url")
        assert hasattr(entry, "arxiv_id")
        assert hasattr(entry, "entry_type")
        assert hasattr(entry, "notes")
        assert hasattr(entry, "publication_date")
    
    print(f"Loaded {len(entries)} Notion entries successfully.")
    for entry in entries[:3]:
        print(f"- {entry.title} (Arxiv ID: {entry.arxiv_id})")
        print(f"  Topic: {entry.topic}")
        print(f"  Keywords: {entry.keywords}")

if __name__ == "__main__":
    test_notion_knowledge_base_loader()