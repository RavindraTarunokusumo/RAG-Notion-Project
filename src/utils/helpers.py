import hashlib
import re

from langchain_core.documents import Document


def extract_arxiv_id(url: str) -> str | None:
    """
    Extract Arxiv ID from various URL formats.
    
    Supported formats:
    - https://arxiv.org/abs/2401.12345
    - https://arxiv.org/pdf/2401.12345.pdf
    - http://arxiv.org/abs/2401.12345v2
    - arxiv:2401.12345
    """
    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+)(v\d+)?',
        r'arxiv\.org/pdf/(\d+\.\d+)(v\d+)?',
        r'arxiv:(\d+\.\d+)(v\d+)?',
        r'(\d{4}\.\d{4,5})(v\d+)?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
             return match.group(1)
    return None

def deduplicate_documents(docs: list[Document]) -> list[Document]:
    """Remove duplicate documents based on content hash."""
    seen_hashes = set()
    unique_docs = []
    
    for doc in docs:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    
    return unique_docs

def format_documents_for_prompt(docs: list[Document], max_chars: int = 15000) -> str:
    """Format documents for inclusion in LLM prompts."""
    formatted_parts = []
    total_chars = 0
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "Untitled")
        content = doc.page_content.strip()
        
        doc_text = f"Source [{i}]: {title} ({source})\n{content}\n"
        
        if total_chars + len(doc_text) > max_chars:
            break
            
        formatted_parts.append(doc_text)
        total_chars += len(doc_text)
    
    return "\n---\n".join(formatted_parts)

def safe_json_parse(text: str) -> dict | None:
    """Safely parse JSON from LLM output, handling common issues."""
    import json
    
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object in text
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
             return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None
