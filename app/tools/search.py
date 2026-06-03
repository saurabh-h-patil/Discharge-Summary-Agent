"""
Search Tool — search across all extracted clinical notes for specific terms.
"""

import json
from langchain_core.tools import tool


# Module-level store for searchable documents (set during agent initialization)
_searchable_documents: list[dict] = []


def set_searchable_documents(documents: list[dict]):
    """Register documents for searching. Called during agent setup."""
    global _searchable_documents
    _searchable_documents = documents


@tool
def search_across_notes(query: str) -> str:
    """Search across all extracted clinical notes for a specific term or phrase.
    
    Use this to find specific information across multiple documents, such as:
    - A specific medication name
    - A diagnosis or condition
    - A date or event
    - Lab test names
    - Provider names
    
    Args:
        query: The search term or phrase to look for (case-insensitive)
    """
    if not _searchable_documents:
        return json.dumps({
            "status": "error",
            "message": "No documents loaded for searching.",
        })

    results = []
    query_lower = query.lower()

    for doc in _searchable_documents:
        doc_text = doc.get("text", "")
        if query_lower in doc_text.lower():
            # Find the relevant lines
            lines = doc_text.split("\n")
            matching_lines = []
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    # Include context: one line before and after
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    context = "\n".join(lines[start:end])
                    matching_lines.append({
                        "line_number": i + 1,
                        "context": context.strip(),
                    })

            results.append({
                "document_id": doc.get("doc_id", "unknown"),
                "document_type": doc.get("document_type", "unknown"),
                "source_file": doc.get("source_file", "unknown"),
                "matches": matching_lines[:10],  # Limit matches per doc
            })

    if not results:
        return json.dumps({
            "status": "not_found",
            "message": f"No matches found for '{query}' across {len(_searchable_documents)} documents.",
        })

    return json.dumps({
        "status": "found",
        "query": query,
        "total_documents_matched": len(results),
        "results": results,
    })
