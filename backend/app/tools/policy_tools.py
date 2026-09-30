"""LangChain tool for searching company policies in ChromaDB."""
from langchain_core.tools import tool

from app.vectorstore.store import search


def format_context(results: list[dict]) -> str:
    if not results:
        return ""
    blocks = []
    for r in results:
        meta = r["metadata"]
        blocks.append(f"[{meta.get('title', 'Policy')} | {meta.get('source', '?')}]\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


@tool
def search_company_policies(query: str) -> str:
    """Search the internal company knowledge base (leave, benefits, remote work,
    expenses, code of conduct) and return the most relevant passages.

    Args:
        query: A natural-language question about company policy.
    """
    context = format_context(search(query))
    return context or "No relevant policy information found."
