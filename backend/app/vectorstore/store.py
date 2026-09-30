"""ChromaDB (HTTP server) access + Gemini embeddings."""
import logging
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_client() -> "chromadb.api.ClientAPI":
    s = get_settings()
    return chromadb.HttpClient(
        host=s.chroma_host,
        port=s.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


@lru_cache
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    s = get_settings()
    return GoogleGenerativeAIEmbeddings(model=s.embedding_model, api_key=s.google_api_key)


def get_collection():
    return get_client().get_or_create_collection(
        name=get_settings().chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def vectorstore_is_healthy() -> bool:
    try:
        get_client().heartbeat()
        return True
    except Exception:  # noqa: BLE001
        return False


def search(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """Semantic search over the policy collection."""
    k = k or get_settings().rag_top_k
    collection = get_collection()
    if collection.count() == 0:
        return []
    embedding = get_embeddings().embed_query(query)
    res = collection.query(
        query_embeddings=[embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        results.append({"text": doc, "metadata": meta or {}, "distance": dist})
    return results
