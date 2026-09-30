"""Ingest policy text files (data/policies/*.txt) into ChromaDB."""
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.vectorstore.store import get_client, get_collection, get_embeddings

logger = logging.getLogger(__name__)


def ingest_policies(force: bool = False) -> int:
    """Chunk, embed and store policy documents. Returns number of chunks written.

    Skips work if the collection already has data, unless ``force`` is True
    (which drops and rebuilds the collection).
    """
    settings = get_settings()
    if force:
        try:
            get_client().delete_collection(settings.chroma_collection)
        except Exception:  # noqa: BLE001
            pass
    collection = get_collection()
    if not force and collection.count() > 0:
        logger.info("Policy collection already populated; skipping ingestion")
        return 0

    files = sorted((Path(settings.data_dir) / "policies").glob("*.txt"))
    if not files:
        logger.warning("No policy files found in %s", settings.data_dir)
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    ids, texts, metas = [], [], []
    for path in files:
        content = path.read_text(encoding="utf-8").strip()
        first_line = content.splitlines()[0] if content else path.stem
        title = first_line.removeprefix("Title:").strip() or path.stem
        for i, chunk in enumerate(splitter.split_text(content)):
            ids.append(f"{path.stem}-{i}")
            texts.append(chunk)
            metas.append({"source": path.name, "title": title, "chunk": i})

    embeddings = get_embeddings().embed_documents(texts)
    collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
    logger.info("Ingested %d chunks from %d files", len(texts), len(files))
    return len(texts)
