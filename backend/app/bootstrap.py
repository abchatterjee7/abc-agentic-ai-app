"""Startup tasks: wait for services, create tables, seed data, ingest policies."""
import logging
import time

from app.config import get_settings
from app.db.database import db_is_healthy, init_db
from app.db.seed import seed_orders
from app.vectorstore.ingest import ingest_policies
from app.vectorstore.store import vectorstore_is_healthy

logger = logging.getLogger(__name__)


def _wait(check, name: str, attempts: int = 30, delay: float = 2.0) -> bool:
    for i in range(1, attempts + 1):
        if check():
            return True
        logger.info("Waiting for %s (%d/%d)...", name, i, attempts)
        time.sleep(delay)
    logger.error("%s is not reachable", name)
    return False


def bootstrap() -> None:
    settings = get_settings()

    if _wait(db_is_healthy, "PostgreSQL"):
        init_db()
        seed_orders()
    else:
        logger.error("Skipping DB init: PostgreSQL unreachable")

    if _wait(vectorstore_is_healthy, "ChromaDB"):
        if settings.google_api_key:
            try:
                ingest_policies()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Policy ingestion failed (check GOOGLE_API_KEY). "
                    "Retry with: python -m app.cli ingest --force"
                )
        else:
            logger.warning("GOOGLE_API_KEY not set; skipping policy ingestion")
