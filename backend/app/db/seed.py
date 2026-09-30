"""Seed the orders table from data/seed_orders.json (only when empty)."""
import json
import logging
from pathlib import Path

from sqlalchemy import delete, func, select

from app.config import get_settings
from app.db.database import init_db, session_scope
from app.db.models import Order

logger = logging.getLogger(__name__)


def seed_orders(reset: bool = False) -> int:
    init_db()
    path = Path(get_settings().data_dir) / "seed_orders.json"
    if not path.exists():
        logger.warning("Seed file not found: %s", path)
        return 0
    rows = json.loads(path.read_text(encoding="utf-8"))
    with session_scope() as session:
        if reset:
            session.execute(delete(Order))
        elif session.scalar(select(func.count()).select_from(Order)):
            return 0
        session.add_all(Order(**row) for row in rows)
    logger.info("Seeded %d orders", len(rows))
    return len(rows)
