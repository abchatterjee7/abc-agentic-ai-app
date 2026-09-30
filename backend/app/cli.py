"""Maintenance commands.

    python -m app.cli seed [--reset]     # (re)load sample orders
    python -m app.cli ingest [--force]   # (re)build the policy vector index
"""
import argparse
import logging

from app.db.seed import seed_orders
from app.vectorstore.ingest import ingest_policies


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_seed = sub.add_parser("seed", help="Seed sample orders")
    p_seed.add_argument("--reset", action="store_true", help="Delete existing orders first")
    p_ing = sub.add_parser("ingest", help="Ingest policy documents into ChromaDB")
    p_ing.add_argument("--force", action="store_true", help="Drop and rebuild the collection")
    args = parser.parse_args()

    if args.cmd == "seed":
        print(f"Seeded {seed_orders(reset=args.reset)} orders")
    else:
        print(f"Ingested {ingest_policies(force=args.force)} chunks")


if __name__ == "__main__":
    main()
