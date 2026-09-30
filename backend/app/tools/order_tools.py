"""LangChain tools that operate on the orders database."""
import logging

from langchain_core.tools import tool
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.db.database import session_scope
from app.db.models import Order

logger = logging.getLogger(__name__)

CANCELLABLE_STATUSES = {"pending", "processing", "confirmed"}


def _normalize(order_id: str) -> str:
    return order_id.strip().upper()


@tool
def get_order_status(order_id: str) -> str:
    """Look up the current status and details of a customer order.

    Args:
        order_id: The order identifier, for example 'ORD-1001'.
    """
    oid = _normalize(order_id)
    try:
        with session_scope() as session:
            order = session.scalar(select(Order).where(Order.order_id == oid))
            if order is None:
                return f"No order found with ID {oid}."
            return (
                f"Order {order.order_id}: status={order.status}; item={order.item} "
                f"x{order.quantity}; total={order.total_amount}; "
                f"customer={order.customer_name}; "
                f"placed={order.created_at:%Y-%m-%d}."
            )
    except SQLAlchemyError:
        logger.exception("Database error in get_order_status")
        return "A database error occurred while looking up the order."


@tool
def cancel_order(order_id: str) -> str:
    """Cancel a customer order. Only orders that are pending, processing or
    confirmed can be cancelled; shipped or delivered orders cannot.

    Args:
        order_id: The order identifier, for example 'ORD-1001'.
    """
    oid = _normalize(order_id)
    try:
        with session_scope() as session:
            order = session.scalar(
                select(Order).where(Order.order_id == oid).with_for_update()
            )
            if order is None:
                return f"No order found with ID {oid}."
            if order.status == "cancelled":
                return f"Order {oid} is already cancelled."
            if order.status not in CANCELLABLE_STATUSES:
                return (
                    f"Order {oid} cannot be cancelled because its status is "
                    f"'{order.status}'."
                )
            order.status = "cancelled"
            return f"Order {oid} has been cancelled successfully."
    except SQLAlchemyError:
        logger.exception("Database error in cancel_order")
        return "A database error occurred while cancelling the order."
