"""LangGraph state definition."""
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Route = Literal["rag", "tool", "general"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    next_step: str          # route chosen by the intent classifier
    context: str            # retrieved policy text (RAG path)
    sources: list[str]      # human-readable source names for the UI
