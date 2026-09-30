"""LangGraph workflow: intent -> (RAG | DB tools | general) -> response."""
import logging
from functools import lru_cache
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app.agents.state import AgentState, Route
from app.config import get_settings
from app.tools import DB_TOOLS
from app.tools.policy_tools import format_context
from app.vectorstore.store import search

logger = logging.getLogger(__name__)

# One in-memory checkpointer shared by every model's graph, so conversation
# history (keyed by thread_id) survives across requests and model switches.
# Swap for a Postgres checkpointer if you need persistence across restarts.
_checkpointer = MemorySaver()

CLASSIFIER_SYSTEM = """You route messages for a company assistant.
Choose exactly one route:
- "rag": questions about company policies, HR, leave, benefits, remote work, expenses, conduct.
- "tool": requests to check the status of an order or cancel an order (or that mention an order ID).
- "general": greetings, small talk, thanks, or anything else.
Use the recent conversation to resolve follow-ups (e.g. "cancel it" after discussing an order -> "tool")."""

TOOL_SYSTEM = """You are an order-support agent with access to the orders database.
- Use get_order_status to look up orders and cancel_order to cancel them.
- Only call cancel_order when the user has clearly asked to cancel a specific order.
- If the user has not given an order ID (and none appears earlier in the conversation), ask for it.
- Report tool results faithfully and concisely. Never invent order details."""

RAG_SYSTEM = """You are a helpful company-policy assistant.
Answer ONLY using the context below. If the context does not contain the answer, say you could not
find it in the company knowledge base and suggest contacting HR. Be concise and specific.

Context:
{context}"""

GENERAL_SYSTEM = """You are a friendly, concise company assistant. You can answer questions about
company policies and help with order status checks and cancellations. For anything else, chat
politely and briefly."""


class Intent(BaseModel):
    route: str = Field(
        description="Exactly one of: 'rag', 'tool', 'general' - the path that should handle the user's message"
    )


def _text(content) -> str:
    """Gemini may return content as a list of parts; flatten to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


def _recent_transcript(messages, limit: int = 6) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {_text(m.content)}")
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            lines.append(f"Assistant: {_text(m.content)}")
    return "\n".join(lines[-limit:])


def build_graph(model_name: str):
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=model_name, api_key=settings.google_api_key, temperature=0.2, max_retries=3
    )
    classifier = ChatGoogleGenerativeAI(
        model=model_name, api_key=settings.google_api_key, temperature=0, max_retries=3
    ).with_structured_output(Intent)
    llm_with_tools = llm.bind_tools(DB_TOOLS)

    # ---- Node 1: intent classifier -------------------------------------
    def intent_classifier_node(state: AgentState) -> dict:
        try:
            result = classifier.invoke(
                [
                    SystemMessage(content=CLASSIFIER_SYSTEM),
                    HumanMessage(content=_recent_transcript(state["messages"])),
                ]
            )
            route = str(result.route).strip().lower()
            if route not in ("rag", "tool", "general"):
                route = "general"
        except Exception:  # noqa: BLE001
            logger.exception("Intent classification failed; falling back to general")
            route = "general"
        logger.info("Intent route: %s", route)
        # Reset per-turn fields
        return {"next_step": route, "context": "", "sources": []}

    # ---- Node 2: RAG ----------------------------------------------------
    def rag_node(state: AgentState) -> dict:
        question = next(
            (_text(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        results = search(question)
        sources = list(
            dict.fromkeys(
                f"{r['metadata'].get('title', 'Policy')} ({r['metadata'].get('source', '?')})"
                for r in results
            )
        )
        return {"context": format_context(results), "sources": sources}

    # ---- Node 3: tool-calling agent (+ ToolNode) -----------------------
    def tool_agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(
            [SystemMessage(content=TOOL_SYSTEM), *state["messages"]]
        )
        return {"messages": [response]}

    tool_node = ToolNode(DB_TOOLS)

    # ---- Node 4: response generator ------------------------------------
    def response_generator_node(state: AgentState) -> dict:
        step = state.get("next_step", "general")
        last = state["messages"][-1]

        # Tool path: the agent's final (tool-call-free) message is already the answer.
        if step == "tool" and isinstance(last, AIMessage) and last.content and not last.tool_calls:
            return {"sources": state.get("sources", [])}

        if step == "rag":
            system = RAG_SYSTEM.format(
                context=state.get("context") or "(no relevant documents found)"
            )
        else:
            system = GENERAL_SYSTEM
        response = llm.invoke([SystemMessage(content=system), *state["messages"]])
        return {"messages": [response]}

    # ---- Routing ---------------------------------------------------------
    def route_after_intent(
        state: AgentState,
    ) -> Literal["rag_node", "tool_agent", "response_generator"]:
        step = state.get("next_step")
        if step == "rag":
            return "rag_node"
        if step == "tool":
            return "tool_agent"
        return "response_generator"

    def route_after_tool_agent(
        state: AgentState,
    ) -> Literal["tools", "response_generator"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "response_generator"

    graph = StateGraph(AgentState)
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("tool_agent", tool_agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("response_generator", response_generator_node)

    graph.add_edge(START, "intent_classifier")
    graph.add_conditional_edges("intent_classifier", route_after_intent)
    graph.add_edge("rag_node", "response_generator")
    graph.add_conditional_edges("tool_agent", route_after_tool_agent)
    graph.add_edge("tools", "tool_agent")
    graph.add_edge("response_generator", END)

    return graph.compile(checkpointer=_checkpointer)


@lru_cache(maxsize=8)
def get_graph(model_name: str):
    return build_graph(model_name)
