"""FastAPI application."""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.graph import _text, get_graph
from app.bootstrap import bootstrap
from app.config import get_settings
from app.db.database import db_is_healthy
from app.schemas import ChatRequest, ChatResponse, HealthResponse, ModelsResponse
from app.vectorstore.store import vectorstore_is_healthy

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(bootstrap)
    yield


app = FastAPI(title="ABC Agentic AI Assistant", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    db_ok, vs_ok = await asyncio.gather(
        asyncio.to_thread(db_is_healthy), asyncio.to_thread(vectorstore_is_healthy)
    )
    return HealthResponse(
        status="ok" if db_ok and vs_ok else "degraded",
        database=db_ok,
        vector_store=vs_ok,
        api_key_configured=bool(settings.google_api_key),
        default_model=settings.llm_model,
    )


@app.get("/api/v1/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    return ModelsResponse(default=settings.llm_model, models=settings.model_list)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not settings.google_api_key:
        raise HTTPException(503, "GOOGLE_API_KEY is not configured on the server.")
    model = req.model or settings.llm_model
    if model not in settings.model_list:
        raise HTTPException(400, f"Unknown model '{model}'. Options: {settings.model_list}")

    graph = get_graph(model)
    config = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": 15}
    try:
        result = await graph.ainvoke({"messages": [HumanMessage(content=req.message)]}, config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent execution failed")
        raise HTTPException(502, f"Agent execution failed: {exc}") from exc

    messages = result["messages"]
    # Messages produced during this turn = everything after the last human message.
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    turn = messages[last_human + 1 :]

    tools_used: list[str] = []
    for m in turn:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                if call["name"] not in tools_used:
                    tools_used.append(call["name"])
    if result.get("next_step") == "rag":
        tools_used.append("search_company_policies")

    answer = next(
        (m.content for m in reversed(turn) if isinstance(m, AIMessage) and m.content and not m.tool_calls),
        "Sorry, I could not produce a response.",
    )
    return ChatResponse(
        response=_text(answer),
        tool_used=", ".join(tools_used) if tools_used else "none",
        sources=result.get("sources", []) if result.get("next_step") == "rag" else [],
    )
