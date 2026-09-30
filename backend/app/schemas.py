"""Pydantic request/response schemas."""
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(default="default", min_length=1, max_length=128)
    model: Optional[str] = Field(default=None, description="Override the default LLM")


class ChatResponse(BaseModel):
    response: str
    tool_used: str = "none"
    sources: list[str] = []


class HealthResponse(BaseModel):
    status: str
    database: bool
    vector_store: bool
    api_key_configured: bool
    default_model: str


class ModelsResponse(BaseModel):
    default: str
    models: list[str]
