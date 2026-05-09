"""Pydantic schemas for the public chat API."""

from typing import Any, Literal

from pydantic import BaseModel, Field

UserRole = Literal["cfo", "finance_manager", "accountant", "auditor"]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    user_role: UserRole
    company_id: int = Field(..., gt=0)
    thread_id: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    suggested_action: dict[str, Any] | None = None
    route: str
    route_reasoning: str | None = None
    thread_id: str
    errors: list[str] = Field(default_factory=list)


class ComponentStatus(BaseModel):
    ok: bool
    error: str | None = None


class ReadyResponse(BaseModel):
    ok: bool
    components: dict[str, ComponentStatus]
