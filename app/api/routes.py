"""HTTP routes for the chat endpoint."""

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import chat_service_dependency
from app.api.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter()

ChatServiceDep = Annotated[ChatService, Depends(chat_service_dependency)]


@router.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatServiceDep,
) -> ChatResponse:
    result = await service.ask(
        question=request.question,
        user_role=request.user_role,
        company_id=request.company_id,
        thread_id=request.thread_id,
    )
    return ChatResponse(**asdict(result))
