"""FastAPI dependency providers."""

from app.services.chat_service import ChatService, get_chat_service


def chat_service_dependency() -> ChatService:
    return get_chat_service()
