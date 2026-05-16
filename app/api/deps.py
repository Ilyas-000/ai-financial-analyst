"""FastAPI dependency providers."""

from fastapi import HTTPException, Request

from app.services.chat_service import ChatService


def chat_service_dependency(request: Request) -> ChatService:
    """Return a per-request ``ChatService`` bound to the lifespan-owned graph.

    The compiled graph (with its ``AsyncPostgresSaver``) is created once on
    application startup and stored on ``app.state``. Any request that lands
    before lifespan finishes wiring it surfaces as a 503 rather than crashing
    inside the route.
    """
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="agent graph not initialised")
    return ChatService(graph)
