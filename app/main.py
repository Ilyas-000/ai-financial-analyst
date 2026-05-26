"""FastAPI entrypoint.

Owns the application lifespan (engine + Qdrant client + LangGraph
checkpointer pool + compiled graph), the chat router, and the operational
``/health`` / ``/ready`` endpoints. Chainlit is mounted on top of this app
in I-09 — the AI core itself never imports Chainlit.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from chainlit.utils import mount_chainlit
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.routes import router as chat_router
from app.api.schemas import ComponentStatus, ReadyResponse
from app.config import get_settings
from app.db.checkpointer import checkpointer_lifespan
from app.db.session import get_engine
from app.graph.build import build_agent_graph
from app.observability.langfuse_handler import shutdown_langfuse
from app.rag.qdrant_store import get_qdrant_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Warm caches; do not fail startup on transient infra hiccups for the
    # SQL engine and Qdrant client — /ready surfaces real component health.
    get_engine()
    get_qdrant_client()

    # The LangGraph checkpointer needs a live psycopg pool kept open for the
    # whole app lifetime. Postgres must be reachable here — without the saver
    # we cannot serve /api/chat, so failing fast is the right behaviour.
    async with checkpointer_lifespan(settings) as (pool, saver):
        app.state.checkpoint_pool = pool
        app.state.checkpointer = saver
        app.state.graph = build_agent_graph(checkpointer=saver)
        logger.info("application started, env=%s", settings.app_env)
        try:
            yield
        finally:
            shutdown_langfuse()
            engine = get_engine()
            await engine.dispose()
            logger.info("application shutdown complete")


app = FastAPI(
    title="AI Financial Analyst",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(chat_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def _check_postgres(timeout: float) -> ComponentStatus:
    try:
        engine = get_engine()
        async with asyncio.timeout(timeout):
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        return ComponentStatus(ok=True)
    except Exception as exc:
        return ComponentStatus(ok=False, error=f"{type(exc).__name__}: {exc}")


async def _check_qdrant(timeout: float) -> ComponentStatus:
    try:
        client = get_qdrant_client()
        async with asyncio.timeout(timeout):
            await asyncio.to_thread(client.get_collections)
        return ComponentStatus(ok=True)
    except Exception as exc:
        return ComponentStatus(ok=False, error=f"{type(exc).__name__}: {exc}")


async def _check_ollama(timeout: float) -> ComponentStatus:
    settings = get_settings()
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
        return ComponentStatus(ok=True)
    except Exception as exc:
        return ComponentStatus(ok=False, error=f"{type(exc).__name__}: {exc}")


@app.get("/ready")
async def ready():
    settings = get_settings()
    timeout = settings.ready_check_timeout_seconds
    postgres, qdrant, ollama = await asyncio.gather(
        _check_postgres(timeout),
        _check_qdrant(timeout),
        _check_ollama(timeout),
    )
    components = {"postgres": postgres, "qdrant": qdrant, "ollama": ollama}
    overall_ok = all(c.ok for c in components.values())
    body = ReadyResponse(ok=overall_ok, components=components).model_dump()
    if not overall_ok:
        return JSONResponse(status_code=503, content=body)
    return body


# Mount the Chainlit demo UI on /chat. The target file is loaded lazily by
# Chainlit on first request; by then ``app.main`` (and ``app.state.graph``
# inside lifespan) are fully initialised. The chainlit module is the ONLY
# place that may import ``chainlit`` — see rules.md §2.1 and the invariant
# test in tests/test_chainlit_isolation.py.
_CHAINLIT_TARGET = str(Path(__file__).resolve().parents[1] / "app" / "ui" / "chainlit_app.py")
mount_chainlit(app=app, target=_CHAINLIT_TARGET, path="/chat")
