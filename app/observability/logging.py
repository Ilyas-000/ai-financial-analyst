"""Unified logging setup (structlog + stdlib bridge).

The app logs via ``structlog.get_logger(__name__)``. ``configure_logging``:

1. Builds the structlog processor chain (merge contextvars, level, timestamp,
   exceptions) ending in ``wrap_for_formatter`` — rendering is delegated to
   the stdlib handler.
2. Routes the stdlib root handler through ``ProcessorFormatter`` with the same
   ``foreign_pre_chain``, so uvicorn / sqlalchemy / Langfuse loggers share one
   format.
3. Picks ``JSONRenderer`` (default) vs ``ConsoleRenderer`` per ``log_format``.

Per-request context bound in ``ChatService`` via ``bind_contextvars`` is
carried across ``await``, so every line in one graph run — including nested
LangGraph nodes — inherits ``thread_id`` / ``user_id`` / ``route`` etc.
"""

import logging
import sys

import structlog

from app.config import Settings


def configure_logging(settings: Settings) -> None:
    """Set up structlog + stdlib root handler. Safe to call multiple times."""
    level = _resolve_level(settings.log_level)
    renderer = _renderer_for(settings.log_format)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Loud third-party defaults that would otherwise drown app logs at INFO.
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))
    logging.getLogger("httpcore").setLevel(max(level, logging.WARNING))


def _resolve_level(name: str) -> int:
    value = logging.getLevelName(name.upper())
    return value if isinstance(value, int) else logging.INFO


def _renderer_for(log_format: str) -> structlog.types.Processor:
    if log_format.lower() == "console":
        return structlog.dev.ConsoleRenderer(colors=False)
    return structlog.processors.JSONRenderer()
