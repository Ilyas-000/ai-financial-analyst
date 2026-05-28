"""Unified logging setup (structlog + stdlib bridge).

The whole app talks to ``structlog.get_logger(__name__)``. ``configure_logging``
wires three things together:

1. structlog processor chain — merges contextvars, adds level/timestamp,
   formats exceptions; the last step is ``ProcessorFormatter.wrap_for_formatter``
   so the actual rendering is delegated to the stdlib handler below.
2. stdlib root handler that uses ``structlog.stdlib.ProcessorFormatter`` with
   the same ``foreign_pre_chain``. This forces uvicorn / sqlalchemy / Langfuse
   stdlib loggers through the same renderer, so everything in stdout shares
   one format.
3. Renderer choice — ``JSONRenderer`` when ``settings.log_format == "json"``,
   ``ConsoleRenderer`` otherwise. JSON is the default; ``console`` is meant
   for local interactive runs where a tail of plain text reads better.

Per-request context (``thread_id``, ``user_id``, ``user_role``,
``company_id``, ``route``) is bound via :func:`structlog.contextvars.bind_contextvars`
in ``ChatService``. Because ``contextvars`` are propagated across ``await``
boundaries, every log line emitted inside one graph run inherits the same
attributes — including from nested LangGraph nodes and asyncio tasks the
graph spawns.
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
