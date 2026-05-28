"""I-12 unified logging: structlog setup + per-request contextvars binding.

These tests pin down two contracts:

* ``configure_logging`` produces a structlog logger that emits a single
  JSON line per call when ``log_format=json``, and a plain console line
  when ``log_format=console``.
* ``ChatService`` binds ``thread_id`` / ``user_id`` / ``user_role`` /
  ``company_id`` to ``structlog.contextvars`` for the duration of the
  request — they are visible to nested log calls during the graph run and
  cleared once the call returns.
"""

import io
import json
import logging

import pytest
import structlog
import structlog.testing

from app.config import Settings
from app.observability.logging import configure_logging
from app.services.chat_service import (
    ChatResult,
    ChatService,
    _bound_log_context,
    synth_user_id,
)


@pytest.fixture(autouse=True)
def _reset_structlog():
    # Each test re-configures structlog from scratch so contextvars don't leak.
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()


def _capture_root_stream() -> tuple[io.StringIO, logging.StreamHandler]:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    root = logging.getLogger()
    root.addHandler(handler)
    return buf, handler


def test_synth_user_id_shape() -> None:
    assert synth_user_id(1, "cfo") == "tenant:1:cfo"
    assert synth_user_id(42, "finance_manager") == "tenant:42:finance_manager"


def test_configure_logging_json_emits_single_json_line() -> None:
    configure_logging(Settings(log_format="json", log_level="INFO"))
    buf, handler = _capture_root_stream()
    try:
        # Replace formatter on our capture handler with the same formatter
        # the root logger uses, so we see the rendered JSON, not raw msg.
        root = logging.getLogger()
        handler.setFormatter(root.handlers[0].formatter)
        logger = structlog.get_logger("app.test")
        with _bound_log_context(
            thread_id="t-1",
            user_id="tenant:1:cfo",
            user_role="cfo",
            company_id=1,
        ):
            logger.info("hello_event", extra_field=99)
    finally:
        logging.getLogger().removeHandler(handler)

    line = buf.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "hello_event"
    assert payload["thread_id"] == "t-1"
    assert payload["user_id"] == "tenant:1:cfo"
    assert payload["user_role"] == "cfo"
    assert payload["company_id"] == 1
    assert payload["extra_field"] == 99
    assert payload["level"] == "info"


def test_configure_logging_console_format() -> None:
    configure_logging(Settings(log_format="console", log_level="INFO"))
    buf, handler = _capture_root_stream()
    try:
        root = logging.getLogger()
        handler.setFormatter(root.handlers[0].formatter)
        logger = structlog.get_logger("app.test")
        logger.warning("console_event", a=1)
    finally:
        logging.getLogger().removeHandler(handler)

    output = buf.getvalue()
    assert "console_event" in output
    assert "a=1" in output
    # Console renderer does NOT emit JSON.
    assert not output.strip().startswith("{")


def test_bound_log_context_clears_on_exit() -> None:
    with _bound_log_context(thread_id="t-x", user_id="u", user_role="cfo", company_id=7):
        ctx = structlog.contextvars.get_contextvars()
        assert ctx["thread_id"] == "t-x"
        assert ctx["user_id"] == "u"
        assert ctx["company_id"] == 7
    after = structlog.contextvars.get_contextvars()
    assert "thread_id" not in after
    assert "user_id" not in after
    assert "company_id" not in after


@pytest.mark.asyncio
async def test_chat_service_binds_contextvars_during_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """During ``ask``, the contextvars are visible to nested log capture."""
    import app.services.chat_service as chat_service_module
    from app.guardrails.output_guard import OutputGuardResult

    seen: dict[str, object] = {}

    async def _fake_ainvoke(initial, config):
        # Snapshot contextvars at the moment the graph is invoked — every
        # log call inside the graph will merge these via merge_contextvars.
        seen["ctx"] = dict(structlog.contextvars.get_contextvars())
        seen["initial"] = initial
        return {"final_answer": "ok", "route": "direct_answer", "sources": []}

    class _StubGraph:
        async def ainvoke(self, initial, config=None):
            return await _fake_ainvoke(initial, config)

    async def _passthrough_guard(*, answer, sources, suggested_action, **kwargs):
        # Capture the user_id passed in so we know ChatService threaded it.
        seen["guard_user_id"] = kwargs.get("user_id")
        return OutputGuardResult(
            answer=answer, sources=list(sources), suggested_action=suggested_action
        )

    monkeypatch.setattr(chat_service_module, "apply_output_guard", _passthrough_guard)

    service = ChatService(_StubGraph())
    result: ChatResult = await service.ask(
        question="привет",
        user_role="cfo",
        company_id=42,
        thread_id="thr-9",
    )

    ctx = seen["ctx"]
    assert ctx["thread_id"] == "thr-9"
    assert ctx["user_id"] == "tenant:42:cfo"
    assert ctx["user_role"] == "cfo"
    assert ctx["company_id"] == 42
    # Initial graph state also carries user_id (route to sql_analyst needs it).
    assert seen["initial"]["user_id"] == "tenant:42:cfo"
    # Output guard receives the same synthesised user_id.
    assert seen["guard_user_id"] == "tenant:42:cfo"
    # Contextvars are cleared after the call returns.
    after = structlog.contextvars.get_contextvars()
    assert "thread_id" not in after
    assert result.answer == "ok"
