"""Unit tests for the I-09 streaming wrapper ``ChatService.astream_ask``.

The wrapper translates LangGraph ``astream_events`` into the
``StreamChunk(kind="reset"|"token"|"result")`` contract consumed by Chainlit.
We feed in synthetic events (no live graph, no LLM) and verify the mapping.
"""

from types import SimpleNamespace

from app.graph.llm import FINAL_ANSWER_TAG, LLMUnavailableError
from app.services.chat_service import ChatResult, ChatService


def _stream_event(kind: str, *, tags: list[str], content: str = "") -> dict:
    chunk = SimpleNamespace(content=content)
    return {"event": kind, "tags": tags, "data": {"chunk": chunk}}


class _FakeGraph:
    def __init__(self, events, final_state):
        self._events = events
        self._final_state = final_state
        self.calls: list[tuple[dict, dict]] = []

    async def astream_events(self, initial, *, config, version):
        self.calls.append((initial, config))
        assert version == "v2"
        for event in self._events:
            yield event

    async def aget_state(self, config):
        return SimpleNamespace(values=self._final_state)


class _ExplodingGraph:
    def __init__(self, exc):
        self._exc = exc

    async def astream_events(self, initial, *, config, version):
        # Need to yield something to keep this an async generator, then raise.
        if False:
            yield {}
        raise self._exc

    async def aget_state(self, config):
        raise AssertionError("aget_state should not be called when stream errored")


async def _collect(stream):
    return [chunk async for chunk in stream]


async def test_astream_ask_yields_token_then_result():
    events = [
        _stream_event("on_chat_model_start", tags=[FINAL_ANSWER_TAG]),
        _stream_event("on_chat_model_stream", tags=[FINAL_ANSWER_TAG], content="Здра"),
        _stream_event("on_chat_model_stream", tags=[FINAL_ANSWER_TAG], content="вствуйте."),
        _stream_event("on_chat_model_end", tags=[FINAL_ANSWER_TAG]),
    ]
    final_state = {
        "final_answer": "Здравствуйте.",
        "sources": [],
        "suggested_action": None,
        "route": "direct_answer",
        "route_reasoning": "greeting",
        "errors": [],
    }
    graph = _FakeGraph(events, final_state)
    service = ChatService(graph)

    chunks = await _collect(
        service.astream_ask(
            question="Привет",
            user_role="finance_manager",
            company_id=1,
            thread_id="t-1",
        )
    )

    kinds = [c.kind for c in chunks]
    assert kinds == ["reset", "token", "token", "result"]
    assert "".join(c.token for c in chunks if c.kind == "token") == "Здравствуйте."

    final = chunks[-1].result
    assert isinstance(final, ChatResult)
    assert final.answer == "Здравствуйте."
    assert final.route == "direct_answer"
    assert final.thread_id == "t-1"


async def test_astream_ask_ignores_untagged_events():
    """Events without the ``final_answer`` tag (supervisor, condense, …) are
    intermediate; they MUST NOT reach the UI as tokens."""
    events = [
        _stream_event("on_chat_model_start", tags=["supervisor"]),
        _stream_event(
            "on_chat_model_stream", tags=["supervisor"], content="this is routing thought"
        ),
        _stream_event("on_chat_model_start", tags=[FINAL_ANSWER_TAG]),
        _stream_event("on_chat_model_stream", tags=[FINAL_ANSWER_TAG], content="реальный ответ"),
    ]
    final_state = {"final_answer": "реальный ответ", "sources": [], "errors": []}
    graph = _FakeGraph(events, final_state)
    service = ChatService(graph)

    chunks = await _collect(
        service.astream_ask(
            question="?", user_role="finance_manager", company_id=1, thread_id="t-2"
        )
    )

    tokens = "".join(c.token for c in chunks if c.kind == "token")
    assert tokens == "реальный ответ"
    assert "routing thought" not in tokens


async def test_astream_ask_multiple_resets_for_route_both():
    """In route=both the specialists fire first, then Finalize-combined.
    Each new final-answer LLM run emits a ``reset`` so the UI clears the
    intermediate text and the last stream wins."""
    events = [
        _stream_event("on_chat_model_start", tags=[FINAL_ANSWER_TAG]),
        _stream_event("on_chat_model_stream", tags=[FINAL_ANSWER_TAG], content="SQL summary"),
        _stream_event("on_chat_model_start", tags=[FINAL_ANSWER_TAG]),
        _stream_event("on_chat_model_stream", tags=[FINAL_ANSWER_TAG], content="docs summary"),
        _stream_event("on_chat_model_start", tags=[FINAL_ANSWER_TAG]),
        _stream_event(
            "on_chat_model_stream", tags=[FINAL_ANSWER_TAG], content="combined synthesis"
        ),
    ]
    final_state = {
        "final_answer": "combined synthesis",
        "sources": [{"type": "sql"}, {"type": "doc"}],
        "route": "both",
        "errors": [],
    }
    graph = _FakeGraph(events, final_state)
    service = ChatService(graph)

    chunks = await _collect(
        service.astream_ask(
            question="сравни", user_role="finance_manager", company_id=1, thread_id="t-3"
        )
    )

    resets = [c for c in chunks if c.kind == "reset"]
    assert len(resets) == 3, "every new final-answer run should emit a reset"

    final = chunks[-1].result
    assert final is not None
    assert final.answer == "combined synthesis"
    assert final.route == "both"
    assert [s["type"] for s in final.sources] == ["sql", "doc"]


async def test_astream_ask_returns_fallback_result_on_llm_unavailable():
    graph = _ExplodingGraph(LLMUnavailableError("ollama refused"))
    service = ChatService(graph)

    chunks = await _collect(
        service.astream_ask(
            question="?", user_role="finance_manager", company_id=1, thread_id="t-err"
        )
    )

    assert len(chunks) == 1
    assert chunks[0].kind == "result"
    final = chunks[0].result
    assert final is not None
    assert "временно недоступен" in final.answer
    assert final.route == "error"
    assert final.errors and final.errors[0].startswith("llm_unavailable")


async def test_astream_ask_generates_thread_id_when_missing():
    events = [_stream_event("on_chat_model_start", tags=[FINAL_ANSWER_TAG])]
    final_state = {"final_answer": "ok", "sources": [], "errors": []}
    graph = _FakeGraph(events, final_state)
    service = ChatService(graph)

    chunks = await _collect(
        service.astream_ask(question="?", user_role="finance_manager", company_id=1)
    )

    final = chunks[-1].result
    assert final is not None and final.thread_id
    # And the same thread_id was passed into the graph config.
    assert graph.calls[0][1]["configurable"]["thread_id"] == final.thread_id
