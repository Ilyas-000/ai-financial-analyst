"""Unit tests for the I-10 Langfuse observability factory.

We avoid hitting a live Langfuse server — the tests only verify:

* OFF-mode is a true no-op (no client, no callback, CM still composes).
* Empty credentials with the flag on degrade to OFF instead of crashing.
* ON-mode constructs a real ``LangchainCallbackHandler``.
* ``ChatService._build_config`` flips the ``callbacks`` key in lockstep.

``_init_langfuse_client`` and ``get_langfuse_callback`` are ``lru_cache``d at
module level, so every test that flips settings clears both caches first.
"""

from collections.abc import Iterator

import pytest

from app.observability import langfuse_handler


def _reset_caches() -> None:
    langfuse_handler._init_langfuse_client.cache_clear()
    langfuse_handler.get_langfuse_callback.cache_clear()


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    _reset_caches()
    yield
    _reset_caches()


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    """Patch ``get_settings`` inside the handler module with a frozen object.

    The real ``get_settings()`` is itself cached via ``lru_cache``; we patch
    the binding the handler imported so we don't have to mutate process env.
    """
    from app.config import Settings

    settings = Settings(**overrides)
    monkeypatch.setattr(langfuse_handler, "get_settings", lambda: settings)


def test_disabled_returns_no_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, enable_langfuse=False)
    assert langfuse_handler.get_langfuse_callback() is None


def test_disabled_context_manager_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, enable_langfuse=False)
    with langfuse_handler.langfuse_trace_attributes(
        thread_id="t", user_role="cfo", company_id=1
    ):
        # Just composing the CM should not raise or require Langfuse imports.
        pass


def test_enabled_without_keys_degrades_to_disabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_settings(
        monkeypatch,
        enable_langfuse=True,
        langfuse_public_key="",
        langfuse_secret_key="",
    )
    with caplog.at_level("WARNING", logger=langfuse_handler.logger.name):
        assert langfuse_handler.get_langfuse_callback() is None
    assert any("LANGFUSE_PUBLIC_KEY" in r.message for r in caplog.records)


def test_enabled_with_keys_returns_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(
        monkeypatch,
        enable_langfuse=True,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="http://localhost:3000",
    )
    handler = langfuse_handler.get_langfuse_callback()
    assert handler is not None
    # We don't want to assert on the concrete class path (it may shift between
    # langfuse minors) — just confirm it lives under the langfuse package and
    # behaves like a LangChain callback (has the documented hooks).
    assert type(handler).__module__.startswith("langfuse.")
    assert hasattr(handler, "on_llm_start")
    assert hasattr(handler, "on_llm_end")


def test_enabled_context_manager_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(
        monkeypatch,
        enable_langfuse=True,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    with langfuse_handler.langfuse_trace_attributes(
        thread_id="t-1", user_role="finance_manager", company_id=42
    ):
        pass


def test_chat_service_build_config_omits_callbacks_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.chat_service import ChatService

    _patch_settings(monkeypatch, enable_langfuse=False)
    cfg = ChatService(graph=object())._build_config("t-x")
    assert cfg == {"configurable": {"thread_id": "t-x"}}


def test_chat_service_build_config_attaches_callback_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.chat_service import ChatService

    _patch_settings(
        monkeypatch,
        enable_langfuse=True,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    cfg = ChatService(graph=object())._build_config("t-y")
    assert cfg["configurable"] == {"thread_id": "t-y"}
    callbacks = cfg["callbacks"]
    assert len(callbacks) == 1
    assert type(callbacks[0]).__module__.startswith("langfuse.")
