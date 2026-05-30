"""Pytest-wide fixtures."""

import logging
from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def _silence_langfuse_shutdown_logging() -> Iterator[None]:
    """Mute Langfuse's atexit logging so it can't write to closed pytest streams.

    With ``LANGFUSE_DEBUG=true`` the client attaches a StreamHandler and logs at
    DEBUG from background consumer threads it tears down in an ``atexit`` hook.
    That hook fires *after* pytest closes its captured stdout/stderr, so every
    debug call raises "I/O operation on closed file" (cosmetic, post-results).
    We can't reorder atexit vs pytest teardown, so at session end we drop the
    ``langfuse`` logger below CRITICAL and detach its handlers — records are then
    filtered before any (closed) handler sees them.
    """
    yield
    lf = logging.getLogger("langfuse")
    lf.handlers.clear()
    lf.propagate = False
    lf.setLevel(logging.CRITICAL)
