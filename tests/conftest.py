"""Shared fixtures: a running MockUpstream and a factory for LLMTap proxies pointed at it."""
from __future__ import annotations

from typing import Any, Callable, Iterator

import pytest

from llmtap.config import Config
from llmtap.mock import MockUpstream


@pytest.fixture()
def mock_upstream() -> Iterator[MockUpstream]:
    """A started MockUpstream on 127.0.0.1:<ephemeral>, stopped after the test."""
    m = MockUpstream().start()
    try:
        yield m
    finally:
        m.stop()


@pytest.fixture()
def tap(mock_upstream: MockUpstream) -> Iterator[Callable[..., Any]]:
    """Factory: ``tap(**config_overrides) -> started LLMTap`` with anthropic+openai upstreams on the mock.

    Pass ``upstream_url=...`` to point at something else. Every tap built is shut down after the test.
    """
    from llmtap.proxy import LLMTap  # lazy: keep collection working while proxy.py is in flux

    built: list[Any] = []

    def make(upstream_url: str | None = None, **overrides: Any) -> Any:
        base = (upstream_url or mock_upstream.base_url).rstrip("/")
        upstreams = overrides.pop("upstreams", None)
        cfg = Config(port=0, db_path=":memory:", quiet=True, **overrides)
        cfg.upstreams["anthropic"] = {"base": base, "style": "anthropic"}
        cfg.upstreams["openai"] = {"base": base, "style": "openai"}
        if upstreams:
            cfg.upstreams.update(upstreams)
        t = LLMTap(cfg).start_background()
        built.append(t)
        return t

    try:
        yield make
    finally:
        for t in reversed(built):
            try:
                t.shutdown()
            except Exception:
                pass
