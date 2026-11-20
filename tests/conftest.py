"""Shared fixtures: a running MockUpstream and a factory for LLMTap proxies pointed at it."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from llmtap.config import Config
from llmtap.mock import MockUpstream
from llmtap.vault import KeyInfo, MemoryBackend, Vault


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

# ---------------------------------------------------------------- key vault (tests/test_vault*.py)

class _MockVault(Vault):
    """Keys default to the mock upstream's URL (the tests' proxy sends every provider there)."""
    mock_base = ""

    def add(self, name: str, provider: str, secret: str, base: str | None = None, **limits: Any) -> KeyInfo:
        return super().add(name, provider, secret, base=base or self.mock_base, **limits)


@pytest.fixture()
def vault(tmp_path: Path, mock_upstream: Any) -> Vault:
    v = _MockVault(home=tmp_path / "home", backend=MemoryBackend())
    v.mock_base = mock_upstream.base_url
    return v


@pytest.fixture()
def vtap(tap: Any, vault: Vault, mock_upstream: Any) -> Any:
    """A proxy (anthropic, openai and gemini upstreams on the mock) using a throwaway vault."""
    def make(**overrides: Any) -> Any:
        ups = {"gemini": {"base": mock_upstream.base_url, "style": "gemini"}}
        ups.update(overrides.pop("upstreams", {}))
        t = tap(upstreams=ups, **overrides)
        t._vault = vault
        return t
    return make
