"""pytest setup for LLM tests that run through an llmtap cassette.

Record (once, locally, real key):
    ANTHROPIC_API_KEY=sk-ant-... \
      llmtap wrap --cassette cassettes/llm.jsonl --mode record -- pytest -q

Replay (CI, offline, fake key, $0):
    ANTHROPIC_API_KEY=fake-key \
      llmtap wrap --cassette cassettes/llm.jsonl --mode replay -- pytest -q

No llmtap import is needed: `llmtap wrap` sets ANTHROPIC_BASE_URL, and the SDK reads it.
"""
from __future__ import annotations

import json
import os
import urllib.request

import pytest


def _llmtap_status() -> str:
    url = os.environ.get("LLMTAP_URL")
    if not url:
        return "not running (LLM tests will be skipped; run under `llmtap wrap --cassette ...`)"
    try:
        with urllib.request.urlopen(f"{url}/_llmtap/api/stats", timeout=2) as r:
            cassette = json.load(r).get("cassette") or {}
    except Exception:  # status is informational only
        cassette = {}
    if cassette:
        return f"{url} cassette={cassette.get('path')} mode={cassette.get('mode')} entries={cassette.get('entries')}"
    return f"{url} (no cassette: live calls)"


def pytest_report_header(config: pytest.Config) -> str:
    return f"llmtap: {_llmtap_status()}"


@pytest.fixture(scope="session")
def claude():
    """An Anthropic client that talks to llmtap.

    max_retries=0: the SDK retries HTTP 409, which is what llmtap returns on a cassette miss.
    Failing fast gives you a clear "cassette miss" error instead of a slow retry loop.
    """
    if not os.environ.get("LLMTAP_URL"):
        pytest.skip("LLM tests run under `llmtap wrap --cassette ...` (see conftest.py)")
    anthropic = pytest.importorskip("anthropic")
    return anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY") or "fake-key-for-replay",
        max_retries=0,
        default_headers={"X-LLMTap-Session": "pytest"},
    )
