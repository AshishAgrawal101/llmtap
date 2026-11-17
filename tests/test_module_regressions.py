"""Regression tests for bugs in demo / mock / doctor / term / report and `llmtap wrap`."""
from __future__ import annotations

import json
import os
import signal
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from llmtap import cli, demo, doctor, report, term
from llmtap.config import Config
from llmtap.models import Exchange, Usage
from llmtap.store import Store


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("COLUMNS", "160")
    for k in list(os.environ):
        if k.startswith("LLMTAP_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("LLMTAP_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)


def _ex(prompt: str, ts: float) -> Exchange:
    body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}]})
    return Exchange(ts=ts, provider="openai", path="/v1/chat/completions", model="gpt-4o", request_body=body,
                    status=200, usage=Usage(10, 5), cost_usd=0.01)


def _no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route plain-HTTP traffic to a dead proxy, as a corporate HTTP_PROXY without NO_PROXY would."""
    for k in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setattr(urllib.request, "_opener", None)  # urlopen() caches the proxies it saw first


# ------------------------------------------------------------------ demo

def test_demo_ignores_project_cassette_and_budgets(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    """`llmtap demo` in a project with a llmtap.toml must not write fake traffic into its cassette."""
    (tmp_path / "llmtap.toml").write_text('cassette = "tape.jsonl"\nbudget_session_usd = 0.000001\n', encoding="utf-8")
    rc = cli.main(["demo", "--port", "0", "--no-open", "--count", "3", "--delay", "0", "--exit", "--quiet"])
    _out, err = capfd.readouterr()
    assert rc == 0
    assert not (tmp_path / "tape.jsonl").exists()
    assert "Blocked" not in err  # no budget refusals: the demo traffic all went through


def test_demo_send_ignores_http_proxy(tap: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Demo traffic goes to a local proxy; an HTTP_PROXY setting must not swallow it."""
    _no_proxy_env(monkeypatch)
    t = tap()
    first, _pool = demo._script()
    status, _body = demo.send(t.base_url, first[2])
    assert status == 200
