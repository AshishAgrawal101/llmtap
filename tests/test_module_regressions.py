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


# ------------------------------------------------------------------ wrap

def test_wrap_exec_format_error_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capfd: pytest.CaptureFixture[str]) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise OSError(8, "Exec format error")

    monkeypatch.setattr(cli.subprocess, "Popen", boom)
    rc = cli.main(["wrap", "--db", str(tmp_path / "w.db"), "--quiet", "--", "./not-a-program.txt"])
    _out, err = capfd.readouterr()
    assert rc == 126
    assert "Exec format error" in err and "Traceback" not in err


def test_wrap_signal_forwarding_survives_unsupported_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            capfd: pytest.CaptureFixture[str]) -> None:
    """On Windows Popen.send_signal(SIGINT) raises ValueError; that must not escape the forwarding thread."""
    errors: list[BaseException | None] = []
    monkeypatch.setattr(threading, "excepthook", lambda a: errors.append(a.exc_value))

    class SlowToExit:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def poll(self) -> None:
            return None

        def send_signal(self, sig: int) -> None:
            raise ValueError(f"Unsupported signal: {sig}")

        def wait(self) -> int:
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)  # what Ctrl-C does to llmtap
            time.sleep(0.8)  # the child is still busy when the forwarder fires
            return 0

    monkeypatch.setattr(cli.subprocess, "Popen", SlowToExit)
    rc = cli.main(["wrap", "--db", str(tmp_path / "w.db"), "--quiet", "--", "child"])
    capfd.readouterr()
    assert rc == 0
    assert errors == []


def test_wrap_keeps_users_no_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                   capfd: pytest.CaptureFixture[str]) -> None:
    """Python, curl & co. prefer lowercase no_proxy; it must still contain the user's own NO_PROXY hosts."""
    seen: dict[str, str] = {}

    class Done:
        def __init__(self, _cmd: Any, env: dict[str, str] | None = None, **_k: Any) -> None:
            seen.update(env or {})

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setenv("NO_PROXY", "corp.example")
    monkeypatch.setattr(cli.subprocess, "Popen", Done)
    assert cli.main(["wrap", "--db", str(tmp_path / "w.db"), "--quiet", "--", "child"]) == 0
    capfd.readouterr()
    values = [v for k, v in seen.items() if k.lower() == "no_proxy"]
    assert values
    for v in values:
        assert "corp.example" in v.split(",") and "127.0.0.1" in v.split(",")


# ------------------------------------------------------------------ doctor

def test_doctor_health_probe_brackets_ipv6(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def fake_open(self: Any, url: Any, data: Any = None, timeout: Any = None) -> Any:
        urls.append(url if isinstance(url, str) else url.full_url)
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(doctor, "_port_in_use", lambda h, p: True)
    monkeypatch.setattr(doctor, "_listener_info", lambda p: "")
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_open)
    doctor.check_port("::1", 4141)
    assert urls == ["http://[::1]:4141/_llmtap/health"]

    monkeypatch.setattr(doctor, "_is_llmtap", lambda h, p: "9.9")
    c = doctor.check_port("::1", 4141)
    assert "http://[::1]:4141/_llmtap/" in c.detail


def test_doctor_detects_running_llmtap_despite_http_proxy(tap: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_proxy_env(monkeypatch)
    t = tap()
    c = doctor.check_port("127.0.0.1", t.port)
    assert c.level == doctor.OK and "already running" in c.title


def test_doctor_db_path_with_hash(tmp_path: Path) -> None:
    p = tmp_path / "a#b" / "llmtap.db"
    s = Store(str(p))
    s.record(_ex("hi", time.time()))
    s.close()
    c = doctor.check_db(Config(db_path=str(p)))
    assert c.level == doctor.OK, c
    assert "1 exchanges recorded" in c.detail
