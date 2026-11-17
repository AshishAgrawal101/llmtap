"""CLI integration for `report`, `doctor` and the new `stats` savings line (all offline)."""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import pytest

from llmtap import cli, doctor, __version__
from llmtap.models import Exchange, Usage
from llmtap.store import Store


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("COLUMNS", "160")
    for k in list(os.environ):
        if k.startswith("LLMTAP_") or k in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("LLMTAP_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def _ex(prompt: str, ts: float, cost: float, **kw: object) -> Exchange:
    body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}]})
    return Exchange(ts=ts, provider="openai", path="/v1/chat/completions", model="gpt-4o", request_body=body,
                    status=200, usage=Usage(input_tokens=100, output_tokens=20), cost_usd=cost, latency_ms=900.0, **kw)


@pytest.fixture()
def db(tmp_path: Path) -> str:
    p = str(tmp_path / "t.db")
    s = Store(p)
    now = time.time()
    for i in range(4):
        s.record(_ex("what is 2+2? <script>alert(1)</script>", now - 100 + i, 0.25))
    s.record(_ex("unique", now - 50, 0.5))
    s.record(_ex("old", now - 30 * 86400, 9.0))
    s.close()
    return p


# ------------------------------------------------------------------ report

def test_report_default_writes_html(db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["report", "--db", db]) == 0
    out = capsys.readouterr().out
    f = tmp_path / "llmtap-report.html"
    assert f.is_file() and "llmtap-report.html" in out and "$0.7500 avoidable waste" in out
    h = f.read_text()
    assert "<script" not in h.lower() and "&lt;script&gt;" in h
    assert "$9.00" not in h                       # default --since 7d excludes the old row


def test_report_since_all_and_output(db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    o = tmp_path / "r.html"
    assert cli.main(["report", "--db", db, "--since", "all", "-o", str(o)]) == 0
    assert "$9.00" in o.read_text() or "$10.75" in o.read_text()


def test_report_markdown_stdout_and_ext_inference(db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["report", "--db", db, "--format", "md"]) == 0
    md = capsys.readouterr().out
    assert md.startswith("## llmtap report") and "$0.7500 avoidable" in md
    o = tmp_path / "pr.md"
    assert cli.main(["report", "--db", db, "-o", str(o)]) == 0
    assert o.read_text().startswith("## llmtap report")


def test_report_json(db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["report", "--db", db, "-f", "json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["totals"]["requests"] == 5 and d["waste"]["duplicate_requests"] == 3
    assert d["waste"]["savings_usd"] == pytest.approx(0.75)


def test_report_html_to_stdout(db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["report", "--db", db, "-o", "-"]) == 0
    assert capsys.readouterr().out.startswith("<!doctype html>")


def test_report_empty_db_and_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import webbrowser
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda u, *a, **k: opened.append(u) or True)
    assert cli.main(["report", "--db", str(tmp_path / "e.db"), "-o", "e.html", "--open"]) == 0
    assert (tmp_path / "e.html").is_file() and opened and opened[0].startswith("file://")
    assert cli.main(["report", "--db", str(tmp_path / "e.db"), "-f", "md"]) == 0


def test_report_bad_since(db: str) -> None:
    assert cli.main(["report", "--db", db, "--since", "bogus"]) == 2


# ------------------------------------------------------------------ stats

def test_stats_savings_line(db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["stats", "--db", db, "--since", "7d"]) == 0
    out = capsys.readouterr().out
    assert "Potential savings from caching: $0.7500 (3 duplicate requests)" in out
    assert "MODEL" in out                      # existing table still there
    assert cli.main(["stats", "--db", db, "--since", "7d", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["potential_savings_usd"] == pytest.approx(0.75) and d["duplicate_requests"] == 3 and d["requests"] == 5


def test_stats_empty_has_no_savings_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["stats", "--db", str(tmp_path / "x.db")]) == 0
    out = capsys.readouterr().out
    assert "No LLM requests" in out and "Potential savings" not in out


# ------------------------------------------------------------------ doctor

@pytest.fixture()
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_port_in_use", lambda host, port: False)


def test_doctor_all_good(offline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SUPERSECRETVALUE123")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:4141")
    monkeypatch.setenv("LLMTAP_CACHE", "1")
    assert cli.main(["doctor", "--db", str(tmp_path / "d.db")]) == 0
    out = capsys.readouterr().out
    assert "✓ Python" in out and "✓ Port 4141 is free" in out and "Database writable" in out
    assert "ANTHROPIC_BASE_URL points at llmtap" in out
    assert "✓ ANTHROPIC_API_KEY set" in out and "OPENAI_API_KEY not set" in out
    assert "SUPERSECRETVALUE" not in out
    assert "LLMTAP_CACHE" in out and "SQLite" in out and "No config file found" in out
    assert "OPENAI_BASE_URL not set" in out and 'eval "$(llmtap env)"' in out


def test_doctor_port_busy_is_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(doctor, "_port_in_use", lambda host, port: True)
    monkeypatch.setattr(doctor, "_is_llmtap", lambda host, port: None)
    monkeypatch.setattr(doctor, "_listener_info", lambda port: "users:((\"nginx\",pid=1))")
    assert cli.main(["doctor", "--db", str(tmp_path / "d.db"), "--port", "4141"]) == 0
    out = capsys.readouterr().out
    assert "! Port 4141 is in use (users:" in out and "--port 4142" in out


def test_doctor_detects_running_llmtap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(doctor, "_port_in_use", lambda host, port: True)
    monkeypatch.setattr(doctor, "_is_llmtap", lambda host, port: "9.9.9")
    assert cli.main(["doctor", "--db", str(tmp_path / "d.db")]) == 0
    assert "llmtap v9.9.9 is already running" in capsys.readouterr().out


def test_doctor_real_port_check() -> None:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert doctor._port_in_use("127.0.0.1", port) is True
    finally:
        s.close()
    assert doctor._port_in_use("127.0.0.1", port) is False


def test_doctor_hard_failures(offline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(os, "access", lambda *a, **k: False)
    assert cli.main(["doctor", "--db", str(tmp_path / "ro" / "d.db")]) == 1
    out = capsys.readouterr().out
    assert "✗ Database path not writable" in out and "problem" in out


def test_doctor_bad_python_and_bad_config(offline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import sys
    from collections import namedtuple
    VI = namedtuple("VI", "major minor micro releaselevel serial")
    monkeypatch.setattr(sys, "version_info", VI(3, 9, 0, "final", 0))
    cfgf = tmp_path / "bad.toml"
    cfgf.write_text("this is = = not toml")
    rc = cli.main(["doctor", "--db", str(tmp_path / "d.db"), "--config", str(cfgf)]) if False else None
    chk = doctor.check_python()
    assert chk.level == doctor.FAIL and "3.11" in chk.detail
    assert doctor.check_config(str(cfgf)).level == doctor.FAIL
    assert doctor.check_config(str(tmp_path / "missing.toml")).level == doctor.FAIL
    good = tmp_path / "ok.toml"
    good.write_text("[llmtap]\nport = 5000\n")
    assert doctor.check_config(str(good)).level == doctor.OK


def test_doctor_base_url_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    from llmtap.config import Config
    cfg = Config()
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    c = doctor._base_url_check("OPENAI_BASE_URL", cfg)
    assert c.level == doctor.WARN and "bypasses llmtap" in c.title
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:4141/v1")
    assert doctor._base_url_check("OPENAI_BASE_URL", cfg).level == doctor.OK
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9999/v1")
    assert doctor._base_url_check("OPENAI_BASE_URL", cfg).level == doctor.WARN


def test_doctor_sqlite_note(monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.40.1")
    c = doctor.check_sqlite()
    assert c.level == doctor.OK and "3.45" in c.detail
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.50.0")
    assert doctor.check_sqlite().detail == ""


def test_doctor_online_monkeypatched(offline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake(url: str, timeout: float = 5.0) -> tuple[bool, str]:
        assert timeout == 5.0
        return ("anthropic" in url or "openai.com" in url, "HTTP 405" if "anthropic" in url or "openai.com" in url else "timed out")
    monkeypatch.setattr(doctor, "probe_host", fake)
    assert cli.main(["doctor", "--online", "--db", str(tmp_path / "d.db")]) == 0     # unreachable = warning only
    out = capsys.readouterr().out
    assert "anthropic reachable (api.anthropic.com)" in out and "HTTP 405" in out
    assert "! groq unreachable (api.groq.com)" in out and "timed out" in out


def test_doctor_no_network_without_online(offline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("network used")
    monkeypatch.setattr(doctor, "probe_host", boom)
    assert cli.main(["doctor", "--db", str(tmp_path / "d.db")]) == 0


def test_new_commands_listed_in_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "report" in out and "doctor" in out


@pytest.mark.parametrize("argv", [["prices", "claude"], ["doctor"], ["scan", "-"]])
def test_cli_survives_non_unicode_output_encoding(argv: list[str], tmp_path: Path) -> None:
    """Windows redirects output in the ANSI code page (cp1252); tables and check marks must not crash."""
    import subprocess
    import sys

    env = dict(os.environ, PYTHONIOENCODING="cp1252", LLMTAP_HOME=str(tmp_path / "home"))
    p = subprocess.run([sys.executable, "-m", "llmtap", *argv], input=b"nothing secret here\n",
                       capture_output=True, env=env, timeout=60)
    assert b"codec can't encode" not in p.stderr, p.stderr
    assert p.returncode in (0, 1), p.stderr  # doctor may report setup problems (1); 2 is the crash path
    assert p.stdout
