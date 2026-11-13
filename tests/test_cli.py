"""Tests for llmtap.cli: parse_since, argument parsing, and the subcommands end-to-end."""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from llmtap import cli
from llmtap.models import Exchange, Usage

PY = sys.executable


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No colors, no user config, no stray LLMTAP_* env, and a private LLMTAP_HOME."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("COLUMNS", "160")
    for k in list(os.environ):
        if k.startswith("LLMTAP_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("LLMTAP_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------------------ parse_since

def test_parse_since_relative() -> None:
    now = 1_000_000.0
    assert cli.parse_since("24h", now=now) == now - 86400
    assert cli.parse_since("7d", now=now) == now - 7 * 86400
    assert cli.parse_since("30m", now=now) == now - 1800
    assert cli.parse_since("45s", now=now) == now - 45
    assert cli.parse_since("2w", now=now) == now - 14 * 86400
    assert cli.parse_since("1h30m", now=now) == now - 5400
    assert cli.parse_since("90", now=now) == now - 90
    assert cli.parse_since(" 1.5h ", now=now) == now - 5400


def test_parse_since_all_and_none() -> None:
    for v in (None, "all", "ALL", "", "0"):
        assert cli.parse_since(v) is None


def test_parse_since_today_and_dates() -> None:
    t = cli.parse_since("today")
    assert t is not None
    lt = time.localtime(t)
    assert (lt.tm_hour, lt.tm_min, lt.tm_sec) == (0, 0, 0)
    assert t <= time.time()
    y = cli.parse_since("yesterday")
    assert y is not None and 82_000 < t - y < 90_000
    d = cli.parse_since("2026-01-31")
    assert time.localtime(d)[:3] == (2026, 1, 31)


@pytest.mark.parametrize("bad", ["abc", "24x", "h", "1h junk", "-"])
def test_parse_since_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        cli.parse_since(bad)


def test_since_flag_rejects_garbage(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["stats", "--since", "bogus"]) == 2
    assert "invalid" in capsys.readouterr().err


# ------------------------------------------------------------------------------ arg parsing

def test_parse_serve_flags_map_to_config(tmp_path: Path) -> None:
    args = cli.parse_args([
        "serve", "--host", "0.0.0.0", "--port", "5050", "--db", str(tmp_path / "x.db"), "--cache", "--cache-ttl", "60",
        "--cassette", "c.jsonl", "--mode", "replay", "--allow-miss", "--guard", "block", "--pii",
        "--budget-daily", "5", "--budget-session", "1.5", "--loop-threshold", "3", "--loop-window", "30", "--quiet",
        "--timeout", "12", "--upstream", "local=http://localhost:9999/,openai",
    ])
    assert args.command == "serve"
    cfg = cli.make_config(args)
    assert (cfg.host, cfg.port, cfg.db_path) == ("0.0.0.0", 5050, str(tmp_path / "x.db"))
    assert cfg.cache is True and cfg.cache_ttl_s == 60
    assert (cfg.cassette, cfg.mode, cfg.allow_miss) == ("c.jsonl", "replay", True)
    assert (cfg.guard, cfg.guard_pii) == ("block", True)
    assert (cfg.budget_daily_usd, cfg.budget_session_usd) == (5.0, 1.5)
    assert (cfg.loop_threshold, cfg.loop_window_s) == (3, 30)
    assert cfg.quiet is True and cfg.upstream_timeout_s == 12.0
    assert cfg.upstreams["local"] == {"base": "http://localhost:9999", "style": "openai"}
    assert "anthropic" in cfg.upstreams


def test_unset_flags_keep_defaults_and_toml(tmp_path: Path) -> None:
    (tmp_path / "llmtap.toml").write_text('[llmtap]\nguard = "redact"\nport = 7777\n')
    args = cli.parse_args(["serve"])
    cfg = cli.make_config(args)
    assert cfg.guard == "redact" and cfg.port == 7777 and cfg.cache is False
    cfg = cli.make_config(cli.parse_args(["serve", "--guard", "off", "--no-cache"]))
    assert cfg.guard == "off" and cfg.cache is False


def test_config_flag(tmp_path: Path) -> None:
    p = tmp_path / "other.toml"
    p.write_text("loop_threshold = 9\n")
    cfg = cli.make_config(cli.parse_args(["serve", "--config", str(p)]))
    assert cfg.loop_threshold == 9


def test_global_db_flag_before_subcommand(tmp_path: Path) -> None:
    args = cli.parse_args(["--db", "a.db", "ls"])
    assert cli.make_config(args).db_path == "a.db"
    args = cli.parse_args(["--db", "a.db", "ls", "--db", "b.db"])
    assert cli.make_config(args).db_path == "b.db"


def test_parse_wrap_splits_on_double_dash() -> None:
    args = cli.parse_args(["wrap", "--cache", "--quiet", "--", "python3", "-c", "print(1)", "--cache", "--"])
    assert args.command == "wrap"
    assert args.cmd == ["python3", "-c", "print(1)", "--cache", "--"]
    assert args.cache is True and args.quiet is True


def test_parse_every_subcommand() -> None:
    cases: dict[str, list[str]] = {
        "serve": ["serve"],
        "wrap": ["wrap", "--", "true"],
        "env": ["env", "--port", "1234"],
        "stats": ["stats", "--since", "7d", "--by", "tag:feature", "--json"],
        "ls": ["ls", "-n", "5", "-q", "hello", "--source", "cache"],
        "show": ["show", "abcd1234", "--raw"],
        "tail": ["tail", "-n", "3"],
        "export": ["export", "--format", "csv", "-o", "out.csv", "--since", "all"],
        "prices": ["prices", "claude"],
        "scan": ["scan", "a.txt", "b.txt", "--pii"],
        "clear": ["clear", "--yes"],
        "version": ["version"],
        "demo": ["demo", "--port", "0", "--no-open", "--count", "3"],
    }
    assert set(cases) == set(cli.COMMANDS)
    for name, argv in cases.items():
        a = cli.parse_args(argv)
        assert a.command == name
    a = cli.parse_args(cases["stats"])
    assert a.by == "tag:feature" and a.json and a.since < time.time()
    a = cli.parse_args(cases["ls"])
    assert (a.limit, a.query, a.source) == (5, "hello", "cache")
    a = cli.parse_args(cases["show"])
    assert a.id == "abcd1234" and a.raw
    a = cli.parse_args(cases["export"])
    assert (a.format, a.output, a.since) == ("csv", "out.csv", None)
    a = cli.parse_args(cases["scan"])
    assert a.files == ["a.txt", "b.txt"] and a.pii
    a = cli.parse_args(cases["demo"])
    assert a.port == 0 and a.no_open and a.count == 3


def test_help_and_bad_args(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0
    assert "wrap" in capsys.readouterr().out
    assert cli.main([]) == 0
    assert cli.main(["nope"]) == 2
    assert cli.main(["serve", "--guard", "maybe"]) == 2


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    from llmtap import __version__
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == f"llmtap {__version__}"


def test_wrap_without_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["wrap"]) == 2
    assert "nothing to run" in capsys.readouterr().err


# ------------------------------------------------------------------------------ env

def test_env_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["env"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "export ANTHROPIC_BASE_URL=http://127.0.0.1:4141",
        "export OPENAI_BASE_URL=http://127.0.0.1:4141/v1",
        "export OPENAI_API_BASE=http://127.0.0.1:4141/v1",
        "export LLMTAP_URL=http://127.0.0.1:4141",
    ]
    assert cli.main(["env", "--port", "9000", "--shell", "fish"]) == 0
    out = capsys.readouterr().out
    assert "set -gx OPENAI_BASE_URL http://127.0.0.1:9000/v1" in out
    assert cli.main(["env", "--host", "0.0.0.0", "--port", "1"]) == 0
    assert "http://127.0.0.1:1" in capsys.readouterr().out


# ------------------------------------------------------------------------------ scan

FAKE_KEY = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0" * 4 + "-xyzAA"


def test_scan_file_with_secret(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "leak.py"
    f.write_text(f'print("hi")\nclient = Anthropic(api_key="{FAKE_KEY}")\naws = "AKIAIOSFODNN7EXAMPLE"\n')
    assert cli.main(["scan", str(f)]) == 1
    out = capsys.readouterr().out
    assert "anthropic_key" in out and "aws_access_key" in out
    assert f"{f}:2:" in out and f"{f}:3:" in out
    assert FAKE_KEY not in out  # previews are masked


def test_scan_clean_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "clean.txt"
    f.write_text("nothing to see here, just a task-list and a risk-sk-assessment\n")
    assert cli.main(["scan", str(f)]) == 0
    assert "no secrets" in capsys.readouterr().out


def test_scan_stdin_and_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("token: AKIAIOSFODNN7EXAMPLE\n"))
    assert cli.main(["scan", "-", "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data and data[0]["kind"] == "aws_access_key" and data[0]["file"] == "<stdin>" and data[0]["line"] == 1
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello world"))
    assert cli.main(["scan"]) == 0


def test_scan_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["scan", "/definitely/not/here.txt"]) == 2


def test_scan_pii_flag(tmp_path: Path) -> None:
    f = tmp_path / "pii.txt"
    f.write_text("contact me at jane.doe@example.com\n")
    assert cli.main(["scan", str(f)]) == 0
    assert cli.main(["scan", "--pii", str(f)]) == 1


# ------------------------------------------------------------------------------ store-backed commands

def _populate(db: str) -> list[Exchange]:
    from llmtap.store import Store
    now = time.time()
    exs = [
        Exchange(id="aaaa1111" + "0" * 24, ts=now - 3600 * 50, provider="anthropic", style="anthropic",
                 path="/v1/messages", model="claude-sonnet-4-5", status=200,
                 request_body=json.dumps({"model": "claude-sonnet-4-5", "system": "be terse",
                                          "messages": [{"role": "user", "content": "old question about pandas"}]}),
                 response_body="{}", output_text="old answer", usage=Usage(1000, 200), cost_usd=0.006,
                 latency_ms=900),
        Exchange(id="bbbb2222" + "0" * 24, ts=now - 60, provider="openai", path="/v1/chat/completions",
                 model="gpt-4o-mini", status=200,
                 request_body=json.dumps({"model": "gpt-4o-mini", "messages": [
                     {"role": "system", "content": "sys"},
                     {"role": "user", "content": [{"type": "text", "text": "summarize this doc"}]}]}),
                 response_body="{}", output_text="Echo: summarize this doc", usage=Usage(500, 50), cost_usd=0.0001,
                 latency_ms=300, streamed=True, ttft_ms=120, tags={"feature": "summary"}),
        Exchange(id="cccc3333" + "0" * 24, ts=now - 30, provider="openai", path="/v1/chat/completions",
                 model="gpt-4o-mini", status=200, source="cache", request_body='{"messages":[]}',
                 output_text="cached", usage=Usage(500, 50), cost_usd=0.0, saved_usd=0.0001, latency_ms=2),
        Exchange(id="dddd4444" + "0" * 24, ts=now - 10, provider="anthropic", style="anthropic",
                 path="/v1/messages", model="claude-haiku-4-5", status=400, source="blocked",
                 request_body='{"messages":[{"role":"user","content":"key AKIA..."}]}',
                 guard_findings=[{"kind": "aws_access_key", "path": "messages[0].content", "preview": "AKIA…MPLE",
                                  "start": 4, "end": 24}], error="blocked by guard"),
    ]
    s = Store(db)
    for e in exs:
        s.record(e)
    s.close()
    return exs


@pytest.fixture()
def db(tmp_path: Path) -> str:
    pytest.importorskip("llmtap.store")
    p = str(tmp_path / "test.db")
    _populate(p)
    return p


def test_stats_table(db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["stats", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "Requests" in out and "4" in out
    assert "claude-sonnet-4-5" in out and "gpt-4o-mini" in out
    assert "MODEL" in out and "SHARE" in out


def test_stats_since_and_by(db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["stats", "--db", db, "--since", "24h", "--by", "provider", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["requests"] == 3
    keys = {g["key"] for g in data["groups"]}
    assert keys == {"openai", "anthropic"}
    assert cli.main(["stats", "--db", db, "--by", "source"]) == 0
    out = capsys.readouterr().out
    assert "SOURCE" in out and "cache" in out and "blocked" in out
    assert cli.main(["stats", "--db", db, "--by", "tag:feature"]) == 0
    assert "summary" in capsys.readouterr().out


def test_stats_empty_db(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("llmtap.store")
    assert cli.main(["stats", "--db", str(tmp_path / "empty.db")]) == 0
    assert "No LLM requests" in capsys.readouterr().out


def test_ls(db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ls", "--db", db]) == 0
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.startswith("│")]
    assert len(lines) == 5  # header + 4 rows
    assert lines[1].startswith("│ dddd4444")  # newest first
    assert "summarize this doc" in out and "BLOCK" in out and "CACHE" in out
    assert cli.main(["ls", "--db", db, "-n", "1"]) == 0
    assert len([l for l in capsys.readouterr().out.splitlines() if l.startswith("│")]) == 2
    assert cli.main(["ls", "--db", db, "-q", "pandas"]) == 0
    out = capsys.readouterr().out
    assert "aaaa1111" in out and "bbbb2222" not in out
    assert cli.main(["ls", "--db", db, "--json", "--model", "gpt-4o-mini"]) == 0
    rows = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert {r["id"][:8] for r in rows} == {"bbbb2222", "cccc3333"}
    assert "request_body" not in rows[0]


def test_ls_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("llmtap.store")
    assert cli.main(["ls", "--db", str(tmp_path / "e.db")]) == 0
    assert "no exchanges" in capsys.readouterr().out


def test_show(db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["show", "bbbb", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "gpt-4o-mini" in out and "summarize this doc" in out and "Echo: summarize this doc" in out
    assert "ttft" in out and "feature=summary" in out
    assert "▍user" in out and "▍system" in out
    assert cli.main(["show", "aaaa1111", "--db", db, "--raw"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["model"] == "claude-sonnet-4-5" and d["usage"]["input_tokens"] == 1000
    assert cli.main(["show", "dddd", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "aws_access_key" in out and "blocked by guard" in out
    assert cli.main(["show", "zzzz9999", "--db", db]) == 1
    assert "no exchange" in capsys.readouterr().err


def test_export_jsonl(db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["export", "--db", db]) == 0
    rows = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert len(rows) == 4
    assert [r["id"][:4] for r in rows] == ["aaaa", "bbbb", "cccc", "dddd"]  # chronological
    assert "request_body" in rows[0]


def test_export_csv_to_file(db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out.csv"
    assert cli.main(["export", "--db", db, "--format", "csv", "-o", str(out), "--since", "24h"]) == 0
    assert "exported 3" in capsys.readouterr().err
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert set(cli.CSV_COLUMNS) == set(rows[0])
    r = {row["id"][:4]: row for row in rows}
    assert r["bbbb"]["input_tokens"] == "500" and r["bbbb"]["streamed"] == "1"
    assert json.loads(r["bbbb"]["tags"]) == {"feature": "summary"}
    assert r["dddd"]["guard_findings"] == "1" and r["dddd"]["source"] == "blocked"


def test_export_limit_paginates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from llmtap.store import Store
    db = str(tmp_path / "many.db")
    s = Store(db)
    for i in range(1203):
        s.record(Exchange(ts=1000.0 + i, model="m"))
    s.close()
    assert cli.main(["export", "--db", db]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 1203
    assert cli.main(["export", "--db", db, "-n", "700"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 700


def test_tail_once(db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["tail", "--db", db, "-n", "2", "--once"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2
    assert "CACHE" in out[0] and "BLOCK" in out[1]


def test_clear(db: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from llmtap.store import Store
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert cli.main(["clear", "--db", db]) == 1  # non-tty without --yes refuses
    assert Store(db).count() == 4
    assert cli.main(["clear", "--db", db, "--yes"]) == 0
    assert "cleared 4" in capsys.readouterr().out
    assert Store(db).count() == 0


def test_prices(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("llmtap.pricing")
    assert cli.main(["prices"]) == 0
    out = capsys.readouterr().out
    assert "MODEL" in out and "INPUT" in out and "gpt-4o" in out
    assert cli.main(["prices", "claude"]) == 0
    out = capsys.readouterr().out
    assert "claude" in out and "gpt-4o" not in out
    assert cli.main(["prices", "no-such-model-xyz"]) == 1
    capsys.readouterr()
    assert cli.main(["prices", "gpt-4o", "--json"]) == 0
    assert "gpt-4o" in json.loads(capsys.readouterr().out)


# ------------------------------------------------------------------------------ wrap (needs proxy.py)

def _need_proxy() -> None:
    pytest.importorskip("llmtap.proxy")


def test_wrap_sets_env_and_prints_summary(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    _need_proxy()
    db = str(tmp_path / "w.db")
    code = ("import os;print(os.environ['ANTHROPIC_BASE_URL']);print(os.environ['OPENAI_BASE_URL']);"
            "print(os.environ['LLMTAP_URL']);print(os.environ['OPENAI_API_BASE'])")
    rc = cli.main(["wrap", "--db", db, "--quiet", "--", PY, "-c", code])
    out, err = capfd.readouterr()
    assert rc == 0
    lines = out.strip().splitlines()
    assert lines[0].startswith("http://127.0.0.1:") and not lines[0].endswith(":4141")
    assert lines[1] == lines[0] + "/v1" and lines[2] == lines[0] and lines[3] == lines[0] + "/v1"
    assert "summary" in err


def test_wrap_propagates_exit_code(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    _need_proxy()
    rc = cli.main(["wrap", "--db", str(tmp_path / "w.db"), "--quiet", "--", PY, "-c", "raise SystemExit(3)"])
    assert rc == 3
    rc = cli.main(["wrap", "--db", str(tmp_path / "w.db"), "--quiet", "--", "definitely-not-a-command-xyz"])
    assert rc == 127


def test_wrap_records_traffic(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    """A child process that calls the proxy gets recorded and summarized."""
    _need_proxy()
    from llmtap.mock import MockUpstream
    from llmtap.store import Store
    mock = MockUpstream().start()
    try:
        db = str(tmp_path / "w.db")
        child = (
            "import os,json,urllib.request\n"
            "req=urllib.request.Request(os.environ['ANTHROPIC_BASE_URL']+'/v1/messages',"
            "data=json.dumps({'model':'claude-sonnet-4-5','max_tokens':10,"
            "'messages':[{'role':'user','content':'hello wrap'}]}).encode(),"
            "headers={'content-type':'application/json','x-api-key':'test','anthropic-version':'2023-06-01'})\n"
            "h=urllib.request.build_opener(urllib.request.ProxyHandler({}))\n"
            "print(json.loads(h.open(req).read())['content'][0]['text'])\n"
        )
        rc = cli.main(["wrap", "--db", db, "--quiet", "--upstream", f"anthropic={mock.base_url},anthropic",
                       "--", PY, "-c", child])
        out, err = capfd.readouterr()
        assert rc == 0, err
        assert "Echo: hello wrap" in out
        assert "claude-sonnet-4-5" in err and "Requests" in err
        s = Store(db)
        assert s.count() == 1
        s.close()
    finally:
        mock.stop()


def test_demo_exit(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    _need_proxy()
    pytest.importorskip("llmtap.demo")
    rc = cli.main(["demo", "--port", "0", "--no-open", "--count", "4", "--delay", "0", "--exit", "--quiet"])
    out, err = capfd.readouterr()
    assert rc == 0
    assert "llmtap v" in out and "/_llmtap/" in out
    assert "demo summary" in err
