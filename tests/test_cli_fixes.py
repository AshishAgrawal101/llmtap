"""Regression tests for CLI bugs found by a tester (export/tail/doctor/ls/show/scan/clear/env/--since)."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from llmtap import cli, doctor
from llmtap.models import Exchange, Usage
from llmtap.store import Store

ROOT = str(Path(__file__).resolve().parents[1])
EVIL = "hi\x1b]0;PWNED\x07\x1b[2J there"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("COLUMNS", "160")
    for k in list(os.environ):
        if k.startswith("LLMTAP_") or k in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("LLMTAP_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def _db(tmp_path: Path, *exs: Exchange) -> str:
    p = str(tmp_path / "t.db")
    s = Store(p)
    for e in exs:
        s.record(e)
    s.close()
    return p


def _ex(prompt: str = "hello", ts: float | None = None, **kw: object) -> Exchange:
    body = kw.pop("request_body", None) or json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}]})
    return Exchange(ts=time.time() if ts is None else ts, provider="openai", path="/v1/chat/completions",
                    model=kw.pop("model", "gpt-4o"), request_body=body, status=200, usage=Usage(10, 5), **kw)


def _run(*args: str, stdin: bytes | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    e = dict(os.environ, PYTHONPATH=ROOT, **(env or {}))
    return subprocess.run([sys.executable, "-m", "llmtap", *args], input=stdin, env=e, capture_output=True, timeout=60)


# ------------------------------------------------------------------ 1. CSV to stdout

def test_export_csv_stdout_has_no_doubled_cr(tmp_path: Path) -> None:
    db = _db(tmp_path, _ex("a"), _ex("b"))
    r = _run("export", "-f", "csv", "--db", db)
    assert r.returncode == 0, r.stderr
    assert b"\r\r\n" not in r.stdout
    assert r.stdout.count(b"\r\n") == 3  # header + 2 rows, RFC 4180 line endings


# ------------------------------------------------------------------ 2. tail follows by insertion order

def test_tail_prints_rows_with_old_start_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    now = time.time()
    db = _db(tmp_path, _ex("first", ts=now))
    late = _ex("slow request", ts=now - 3600)  # started an hour ago, recorded just now
    calls = {"n": 0}

    def fake_sleep(_s: float) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            s = Store(db)
            s.record(late)
            s.close()
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", fake_sleep)
    assert cli.main(["tail", "--db", db, "-n", "5", "--interval", "0"]) == 0
    out = capsys.readouterr().out
    assert out.count("openai/gpt-4o") == 2


# ------------------------------------------------------------------ 5/6. doctor with a broken config

def test_doctor_reports_invalid_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(doctor, "_port_in_use", lambda host, port: False)
    bad = tmp_path / "bad.toml"
    bad.write_text("this is = = not toml")
    assert cli.main(["doctor", "--db", str(tmp_path / "d.db"), "--config", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "Config file is invalid" in out and "Python" in out


def test_doctor_reports_bad_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                  capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(doctor, "_port_in_use", lambda host, port: False)
    bad = tmp_path / "bad.toml"
    bad.write_text('port = "abc"\n')
    assert cli.main(["doctor", "--db", str(tmp_path / "d.db"), "--config", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "Config file is invalid" in out and "port" in out


def test_doctor_reads_config_as_utf8(tmp_path: Path) -> None:
    p = tmp_path / "u.toml"
    p.write_bytes('# ā✓\nport = 5000\n'.encode("utf-8"))
    code = "import sys; from llmtap import doctor; print(doctor.check_config(sys.argv[1]).level)"
    r = subprocess.run([sys.executable, "-c", code, str(p)], capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, PYTHONPATH=ROOT, PYTHONUTF8="0"))
    assert r.stdout.strip() == "ok", r.stdout + r.stderr


# ------------------------------------------------------------------ 7. malformed bodies

@pytest.mark.parametrize("body", [
    '{"messages": 5}',
    '{"messages": [{"role": "assistant", "content": "x", "tool_calls": [{"function": "x"}]}]}',
    '{"messages": [{"role": "assistant", "tool_calls": 5}]}',
    '{"messages": [{"role": "assistant", "tool_calls": ["x", {"function": {"name": "f"}}]}]}',
])
def test_ls_and_show_survive_malformed_bodies(tmp_path: Path, body: str, capsys: pytest.CaptureFixture[str]) -> None:
    e = _ex(request_body=body)
    db = _db(tmp_path, e)
    assert cli.main(["ls", "--db", db]) == 0
    assert cli.main(["show", e.id, "--db", db]) == 0
    capsys.readouterr()


# ------------------------------------------------------------------ 8. terminal escape injection

def _no_injection(out: str) -> None:
    assert "\x1b]" not in out and "\x1b[2J" not in out and "\x07" not in out


def test_show_and_ls_escape_control_sequences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    e = _ex(EVIL, model="gpt\x1b[2J", output_text="answer\x1b]0;X\x07", error="err\x1b[2J",
            tags={"k\x1b[2J": "v"}, guard_findings=[{"kind": "x", "path": "p", "preview": "AK\x1b[2J"}],
            response_body="")
    db = _db(tmp_path, e)
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    for args in (["ls"], ["show", e.id], ["tail", "--once"]):
        assert cli.main([*args, "--db", db]) == 0
        out = capsys.readouterr().out
        _no_injection(out)
        assert "\x1b[" in out  # our own colors still render
    assert cli.main(["show", e.id, "--db", db]) == 0
    assert "PWNED" in capsys.readouterr().out  # text is kept, just defanged


def test_show_raw_body_escaped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    e = _ex(request_body="not json \x1b]0;PWNED\x07", response_body="resp \x1b[2J", output_text="")
    db = _db(tmp_path, e)
    assert cli.main(["show", e.id, "--db", db]) == 0
    _no_injection(capsys.readouterr().out)


# ------------------------------------------------------------------ 10. small ones

@pytest.mark.parametrize("since", ["99999d", "inf", "1e12", "1969-06-01", "0001-01-01", "1e400"])
def test_extreme_since_is_clamped(tmp_path: Path, since: str, capsys: pytest.CaptureFixture[str]) -> None:
    db = _db(tmp_path, _ex("a"))
    assert cli.main(["stats", "--db", db, "--since", since]) == 0
    assert cli.main(["report", "--db", db, "--since", since, "-f", "md", "-o", "-"]) == 0
    capsys.readouterr()


@pytest.mark.parametrize("since", ["2026-13-45", "2026-02-30"])
def test_impossible_since_date_rejected(since: str) -> None:
    with pytest.raises(ValueError):
        cli.parse_since(since)


def test_nan_since_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError):
        cli.parse_since("nan")
    assert cli.main(["stats", "--since", "nan"]) == 2


def test_clear_eof_at_prompt_means_no(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    db = _db(tmp_path, _ex("a"))

    class TTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", TTY(""))
    assert cli.main(["clear", "--db", db]) == 1
    assert "aborted" in capsys.readouterr().out
    assert Store(db).count() == 1


@pytest.mark.parametrize("cmd", [["export"], ["report", "-f", "md"]])
def test_output_to_missing_dir_is_clean_error(tmp_path: Path, cmd: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    db = _db(tmp_path, _ex("a"))
    rc = cli.main([*cmd, "--db", db, "-o", str(tmp_path / "nodir" / "x.out")])
    assert rc == 1
    assert "llmtap: " in capsys.readouterr().err


def test_scan_stdin_non_locale_bytes(tmp_path: Path) -> None:
    data = "key AKIAIOSFODNN7EXAMPLE ā✓\n".encode("utf-8") + b"\x81\x8d\xff\n"
    r = _run("scan", "-", stdin=data, env={"PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"})
    assert r.returncode == 1, r.stderr  # 1 = secret found (not 2 = crash)
    assert b"aws" in r.stdout.lower()


def test_show_ambiguous_prefix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = _db(tmp_path, _ex("a", id="abcd1111" + "0" * 24), _ex("b", id="abcd2222" + "0" * 24))
    assert cli.main(["show", "abcd", "--db", db]) == 1
    err = capsys.readouterr().err
    assert "ambiguous" in err and "no exchange" not in err
    assert cli.main(["show", "zzzz", "--db", db]) == 1
    assert "no exchange" in capsys.readouterr().err


def test_env_brackets_ipv6(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["env", "--host", "::1", "--port", "4141"]) == 0
    out = capsys.readouterr().out
    assert "http://[::1]:4141" in out and "http://::1:" not in out


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-be", "utf-8-sig"])
def test_scan_file_with_bom(tmp_path: Path, encoding: str, capsys: pytest.CaptureFixture[str]) -> None:
    # Windows PowerShell 5.1 writes UTF-16LE (with BOM) for `echo ... > file`
    f = tmp_path / "out.txt"
    data = "first line\nkey AKIAIOSFODNN7EXAMPLE\n".encode(encoding)
    if encoding == "utf-16-be":
        data = b"\xfe\xff" + data
    f.write_bytes(data)
    assert cli.main(["scan", "--json", str(f)]) == 1
    found = json.loads(capsys.readouterr().out)
    assert [(d["kind"], d["line"], d["col"]) for d in found] == [("aws_access_key", 2, 5)]
