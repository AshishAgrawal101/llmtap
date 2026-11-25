"""``llmtap xray``: the CLI over llmtap/xray.py."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from llmtap.cli import main
from llmtap.models import Exchange, Usage
from llmtap.store import Store


def _body(system: str) -> str:
    return json.dumps({
        "model": "claude-sonnet-4-5", "max_tokens": 100, "system": system,
        "tools": [{"name": "mcp__github__create_issue", "description": "Create an issue. " * 40,
                   "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}}},
                  {"name": "Read", "description": "Read a file. " * 20, "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "fix the bug"}]})


def _db(tmp_path: Path) -> str:
    db = str(tmp_path / "x.db")
    s = Store(db)
    now = time.time() - 120
    s.record(Exchange(id="aaaa1111", ts=now, provider="anthropic", style="anthropic", model="claude-sonnet-4-5",
                      status=200, request_body=_body("You are helpful. Date: 10:31"),
                      usage=Usage(input_tokens=20, cache_write_tokens=900, output_tokens=10)))
    s.record(Exchange(id="bbbb2222", ts=now + 30, provider="anthropic", style="anthropic", model="claude-sonnet-4-5",
                      status=200, request_body=_body("You are helpful. Date: 10:32"),
                      usage=Usage(input_tokens=20, cache_write_tokens=900, output_tokens=10)))
    s.close()
    return db


def test_xray_one_and_summary(tmp_path: Path, capsys: Any) -> None:
    db = _db(tmp_path)
    assert main(["xray", "bbbb2222", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "tools" in out and "mcp__github__create_issue" in out and "cache" in out.lower()
    assert main(["xray", "bbbb", "--db", db, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["breakdown"]["parts"]["tools"]["tokens"] > 0 and data["cache"] is not None
    assert main(["xray", "--db", db, "--since", "all"]) == 0
    assert "tool definitions" in capsys.readouterr().out
    assert main(["xray", "--db", db, "--since", "all", "--json"]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), dict)


def test_xray_empty_and_missing(tmp_path: Path, capsys: Any) -> None:
    db = str(tmp_path / "empty.db")
    Store(db).close()
    assert main(["xray", "--db", db]) == 0 and "no calls" in capsys.readouterr().out
    assert main(["xray", "zzzz9999", "--db", db]) == 1
