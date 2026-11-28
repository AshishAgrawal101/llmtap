"""``llmtap whatif``: the CLI over llmtap/whatif.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmtap.cli import main
from test_xray_cli import _db


def test_whatif_cli(tmp_path: Path, capsys: Any) -> None:
    db = _db(tmp_path)
    assert main(["whatif", "--db", db, "--since", "all"]) == 0
    out = capsys.readouterr().out
    assert "what if" in out and "claude-sonnet-4-5" in out
    assert main(["whatif", "claude-haiku-5-5", "--db", db, "--since", "all", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["compare"][0]["target"] == "claude-haiku-5-5" and "cache" in data


def test_whatif_empty(tmp_path: Path, capsys: Any) -> None:
    from llmtap.store import Store
    db = str(tmp_path / "e.db")
    Store(db).close()
    assert main(["whatif", "--db", db]) == 0 and "no calls" in capsys.readouterr().out
