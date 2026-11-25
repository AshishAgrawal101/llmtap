"""``llmtap statusline``: one short line of spend for an editor or agent status bar (Claude Code's ``statusLine``).

    llmtap $0.42 today · cache 81% · ctx 31k · last $0.03

It only reads the database (read-only, one query), so it is fast enough to run on every refresh, and it works
whether or not the proxy is running.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .budget import local_midnight

__all__ = ["statusline", "SETUP_SNIPPET"]

SETUP_SNIPPET = json.dumps({"statusLine": {"type": "command", "command": "llmtap statusline"}}, indent=2)

# input tokens a provider sent the model, cached ones included (Anthropic reports cache reads/writes apart)
_FULL_INPUT = ("CASE WHEN style IN ('anthropic', 'bedrock') THEN input_tokens + cache_read_tokens + cache_write_tokens "
               "ELSE input_tokens END")


def _usd(x: float) -> str:
    if x and abs(x) < 0.01:
        return f"${x:.3f}"
    return f"${x:,.2f}"


def _tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.0f}k" if n >= 10_000 else f"{n / 1000:.1f}k"
    return str(n)


def _paint(s: str, code: str, color: bool) -> str:
    return f"\033[{code}m{s}\033[0m" if color else s


def gather(db_path: str, now: float | None = None) -> dict[str, Any] | None:
    """Today's numbers from the database, or None when there is no database yet."""
    now = time.time() if now is None else now
    if db_path != ":memory:" and not Path(db_path).is_file():
        return None
    uri = Path(db_path).resolve().as_uri() + "?mode=ro" if db_path != ":memory:" else ":memory:"
    try:
        conn = sqlite3.connect(uri, uri=db_path != ":memory:", timeout=1.0)
    except sqlite3.Error:
        return None
    try:
        since = local_midnight(now)
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0), COALESCE(SUM(saved_usd), 0), "
            "COALESCE(SUM(cache_read_tokens), 0), "
            f"COALESCE(SUM(CASE WHEN source = 'upstream' THEN {_FULL_INPUT} ELSE 0 END), 0), "
            "COALESCE(SUM(CASE WHEN source IN ('blocked', 'budget', 'loop') AND ts >= ? THEN 1 ELSE 0 END), 0) "
            "FROM exchanges WHERE ts >= ?", (now - 600, since)).fetchone()
        last = conn.execute(
            f"SELECT {_FULL_INPUT}, cost_usd, model FROM exchanges WHERE source = 'upstream' AND status < 400 "
            "AND (input_tokens > 0 OR output_tokens > 0) ORDER BY ts DESC LIMIT 1").fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return {
        "requests": int(row[0]), "cost_usd": float(row[1]), "saved_usd": float(row[2]),
        "cache_read": int(row[3]), "full_input": int(row[4]), "blocked_recent": int(row[5]),
        "last_context": int(last[0]) if last else 0, "last_cost": float(last[1]) if last else 0.0,
        "last_model": (last[2] or "") if last else "",
    }


def statusline(db_path: str, budget_daily: float = 0.0, color: bool = True, now: float | None = None) -> str:
    d = gather(db_path, now)
    if d is None:
        return "llmtap: no calls yet"
    parts = [_usd(d["cost_usd"]) + " today"]
    if budget_daily:
        pct = d["cost_usd"] / budget_daily * 100
        parts[0] = f"{_usd(d['cost_usd'])}/{_usd(budget_daily)} today"
        if pct >= 95:
            parts[0] = _paint(parts[0], "31", color)
        elif pct >= 80:
            parts[0] = _paint(parts[0], "33", color)
    if d["full_input"] > 0:
        parts.append(f"cache {d['cache_read'] / d['full_input'] * 100:.0f}%")
    if d["last_context"]:
        parts.append(f"ctx {_tok(d['last_context'])}")
        parts.append(f"last {_usd(d['last_cost'])}")
    if d["saved_usd"] >= 0.01:
        parts.append(f"saved {_usd(d['saved_usd'])}")
    if d["blocked_recent"]:
        parts.append(_paint(f"{d['blocked_recent']} blocked", "31", color))
    return _paint("llmtap", "2", color) + " " + " · ".join(parts)
