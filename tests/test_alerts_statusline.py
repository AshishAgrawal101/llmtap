"""Budget warnings (50/80/95%) and ``llmtap statusline``."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from llmtap.alerts import Alerts, parse_thresholds
from llmtap.models import Exchange, Usage
from llmtap.statusline import statusline
from llmtap.store import Store
from test_vault import _openai, _wait_rows, OPENAI_KEY


def test_parse_thresholds() -> None:
    assert parse_thresholds("95, 50,80%") == [50.0, 80.0, 95.0]
    assert parse_thresholds("") == []
    with pytest.raises(ValueError):
        parse_thresholds("120")
    with pytest.raises(ValueError):
        parse_thresholds("abc")


def test_each_threshold_once_per_window() -> None:
    said: list[str] = []
    a = Alerts([50, 80, 95], emit=lambda t, m, d: said.append(m))
    assert a.check([("daily", "daily budget", 1.0, 5.0, 0.0)]) == []
    assert len(a.check([("daily", "daily budget", 2.6, 5.0, 0.0)])) == 1          # 52%
    assert a.check([("daily", "daily budget", 3.0, 5.0, 0.0)]) == []               # still 60%: already said
    out = a.check([("daily", "daily budget", 4.9, 5.0, 0.0)])                      # jumped past 80 and 95
    assert len(out) == 1 and "95%" in out[0] and "$4.90 of $5.00" in out[0]
    assert a.check([("daily", "daily budget", 5.0, 5.0, 0.0)]) == []               # 100%: the cap speaks
    assert len(a.check([("daily", "daily budget", 2.6, 5.0, 86400.0)])) == 1       # a new day starts over
    assert len(said) == 3


def test_proxy_warns_on_stored_key_cap(vtap: Any, vault: Any) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    t = vtap(budget_warn="50")
    assert _openai(t)[0] == 200
    _wait_rows(t, 1)
    cost = t.store.spent_by_tag("vault_key", "openai", 0)
    assert cost > 0
    vault.update("openai", daily_usd=round(cost * 3.5, 6))   # after the next call: 2 calls = ~57% of the cap
    events = t.bus.subscribe()
    assert _openai(t)[0] == 200
    deadline = time.time() + 5
    warnings: list[dict] = []
    while time.time() < deadline and not warnings:
        try:
            e = events.get(timeout=0.2)
        except Exception:
            continue
        if e.get("type") == "warning":
            warnings.append(e)
    assert warnings and "stored key 'openai' daily cap" in warnings[0]["message"]


def test_daily_budget_warning_published(tap: Any) -> None:
    t = tap(budget_daily_usd=1000.0, budget_warn="50")
    events = t.bus.subscribe()
    ex = Exchange(cost_usd=600.0, source="upstream", provider="openai")
    t.store.record(ex)
    t.budget.add_spend(0)                                 # daily spend is read from the store
    t.alerts.check(Alerts.caps_for(t, ex))
    got = [events.get(timeout=1) for _ in range(1)]
    assert got[0]["type"] == "warning" and got[0]["cap"] == "daily"


def _row(**kw: Any) -> Exchange:
    u = kw.pop("usage", Usage())
    return Exchange(usage=u, **kw)


def test_statusline(tmp_path: Path) -> None:
    db = str(tmp_path / "s.db")
    assert statusline(db, color=False) == "llmtap: no calls yet"
    s = Store(db)
    now = time.time()
    s.record(_row(ts=now - 100, style="anthropic", provider="anthropic", status=200, cost_usd=0.30, model="c",
                  usage=Usage(input_tokens=1000, cache_read_tokens=8000, cache_write_tokens=1000, output_tokens=50)))
    s.record(_row(ts=now - 50, style="openai", provider="openai", status=200, cost_usd=0.12, model="g",
                  usage=Usage(input_tokens=30000, output_tokens=10)))
    s.record(_row(ts=now - 10, source="blocked", status=403))
    s.record(_row(ts=now - 86400 * 3, cost_usd=50.0, status=200, usage=Usage(input_tokens=5)))  # an older day
    s.close()
    line = statusline(db, color=False, now=now)
    assert line.startswith("llmtap $0.42 today")
    assert "cache 20%" in line                         # 8k cached of 40k sent today
    assert "ctx 30k" in line and "last $0.12" in line and "1 blocked" in line
    assert "$0.42/$0.50 today" in statusline(db, budget_daily=0.5, color=False, now=now)
    assert "\033[31m" in statusline(db, budget_daily=0.43, color=True, now=now)


def test_statusline_cli(tmp_path: Path, capsys: Any) -> None:
    from llmtap.cli import main
    assert main(["statusline", "--db", str(tmp_path / "none.db"), "--no-color"]) == 0
    assert capsys.readouterr().out.strip() == "llmtap: no calls yet"
    assert main(["statusline", "--setup"]) == 0
    assert '"command": "llmtap statusline"' in capsys.readouterr().out
