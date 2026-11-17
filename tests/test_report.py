"""Tests for llmtap.report: exact aggregation numbers, waste finder, rendering, XSS safety."""
from __future__ import annotations

import json
import re
import time
from typing import Any

import pytest

from llmtap import __version__, report
from llmtap.models import Exchange, Usage
from llmtap.store import Store

DAY = 86400.0
BASE = time.mktime((2026, 3, 10, 12, 0, 0, 0, 0, -1))   # local noon, 2026-03-10
NOW = BASE + 3600


def body(prompt: str, model: str = "gpt-4o", **extra: Any) -> str:
    return json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], **extra})


def ex(prompt: str = "hi", *, ts: float = BASE, cost: float = 0.0, model: str | None = "gpt-4o", provider: str = "openai",
       source: str = "upstream", status: int = 200, tin: int = 100, tout: int = 50, saved: float = 0.0,
       tags: dict | None = None, findings: list | None = None, latency: float = 1000.0, ttft: float | None = None,
       raw_body: str | None = None, path: str = "/v1/chat/completions") -> Exchange:
    return Exchange(ts=ts, provider=provider, path=path, model=model,
                    request_body=raw_body if raw_body is not None else body(prompt, model or "x"),
                    status=status, usage=Usage(input_tokens=tin, output_tokens=tout), cost_usd=cost, saved_usd=saved,
                    source=source, tags=tags or {}, guard_findings=findings or [], latency_ms=latency, ttft_ms=ttft)


@pytest.fixture()
def store() -> Store:
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture()
def crafted(store: Store) -> Store:
    """Day -2: A x3 (0.10, 0.10, 0.12); Day -1: B x2 (1.0, 1.0), C once (0.5), cache hit of A, 500 on C;
    Day 0: loop-blocked A x2, guard findings, blocked request."""
    d2, d1, d0 = BASE - 2 * DAY, BASE - DAY, BASE
    for i, c in enumerate((0.10, 0.10, 0.12)):
        store.record(ex("summarize the incident report", ts=d2 + i * 10, cost=c, model="claude-sonnet-4-5",
                        provider="anthropic", path="/v1/messages", tags={"feature": "summary"}, latency=2000 + i,
                        ttft=300.0 + i * 100))
    for i in range(2):
        store.record(ex("translate this to French", ts=d1 + i, cost=1.0, tags={"feature": "translate"}, latency=3000))
    store.record(ex("one-off question", ts=d1 + 5, cost=0.5, model="gpt-4o-mini", latency=500))
    store.record(ex("summarize the incident report", ts=d1 + 6, source="cache", saved=0.10, model="claude-sonnet-4-5",
                    provider="anthropic", path="/v1/messages", latency=2))
    store.record(ex("one-off question", ts=d1 + 7, status=500, model="gpt-4o-mini", latency=100))   # error: not waste
    for i in range(2):
        store.record(ex("summarize the incident report", ts=d0 + i, source="loop", status=429, model="claude-sonnet-4-5",
                        provider="anthropic", path="/v1/messages"))
    fnd = [{"kind": "aws_access_key", "path": "messages[0].content", "preview": "AKIA…MPLE", "start": 0, "end": 20},
           {"kind": "email", "path": "messages[0].content", "preview": "j***@acme.com", "start": 0, "end": 5}]
    store.record(ex("my key AKIAIOSFODNN7EXAMPLE", ts=d0 + 10, cost=0.02, findings=fnd))
    store.record(ex("another AKIA leak", ts=d0 + 11, source="blocked", status=400, findings=fnd[:1]))
    return store


# ------------------------------------------------------------------ helpers

def test_percentile() -> None:
    assert report.percentile([], 50) is None
    assert report.percentile([7], 95) == 7
    assert report.percentile([1, 2, 3, 4], 50) == 2.5
    assert report.percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48.0)
    assert report.percentile([5, 1, 3], 0) == 1 and report.percentile([5, 1, 3], 100) == 5


def test_prompt_preview_variants() -> None:
    assert report.prompt_preview(body("hello   \n world")) == "hello world"
    assert len(report.prompt_preview(body("x" * 500))) == 100
    anth = json.dumps({"system": "sys", "messages": [{"role": "user", "content": [{"type": "text", "text": "block text"}]}]})
    assert report.prompt_preview(anth) == "block text"
    assert report.prompt_preview("not json at all") == "not json at all"
    assert report.prompt_preview(json.dumps({"input": "responses api"})) == "responses api"
    assert report.prompt_preview("") == ""


def test_safe_preview_never_leaks() -> None:
    assert report.safe_preview("sk-ant-…a1b2") == "sk-ant-…a1b2"
    assert "AKIAIOSFODNN7EXAMPLE" not in report.safe_preview("AKIAIOSFODNN7EXAMPLE")
    assert report.safe_preview("AKIAIOSFODNN7EXAMPLE") == "AKI…LE"
    assert report.safe_preview(None) == ""


# ------------------------------------------------------------------ build_report numbers

def test_totals_and_breakdowns(crafted: Store) -> None:
    d = report.build_report(crafted, since=None, now=NOW)
    t = d["totals"]
    assert t["requests"] == 12
    assert t["cost_usd"] == pytest.approx(0.10 + 0.10 + 0.12 + 2.0 + 0.5 + 0.02)
    assert t["saved_usd"] == pytest.approx(0.10)
    assert t["cache_hits"] == 1
    assert t["blocked"] == 3           # 2 loop + 1 guard block
    assert t["errors"] == 1            # the 500 (blocked ones don't count)
    assert t["input_tokens"] == 12 * 100 and t["output_tokens"] == 12 * 50 and t["total_tokens"] == 12 * 150
    models = {r["key"]: r for r in d["by_model"]}
    assert models["gpt-4o"]["cost_usd"] == pytest.approx(2.02)
    assert models["claude-sonnet-4-5"]["requests"] == 6
    assert d["by_model"][0]["key"] == "gpt-4o"                       # sorted by cost desc
    assert {r["key"] for r in d["by_provider"]} == {"openai", "anthropic"}
    # three consecutive days, gap-free, ascending
    assert [r["requests"] for r in d["by_day"]] == [3, 5, 4]
    assert [r["key"] for r in d["by_day"]] == sorted(r["key"] for r in d["by_day"])
    assert list(d["by_tag"]) == ["feature"]
    tags = {r["key"]: r for r in d["by_tag"]["feature"]}
    assert tags["translate"]["cost_usd"] == pytest.approx(2.0) and tags["summary"]["requests"] == 3


def test_error_rate_and_sources_exact(crafted: Store) -> None:
    d = report.build_report(crafted, now=NOW)
    # upstream rows: 3 + 2 + 1 + 1(500) + 1(guard one) = 8 ; cache 1 ; loop 2 ; blocked 1
    assert d["by_source"] == {"upstream": 8, "cache": 1, "loop": 2, "blocked": 1}
    assert d["totals"]["requests"] == 12 and d["totals"]["upstream_requests"] == 8
    assert d["totals"]["errors"] == 1
    assert d["totals"]["error_rate"] == pytest.approx(1 / 8)


def test_latency_percentiles(crafted: Store) -> None:
    d = report.build_report(crafted, now=NOW)
    lat = d["latency"]
    # successful upstream latencies sorted: 500, 1000, 2000, 2001, 2002, 3000, 3000
    assert lat["samples"] == 7
    assert lat["p50_ms"] == 2001.0
    assert lat["p95_ms"] == pytest.approx(3000.0)
    assert lat["ttft_samples"] == 3 and lat["ttft_p50_ms"] == pytest.approx(400.0)


def test_waste_finder_exact(crafted: Store) -> None:
    w = report.build_report(crafted, now=NOW)["waste"]
    # A: 3 upstream calls, first (0.10) isn't waste -> 0.10 + 0.12; B: 1.0. Cache hit, 500 error and loop rows excluded.
    assert w["duplicate_requests"] == 3
    assert w["duplicate_groups"] == 2
    assert w["savings_usd"] == pytest.approx(1.22)
    top = w["groups"][0]
    assert top["preview"] == "translate this to French" and top["count"] == 2
    assert top["savings_usd"] == pytest.approx(1.0) and top["total_cost_usd"] == pytest.approx(2.0)
    second = w["groups"][1]
    assert second["count"] == 3 and second["savings_usd"] == pytest.approx(0.22)
    assert second["total_cost_usd"] == pytest.approx(0.32) and second["model"] == "claude-sonnet-4-5"
    assert w["waste_pct"] == pytest.approx(1.22 / 2.84)


def test_waste_ignores_volatile_fields_and_providers(store: Store) -> None:
    store.record(ex("same", ts=BASE, cost=0.4, raw_body=json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "same"}], "user": "u1"})))
    store.record(ex("same", ts=BASE + 1, cost=0.4, raw_body=json.dumps({"messages": [{"role": "user", "content": "same"}], "model": "gpt-4o", "user": "u2"})))
    store.record(ex("same", ts=BASE + 2, cost=0.4, provider="groq"))       # different provider -> different key
    usd, n = report.potential_savings(store)
    assert (usd, n) == (pytest.approx(0.4), 1)


def test_waste_since_filter(crafted: Store) -> None:
    usd, n = report.potential_savings(crafted, since=BASE - DAY - 1)
    assert (usd, n) == (pytest.approx(1.0), 1)      # only B's pair is inside the window
    assert report.potential_savings(crafted, since=BASE + 100) == (0.0, 0)


def test_loops_blocked_and_prevented(crafted: Store) -> None:
    lp = report.build_report(crafted, now=NOW)["loops"]
    assert lp["blocked"] == 2 and lp["budget_blocked"] == 0
    assert lp["groups"][0]["blocked"] == 2 and lp["groups"][0]["preview"] == "summarize the incident report"
    assert lp["est_prevented_usd"] == pytest.approx(2 * 0.32 / 3)


def test_guard_findings_table(crafted: Store) -> None:
    g = report.build_report(crafted, now=NOW)["guard"]
    assert g["total"] == 3 and g["exchanges"] == 2 and g["blocked_requests"] == 1
    kinds = {k["kind"]: k for k in g["kinds"]}
    assert kinds["aws_access_key"]["count"] == 2 and kinds["aws_access_key"]["exchanges"] == 2
    assert kinds["aws_access_key"]["previews"] == ["AKIA…MPLE"] and kinds["aws_access_key"]["pii"] is False
    assert kinds["email"]["pii"] is True
    assert report.build_report(crafted, now=NOW)["totals"]["guard_findings"] == 3


def test_expensive_top10(store: Store) -> None:
    for i in range(15):
        store.record(ex(f"prompt number {i}", ts=BASE + i, cost=0.01 * (i + 1)))
    store.record(ex("free one", ts=BASE, cost=0.0))
    d = report.build_report(store, now=NOW)
    costs = [r["cost_usd"] for r in d["expensive"]]
    assert len(costs) == 10 and costs == sorted(costs, reverse=True)
    assert costs[0] == pytest.approx(0.15) and d["expensive"][0]["preview"] == "prompt number 14"


def test_since_filter_in_totals(crafted: Store) -> None:
    d = report.build_report(crafted, since=BASE - 1, now=NOW)
    assert d["totals"]["requests"] == 4 and d["period"]["since"] == BASE - 1


def test_pagination_over_page_boundary(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "PAGE", 7)
    for i in range(30):
        store.record(ex("dup", ts=BASE + i, cost=0.1))
    d = report.build_report(store, now=NOW)
    assert d["totals"]["requests"] == 30
    assert d["waste"]["duplicate_requests"] == 29 and d["waste"]["savings_usd"] == pytest.approx(2.9)


def test_empty_db(store: Store) -> None:
    d = report.build_report(store, now=NOW)
    assert d["totals"]["requests"] == 0 and d["totals"]["cost_usd"] == 0
    assert d["latency"]["p50_ms"] is None and d["by_day"] == [] and d["waste"]["groups"] == []
    h = report.render_html(d)
    assert "No waste found" in h and "<html" in h
    md = report.render_markdown(d)
    assert "llmtap report" in md
    assert json.loads(report.render_json(d))["totals"]["requests"] == 0
    assert report.potential_savings(store) == (0.0, 0)


def test_json_serializable(crafted: Store) -> None:
    d = report.build_report(crafted, now=NOW)
    again = json.loads(json.dumps(d))
    assert again["waste"]["savings_usd"] == pytest.approx(1.22)
    assert json.loads(report.render_json(d))["version"] == __version__


# ------------------------------------------------------------------ rendering

def test_html_is_self_contained(crafted: Store) -> None:
    h = report.render_html(report.build_report(crafted, now=NOW))
    assert h.startswith("<!doctype html>")
    assert "<script" not in h.lower()
    assert not re.search(r"(src|href)\s*=\s*[\"']https?://", h)
    assert "url(http" not in h and "@import" not in h
    assert "<svg" in h and "<style>" in h
    for needle in ("Waste finder", "Most expensive requests", "Guard findings", "Runaway loops blocked",
                   "Cost by model", "Cost by day", "$1.22", "would save $1.22", f"llmtap v{__version__}",
                   "AKIA…MPLE", "translate this to French"):
        assert needle in h, needle


def test_html_escapes_untrusted_text(store: Store) -> None:
    evil = "<script>alert(1)</script><img src=x onerror=alert(2)>"
    for i in range(2):
        store.record(ex(evil, ts=BASE + i, cost=0.3, model="<b>evil</b>", tags={"feature": "<svg onload=x>"},
                        findings=[{"kind": "<i>k</i>", "path": "", "preview": "<script>x</script>", "start": 0, "end": 1}]))
    h = report.render_html(report.build_report(store, now=NOW))
    assert "<script" not in h.lower()
    assert "<img" not in h and "<b>evil" not in h and "<svg onload" not in h and "<i>k</i>" not in h
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in h
    md = report.render_markdown(report.build_report(store, now=NOW))
    assert "<script>" not in md and "&lt;script&gt;" in md      # no raw HTML in PR comments


def test_full_secret_never_in_report(store: Store) -> None:
    secret = "AKIAIOSFODNN7EXAMPLE"
    store.record(ex("x", ts=BASE, cost=0.1, findings=[{"kind": "aws_access_key", "path": "", "preview": secret, "start": 0, "end": 20}]))
    d = report.build_report(store, now=NOW)
    for out in (report.render_html(d), report.render_markdown(d), report.render_json(d)):
        assert secret not in out


def test_markdown(crafted: Store) -> None:
    md = report.render_markdown(report.build_report(crafted, now=NOW))
    assert md.startswith("## llmtap report")
    assert "**Spend $2.84**" in md
    assert "Waste finder — $1.22 avoidable" in md
    assert "| translate this to French | ×2 | $2.00 | $1.00 |" in md
    assert "### Cost by model" in md and "gpt-4o" in md
    assert "### Guard findings" in md and "`AKIA…MPLE`" in md
    assert "Runaway loops blocked: 2" in md
    assert md.rstrip().endswith(f"_Generated by llmtap v{__version__}_")
    assert len(md.splitlines()) < 80


def test_day_chart_fills_gaps_and_handles_free_models(store: Store) -> None:
    store.record(ex("a", ts=BASE - 3 * DAY, cost=1.0))
    store.record(ex("b", ts=BASE, cost=2.0))
    d = report.build_report(store, now=NOW)
    assert len(d["by_day"]) == 4 and d["by_day"][1]["requests"] == 0
    assert report.render_html(d).count("<rect") >= 4
    s2 = Store(":memory:")
    s2.record(ex("local", model="llama3", cost=0.0))
    h = report.render_html(report.build_report(s2, now=NOW))
    assert "No paid spend" in h
    s2.close()
