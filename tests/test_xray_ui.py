"""Context X-ray in the dashboard (JSON API + UI code) and in the HTML / Markdown report."""
from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.request
from typing import Any

import pytest

from llmtap import dashboard, report
from llmtap.models import Exchange, Usage
from llmtap.store import Store

SONNET = "claude-sonnet-4-5"
SYSTEM = ("You are a coding agent. Follow the repository conventions and keep changes minimal. " * 40
          + "\nToday's date: 2026-11-24 10:31\n" + "Never commit secrets. " * 10)


def tool(name: str, words: int) -> dict[str, Any]:
    return {"name": name, "description": f"{name}: " + "Performs the described operation carefully. " * words,
            "input_schema": {"type": "object", "properties": {"arg": {"type": "string", "description": "argument"}}}}


TOOLS = [tool("Read", 20), tool("Bash", 60), tool("mcp__github__create_issue", 40),
         tool("mcp__github__list_pull_requests", 25), tool("mcp__slack__post_message", 10)]
HISTORY = [
    {"role": "user", "content": "Please fix the failing test in tests/test_api.py"},
    {"role": "assistant", "content": [{"type": "text", "text": "Looking at it. " * 30}]},
    {"role": "user", "content": [{"type": "text", "text": "it still fails", "cache_control": {"type": "ephemeral"}}]},
]


def req(system: str = SYSTEM, tools: list | None = None, messages: list | None = None) -> dict[str, Any]:
    return {"model": SONNET, "max_tokens": 4096,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools": copy.deepcopy(TOOLS if tools is None else tools),
            "messages": copy.deepcopy(HISTORY if messages is None else messages)}


def grow(r: dict[str, Any], system: str | None = None) -> dict[str, Any]:
    nxt = copy.deepcopy(r)
    if system is not None:
        nxt["system"][0]["text"] = system
    nxt["messages"] += [{"role": "assistant", "content": [{"type": "text", "text": "Fixed it."}]},
                        {"role": "user", "content": "thanks, now run lint"}]
    return nxt


def mk(body: Any, *, ts: float, inp: int = 50, cr: int = 0, cw: int = 0, session: str | None = "s1",
       source: str = "upstream", model: str = SONNET, id: str | None = None) -> Exchange:
    kw: dict[str, Any] = {"id": id} if id else {}
    return Exchange(ts=ts, provider="anthropic", style="anthropic", model=model, path="/v1/messages",
                    request_body=body if isinstance(body, str) else json.dumps(body), status=200,
                    usage=Usage(input_tokens=inp, output_tokens=80, cache_read_tokens=cr, cache_write_tokens=cw),
                    source=source, tags={"session": session} if session else {}, **kw)


def miss_pair(t0: float, system2: str = SYSTEM.replace("10:31", "10:32"), **kw: Any) -> tuple[Exchange, Exchange]:
    """Two turns of one conversation; the system prompt changed in between, so the cache missed."""
    first = req(**kw)
    return (mk(first, ts=t0, cw=10_000), mk(grow(first, system=system2), ts=t0 + 30, cw=10_600))


def get(t: Any, path: str, headers: dict[str, str] | None = None) -> tuple[int, Any]:
    r = urllib.request.Request(t.base_url + path, headers=headers or {})
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def get_json(t: Any, path: str) -> tuple[int, Any]:
    status, raw = get(t, path)
    return status, json.loads(raw)


# ------------------------------------------------------------------ dashboard API


def test_exchange_xray_endpoint(tap: Any) -> None:
    t = tap()
    now = time.time()
    prev, cur = miss_pair(now - 120)
    other_session = mk(req(), ts=now - 100, cw=10_000, session="another")     # not part of cur's conversation
    for ex in (prev, cur, other_session):
        t.store.record(ex)

    status, d = get_json(t, f"/_llmtap/api/exchanges/{cur.id}/xray")
    assert status == 200
    assert set(d) >= {"breakdown", "tools", "servers", "cache"}
    bd = d["breakdown"]
    assert bd["parsed"] and bd["scaled"] and bd["input_tokens"] == 50 + 10_600
    assert {"system", "tools", "history", "new_input"} <= set(bd["parts"])
    assert sum(p["tokens"] for p in bd["parts"].values()) == bd["input_tokens"]
    assert len(d["tools"]) == len(TOOLS) and d["tools"][0]["name"] == "Bash"           # ranked by size
    assert [s["server"] for s in d["servers"]] == ["github", "slack"]
    assert d["servers"][0]["tools"] == 2
    c = d["cache"]
    assert d["prev_id"] == prev.id and c["prev_id"] == prev.id
    assert c["verdict"] == "miss" and c["cause"] == "system_changed" and c["lost_usd"] > 0
    assert "10:31" in c["detail"] and "10:32" in c["detail"]

    # the first request of a conversation has nothing to compare against
    status, d = get_json(t, f"/_llmtap/api/exchanges/{prev.id}/xray")
    assert status == 200 and d["cache"] is None and d["prev_id"] is None
    # id prefix works like the detail route; unknown ids are 404
    assert get_json(t, f"/_llmtap/api/exchanges/{cur.id[:10]}/xray")[0] == 200
    status, d = get_json(t, "/_llmtap/api/exchanges/doesnotexist/xray")
    assert status == 404 and "not found" in d["error"]
    # the plain detail route is unchanged
    status, d = get_json(t, f"/_llmtap/api/exchanges/{cur.id}")
    assert status == 200 and d["id"] == cur.id and "request_body" in d


def test_exchange_xray_caps_tools_and_handles_unparseable_bodies(tap: Any) -> None:
    t = tap()
    now = time.time()
    many = [tool(f"mcp__srv__t{i:02d}", 5 + i) for i in range(15)]
    big = mk(req(tools=many), ts=now - 10, cw=12_000)
    junk = mk("this is { not json", ts=now - 5)
    t.store.record(big)
    t.store.record(junk)
    status, d = get_json(t, f"/_llmtap/api/exchanges/{big.id}/xray")
    assert status == 200 and len(d["tools"]) == dashboard.XRAY_TOP_TOOLS
    assert d["servers"][0] == {**d["servers"][0], "server": "srv", "tools": 15}
    status, d = get_json(t, f"/_llmtap/api/exchanges/{junk.id}/xray")
    assert status == 200 and d["breakdown"]["parsed"] is False and d["tools"] == [] and d["servers"] == []


def test_xray_summary_endpoint(tap: Any) -> None:
    t = tap()
    now = time.time()
    prev, cur = miss_pair(now - 300)
    for ex in (prev, cur,
               mk(req(), ts=now - 200, source="cache", cw=0),                    # not upstream: ignored
               mk(req(), ts=now - 3 * 86400, cw=10_000, session="old")):         # before the window
        t.store.record(ex)

    status, d = get_json(t, f"/_llmtap/api/xray?since={now - 3600:.3f}")
    assert status == 200
    assert d["requests"] == 2 and d["analyzed"] == 2 and d["capped"] is False
    assert d["avg_input_tokens"] == (10_050 + 10_650) // 2
    assert {"system", "tools", "history", "new_input"} <= set(d["parts"])
    assert abs(sum(p["share"] for p in d["parts"].values()) - 1.0) < 1e-9
    tb = d["tool_bloat"]
    assert tb["requests_with_tools"] == 2 and tb["servers"][0]["server"] == "github"
    assert d["cache"]["pairs"] == 1 and d["cache"]["total_misses"] == 1
    assert d["cache"]["by_cause"]["system_changed"]["count"] == 1 and d["cache"]["total_lost_usd"] > 0

    status, d = get_json(t, "/_llmtap/api/xray?since=all")
    assert status == 200 and d["requests"] == 3

    # default window: since local midnight (the 3-day-old request is out)
    status, d = get_json(t, "/_llmtap/api/xray")
    assert status == 200 and d["since"] == pytest.approx(dashboard._local_midnight(), abs=1)
    assert d["requests"] == 2


def test_xray_summary_empty_and_capped(tap: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    t = tap()
    status, d = get_json(t, "/_llmtap/api/xray")
    assert status == 200 and d["requests"] == 0 and d["analyzed"] == 0 and d["cache"]["pairs"] == 0
    now = time.time()
    for i in range(5):
        t.store.record(mk(req(), ts=now - 60 + i, cw=10_000, session=f"s{i}"))
    monkeypatch.setattr(dashboard, "XRAY_MAX_REQUESTS", 3)
    status, d = get_json(t, "/_llmtap/api/xray?since=all")
    assert status == 200 and d["requests"] == 3 and d["capped"] is True and d["max_requests"] == 3


def test_xray_endpoints_keep_the_host_check_and_are_read_only(tap: Any) -> None:
    t = tap()
    ex = mk(req(), ts=time.time(), cw=10_000)
    t.store.record(ex)
    for path in (f"/_llmtap/api/exchanges/{ex.id}/xray", "/_llmtap/api/xray"):
        status, _ = get(t, path, {"Host": "evil.example.com"})
        assert status == 403
    r = urllib.request.Request(t.base_url + "/_llmtap/api/xray", data=b"{}", method="POST",
                               headers={"content-type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(r, timeout=15)
    assert e.value.code == 404
    assert t.store.count() == 1


def test_dashboard_html_contains_the_xray_ui(tap: Any) -> None:
    t = tap()
    status, raw = get(t, "/_llmtap/")
    html = raw.decode("utf-8")
    assert status == 200
    assert 'data-t="xray"' in html                          # the drawer tab
    assert "'/xray'" in html and "/api/xray" in html         # lazy per-request fetch + today's summary
    assert 'id="xsum"' in html and "function renderXraySum" in html
    assert "function xrayHtml" in html and "function verdictHtml" in html
    assert "--x-tools" in html                                # part colours are theme tokens
    # untrusted fields go through esc(): spot-check the tool name, server and cache detail
    body = html[html.index("function xrayHtml"):html.index("function xrayTab")]
    assert "esc(t.name)" in body and "esc(s.server)" in body
    assert "esc(c.detail" in html[html.index("function verdictHtml"):]
    assert "https://" not in html.split("<script>", 1)[1].split("BASE", 1)[0]   # no CDN in the script


# ------------------------------------------------------------------ report


@pytest.fixture()
def store() -> Store:
    s = Store(":memory:")
    yield s
    s.close()


def test_report_xray_section(store: Store) -> None:
    now = time.time()
    prev, cur = miss_pair(now - 600)
    store.record(prev)
    store.record(cur)
    data = report.build_report(store)
    x = data["xray"]
    assert x["analyzed_requests"] == 2 and x["eligible_requests"] == 2 and x["capped"] is False
    assert {"system", "tools", "history", "new_input"} <= set(x["parts"])
    assert x["tools"]["top_tools"][0]["name"] == "Bash"
    assert [s["server"] for s in x["tools"]["servers"]] == ["github", "slack"]
    assert x["cache"]["misses"] == 1 and x["cache"]["by_cause"][0]["cause"] == "system_changed"
    assert x["cache"]["lost_usd"] > 0 and x["cache"]["top_misses"][0]["id"] == cur.id
    json.dumps(data)                                           # still JSON-serializable

    html = report.render_html(data)
    assert "Context X-ray" in html and "system prompt changed" in html and "lost to cache misses" in html
    assert "mcp__github__create_issue" in html and "<script" not in html.lower()
    md = report.render_markdown(data)
    assert "### Context X-ray" in md and "| system prompt changed | 1 |" in md
    assert "| MCP server | Tools |" in md


def test_report_xray_escapes_malicious_tool_names_and_prompts(store: Store) -> None:
    evil_tool = "<script>alert('tool')</script>"
    evil_server = 'mcp__<img src=x onerror=alert(1)>__run'
    evil_sys = SYSTEM.replace("10:31", "<img src=x onerror=alert(2)>")
    tools = [tool(evil_tool, 80), tool(evil_server, 30), tool("Read", 5)]
    now = time.time()
    first = req(system=evil_sys, tools=tools)
    store.record(mk(first, ts=now - 60, cw=10_000))
    store.record(mk(grow(first, system=evil_sys.replace("alert(2)", "alert(3)</td><b>")), ts=now - 30, cw=10_600))
    data = report.build_report(store)
    assert data["xray"]["tools"]["top_tools"][0]["name"] == evil_tool
    assert data["xray"]["cache"]["top_misses"][0]["cause"] == "system_changed"
    html = report.render_html(data)
    low = html.lower()
    assert "<script" not in low and "<img" not in low and "</td><b>" not in html
    assert "&lt;script&gt;alert(&#x27;tool&#x27;)&lt;/script&gt;" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html          # the MCP server name, escaped
    assert "alert(3)&lt;/td&gt;&lt;b&gt;" in html                # the cache-miss detail quoting the prompt
    md = report.render_markdown(data)
    assert "<script" not in md and "<img" not in md


def test_report_xray_on_an_empty_db(store: Store) -> None:
    data = report.build_report(store)
    x = data["xray"]
    assert x["analyzed_requests"] == 0 and x["eligible_requests"] == 0 and x["cache"]["pairs"] == 0
    html = report.render_html(data)
    assert "Context X-ray" in html and "No upstream request with a readable body" in html
    assert "### Context X-ray" in report.render_markdown(data)
    json.dumps(data)


def test_report_xray_without_parseable_bodies(store: Store) -> None:
    now = time.time()
    store.record(mk("not json at all {", ts=now - 20))
    store.record(mk('{"truncated": ', ts=now - 10))
    store.record(Exchange(ts=now - 5, provider="openai", style="openai", model="gpt-4o", request_body="",
                          status=200, usage=Usage(input_tokens=10, output_tokens=5)))
    data = report.build_report(store)
    x = data["xray"]
    assert x["eligible_requests"] == 2 and x["analyzed_requests"] == 0 and x["unparsed_requests"] == 2
    assert "No upstream request with a readable body" in report.render_html(data)
    report.render_markdown(data)


def test_report_xray_is_capped_on_huge_databases(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "XRAY_MAX_REQUESTS", 3)
    now = time.time()
    for i in range(5):
        store.record(mk(req(), ts=now - 100 + i, cw=10_000, session=f"s{i}"))
    data = report.build_report(store)
    x = data["xray"]
    assert x["eligible_requests"] == 5 and x["considered_requests"] == 3 and x["capped"] is True
    assert x["analyzed_requests"] == 3
    html = report.render_html(data)
    assert "the most recent 3 of 5" in html and "at most 3 requests" in html
    assert "the most recent 3 of 5" in report.render_markdown(data)
