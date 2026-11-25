"""Tests for llmtap.xray: context breakdown per wire style, scaling to reported usage, tool/MCP bloat,
cache-miss explanations, renderers, and graceful degradation on malformed bodies."""
from __future__ import annotations

import base64
import copy
import json
from typing import Any

import pytest

from llmtap import xray
from llmtap.models import Exchange, Usage
from llmtap.pricing import cost_usd, price_for
from llmtap.store import Store

T0 = 1_795_000_000.0
SONNET = "claude-sonnet-4-5"

# ------------------------------------------------------------------ fixtures

SYSTEM = (
    "You are Claude Code, Anthropic's official CLI for Claude.\n"
    "You are an interactive CLI tool that helps users with software engineering tasks.\n"
    + "Follow the repository conventions and keep changes minimal. " * 40
    + "\n<env>\nWorking directory: /home/dev/project\nPlatform: linux\n"
    "Today's date: 2026-11-24 10:31\n</env>\n"
    + "IMPORTANT: never commit secrets. " * 10
)


def tool(name: str, desc_words: int, props: int = 3) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"{name}: " + "Performs the operation described by the arguments carefully. " * desc_words,
        "input_schema": {
            "type": "object",
            "properties": {f"arg_{i}": {"type": "string", "description": f"argument number {i} for {name}"}
                           for i in range(props)},
            "required": ["arg_0"],
        },
    }


TOOLS = [
    tool("Read", 30), tool("Bash", 60, 5), tool("Edit", 20),
    tool("mcp__github__create_issue", 25, 6), tool("mcp__github__list_pull_requests", 15, 4),
    tool("mcp__slack__post_message", 5, 2),
]

HISTORY = [
    {"role": "user", "content": "Please fix the failing test in tests/test_api.py"},
    {"role": "assistant", "content": [
        {"type": "text", "text": "I'll look at the test first."},
        {"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {"arg_0": "tests/test_api.py"}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_01", "content": "def test_api():\n    assert f() == 2\n" * 40},
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": "The assertion is wrong; let me run it."},
        {"type": "tool_use", "id": "toolu_02", "name": "Bash", "input": {"arg_0": "pytest -q tests/test_api.py"}},
    ]},
]
NEW_TURN = {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "toolu_02", "content": "1 failed, 3 passed in 0.12s",
     "cache_control": {"type": "ephemeral"}},
]}


def anth_req(system: str = SYSTEM, tools: list | None = None, messages: list | None = None, model: str = SONNET,
             **extra: Any) -> dict[str, Any]:
    return {"model": model, "max_tokens": 8192, "stream": True,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools": copy.deepcopy(TOOLS if tools is None else tools),
            "messages": copy.deepcopy(HISTORY + [NEW_TURN] if messages is None else messages), **extra}


def mk(req: Any, *, style: str = "anthropic", provider: str | None = None, model: str | None = SONNET,
       ts: float = T0, inp: int = 0, cr: int = 0, cw: int = 0, out: int = 100, estimated: bool = False,
       session: str | None = "s1", source: str = "upstream", status: int = 200) -> Exchange:
    body = req if isinstance(req, str) else json.dumps(req)
    tags = {"session": session} if session else {}
    return Exchange(ts=ts, provider=provider or style, style=style, model=model, request_body=body, status=status,
                    usage=Usage(input_tokens=inp, output_tokens=out, cache_read_tokens=cr, cache_write_tokens=cw,
                                estimated=estimated),
                    source=source, tags=tags)


def total_tokens(bd: dict) -> int:
    return sum(p["tokens"] for p in bd["parts"].values())


# ------------------------------------------------------------------ 1. breakdown per style


def test_anthropic_breakdown_scales_to_reported_usage():
    ex = mk(anth_req(), inp=420, cr=9000, cw=580)
    bd = xray.breakdown(ex)
    assert bd["shape"] == "anthropic" and bd["parsed"] and bd["scaled"] and not bd["estimated"]
    assert bd["input_tokens"] == 10_000 == total_tokens(bd)          # anthropic: input + cache_read + cache_write
    assert set(bd["parts"]) == {"system", "tools", "history", "new_input"}
    assert bd["counts"]["tools"] == 6 and bd["counts"]["messages"] == 5 and bd["counts"]["new_input_messages"] == 1
    parts = bd["parts"]
    assert parts["tools"]["tokens"] > parts["system"]["tokens"] > parts["new_input"]["tokens"] > 0
    assert abs(sum(p["share"] for p in parts.values()) - 1.0) < 1e-9
    # dollars: the parts add up to the request's input cost
    input_cost = cost_usd(SONNET, Usage(input_tokens=420, cache_read_tokens=9000, cache_write_tokens=580),
                          style="anthropic")
    assert bd["input_cost_usd"] == pytest.approx(input_cost)
    assert sum(p["usd"] for p in parts.values()) == pytest.approx(input_cost)
    # cache-aware: the prefix (tools) is billed at the cache-read rate, the new input at full price
    per_tok = {k: v["usd"] / v["tokens"] for k, v in parts.items()}
    assert per_tok["tools"] < per_tok["new_input"]


def test_breakdown_raw_estimates_when_usage_missing_or_estimated():
    for usage in ({}, {"inp": 999, "estimated": True}):
        bd = xray.breakdown(mk(anth_req(), **usage))
        assert bd["estimated"] and not bd["scaled"]
        assert bd["input_tokens"] == total_tokens(bd) == bd["raw_estimate_tokens"] > 1000
        assert any("estimate" in n for n in bd["notes"])
        assert bd["input_cost_usd"] > 0                                 # priced as uncached input


def test_openai_chat_system_developer_tools_and_trailing_tool_messages():
    req = {
        "model": "gpt-4.1", "temperature": 0.2, "parallel_tool_calls": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. " * 50},
            {"role": "developer", "content": "Answer in English."},
            {"role": "user", "content": "What's the weather in Paris and Rome?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}},
                {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Rome"}'}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"temp": 12, "sky": "cloudy"}' * 10},
            {"role": "tool", "tool_call_id": "c2", "content": '{"temp": 18, "sky": "sunny"}' * 10},
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather by city. " * 30,
                                                    "parameters": {"type": "object", "properties": {
                                                        "city": {"type": "string"}}}}}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}},
    }
    ex = mk(req, style="openai", model="gpt-4.1", inp=2000, cr=1024)
    bd = xray.breakdown(ex)
    assert bd["shape"] == "openai-chat"
    assert bd["input_tokens"] == 2000 == total_tokens(bd)            # openai: cached tokens are a subset of input
    assert bd["uncached_input_tokens"] == 976
    assert bd["counts"]["system_blocks"] == 2 and bd["counts"]["new_input_messages"] == 2
    assert bd["parts"]["new_input"]["tokens"] > bd["parts"]["history"]["tokens"]
    assert "other" in bd["parts"] and bd["parts"]["other"]["tokens"] > 0   # response_format schema
    assert [t["name"] for t in xray.tool_bloat(ex)] == ["get_weather"]


def test_openai_responses_instructions_input_and_mcp_tool():
    req = {
        "model": "gpt-5", "instructions": "You are a coding agent. " * 80, "store": False,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "list the repo files"}]},
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "function_call", "call_id": "fc_1", "name": "shell", "arguments": '{"cmd":"ls"}'},
            {"type": "function_call_output", "call_id": "fc_1", "output": "README.md\nsrc\ntests\n" * 30},
        ],
        "tools": [
            {"type": "function", "name": "shell", "description": "Run a shell command. " * 20,
             "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
            {"type": "web_search"},
            {"type": "mcp", "server_label": "deepwiki", "server_url": "https://mcp.example/mcp",
             "require_approval": "never"},
        ],
    }
    ex = mk(req, style="openai", provider="openai", model="gpt-5", inp=3000, cr=2048)
    bd = xray.breakdown(ex)
    assert bd["shape"] == "openai-responses"
    assert bd["parts"]["system"]["tokens"] > 0 and bd["counts"]["new_input_messages"] == 1
    assert bd["parts"]["history"]["tokens"] > 0 and total_tokens(bd) == 3000
    tools = {t["name"]: t for t in xray.tool_bloat(ex)}
    assert set(tools) == {"shell", "web_search", "mcp:deepwiki"}
    assert tools["mcp:deepwiki"]["server"] == "deepwiki" and tools["mcp:deepwiki"]["kind"] == "mcp"
    assert tools["web_search"]["kind"] == "builtin"
    # a plain string input is the new input
    bd2 = xray.breakdown(mk({"model": "gpt-5", "input": "hello there " * 20}, style="openai", model="gpt-5"))
    assert bd2["parts"]["new_input"]["tokens"] == bd2["input_tokens"] > 0


def test_gemini_generate_content():
    req = {
        "systemInstruction": {"parts": [{"text": "You are a terse assistant. " * 40}]},
        "contents": [
            {"role": "user", "parts": [{"text": "Find flights to Tokyo"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "search_flights", "args": {"to": "HND"}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "search_flights",
                                                             "response": {"flights": ["NH106"] * 50}}}]},
        ],
        "tools": [{"functionDeclarations": [
            {"name": "search_flights", "description": "Search flights. " * 25,
             "parameters": {"type": "OBJECT", "properties": {"to": {"type": "STRING"}}}},
            {"name": "book_flight", "description": "Book a flight. " * 10},
        ]}, {"googleSearch": {}}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000},
    }
    ex = mk(req, style="gemini", provider="gemini", model="gemini-2.5-flash", inp=1500, cr=0)
    bd = xray.breakdown(ex)
    assert bd["shape"] == "gemini" and total_tokens(bd) == 1500
    assert bd["counts"]["messages"] == 3 and bd["counts"]["new_input_messages"] == 1
    assert all(bd["parts"][p]["tokens"] > 0 for p in ("system", "tools", "history", "new_input"))
    assert "other" not in bd["parts"]                                 # generationConfig is not prompt text
    names = [t["name"] for t in xray.tool_bloat(ex)]
    assert names[0] == "search_flights" and set(names) == {"search_flights", "book_flight", "googleSearch"}


def test_bedrock_converse():
    req = {
        "system": [{"text": "You are a support bot for ACME. " * 30}, {"cachePoint": {"type": "default"}}],
        "messages": [
            {"role": "user", "content": [{"text": "My order is late"}]},
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "lookup_order",
                                                           "input": {"id": "A1"}}}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "shipped " * 30}]}}]},
        ],
        "toolConfig": {"tools": [{"toolSpec": {"name": "lookup_order", "description": "Look up an order. " * 20,
                                               "inputSchema": {"json": {"type": "object"}}}},
                                 {"cachePoint": {"type": "default"}}]},
        "inferenceConfig": {"maxTokens": 512},
    }
    ex = mk(req, style="bedrock", provider="bedrock", model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            inp=300, cr=900, cw=0)
    bd = xray.breakdown(ex)
    assert bd["shape"] == "bedrock-converse"
    assert bd["input_tokens"] == 1200 == total_tokens(bd)            # bedrock: anthropic-style exclusive counts
    assert bd["counts"]["tools"] == 1 and bd["counts"]["new_input_messages"] == 1
    assert bd["parts"]["system"]["tokens"] > 0 and bd["input_cost_usd"] > 0
    assert [t["name"] for t in xray.tool_bloat(ex)] == ["lookup_order"]


def test_bedrock_invoke_with_anthropic_body():
    req = anth_req()
    req.pop("model")
    req["anthropic_version"] = "bedrock-2023-05-31"
    ex = mk(req, style="bedrock", provider="bedrock", model="anthropic.claude-3-5-haiku-20241022-v1:0",
            inp=100, cr=5000, cw=200)
    bd = xray.breakdown(ex)
    assert bd["shape"] == "anthropic" and bd["input_tokens"] == 5300 == total_tokens(bd)
    assert bd["counts"]["tools"] == 6


def test_unknown_shape_goes_to_other_and_config_is_ignored():
    ex = mk({"model": "rerank-v3", "query": "cheap flights", "documents": ["doc one " * 20, "doc two"],
             "temperature": 0}, style="openai", model="rerank-v3", inp=80)
    bd = xray.breakdown(ex)
    assert bd["shape"] == "unknown"
    assert bd["parts"]["other"]["tokens"] == 80 and all(bd["parts"][p]["tokens"] == 0 for p in xray.PARTS[:4])
    assert any("unrecognized" in n for n in bd["notes"])


def test_assistant_prefill_counts_last_user_turn_as_new_input():
    msgs = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "write JSON please " * 10}, {"role": "assistant", "content": "{"}]
    bd = xray.breakdown(mk(anth_req(messages=msgs, tools=[])))
    assert bd["counts"]["new_input_messages"] == 2
    assert bd["parts"]["new_input"]["tokens"] > bd["parts"]["history"]["tokens"]


def test_images_are_flat_media_estimates_not_base64_length():
    png = base64.b64encode(b"\x89PNG" + bytes(range(256)) * 2000).decode()       # ~680 KB of base64
    msgs = [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}},
        {"type": "text", "text": "what is in this picture?"}]}]
    bd = xray.breakdown(mk(anth_req(messages=msgs, tools=[], system="short")))
    assert bd["counts"]["media"] == 1
    assert bd["parts"]["new_input"]["raw_tokens"] < xray.MEDIA_TOKENS + 100     # not 170k tokens of base64
    gem = {"contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": png}},
                                                   {"text": "describe"}]}]}
    assert xray.breakdown(mk(gem, style="gemini", model="gemini-2.5-flash"))["counts"]["media"] == 1


# ------------------------------------------------------------------ malformed / truncated bodies

MALFORMED = [
    "", "   ", "{not json", '{"model": "x", "messages": [', "[1, 2, 3]", "null", '"just a string"',
    "[binary request body not stored: application/octet-stream, 1234 bytes]",
    '{"model": "x", "messages": "not a list", "tools": "nope", "system": 42}',
    '{"messages": [null, 1, "str", {"role": null, "content": {"weird": [1, 2]}}], "tools": [null, "x", 3, {}]}',
    '{"contents": [null, {"parts": null}], "systemInstruction": 7, "tools": [{"functionDeclarations": [null, 1]}]}',
    '{"input": [null, 5, {"type": null}], "instructions": ["a", {"b": 1}], "tools": [{"type": "mcp"}]}',
    '{"toolConfig": {"tools": [{"toolSpec": null}, {"cachePoint": {}}]}, "messages": [{"content": [{"text": 1}]}]}',
    "[" * 5000 + "]" * 5000,
    '{"messages": ' + "[" * 3000 + "]" * 3000 + "}",
]


@pytest.mark.parametrize("body", MALFORMED)
def test_never_raises_on_malformed_bodies(body):
    ex = mk(body, inp=100, cr=50, cw=10)
    prev = mk(body, ts=T0 - 10, cw=160)
    bd = xray.breakdown(ex)
    assert total_tokens(bd) == bd["input_tokens"] == 160
    assert isinstance(xray.tool_bloat(ex), list) and isinstance(xray.tool_servers(ex), list)
    xray.explain_cache(prev, ex)
    assert "input" in xray.render_xray_text(ex, prev=prev)
    for style in ("openai", "gemini", "bedrock", "weird"):
        xray.breakdown(mk(body, style=style, inp=10))
    xray.render_summary_text([prev, ex])
    json.dumps(xray.summary_json([prev, ex]))


def test_truncated_body_degrades_to_other():
    body = json.dumps(anth_req())[:3000] + "\n[llmtap: truncated, 4000000 bytes in total]"
    bd = xray.breakdown(mk(body, inp=10, cr=90_000, cw=0))
    assert bd["truncated"] and not bd["parsed"] and bd["shape"] == "unparsed"
    assert bd["parts"]["other"]["tokens"] == 90_010
    assert any("truncated" in n for n in bd["notes"])
    assert xray.tool_bloat(mk(body)) == []


def test_breakdown_of_non_exchange_fields_is_safe():
    ex = Exchange(request_body=None, usage=None, model=None, style=None)  # type: ignore[arg-type]
    bd = xray.breakdown(ex)
    assert bd["input_tokens"] == 0 and bd["parsed"] is False
    assert xray.explain_cache(ex, ex) is None


# ------------------------------------------------------------------ 2. tool / MCP bloat


def test_tool_bloat_ranks_and_groups_mcp_servers():
    ex = mk(anth_req(), inp=400, cr=9000, cw=600)
    items = xray.tool_bloat(ex)
    assert [i["tokens"] for i in items] == sorted((i["tokens"] for i in items), reverse=True)
    assert items[0]["name"] == "Bash"
    bd = xray.breakdown(ex)
    assert sum(i["tokens"] for i in items) == bd["parts"]["tools"]["tokens"]
    assert sum(i["usd"] for i in items) == pytest.approx(bd["parts"]["tools"]["usd"])
    by = {i["name"]: i for i in items}
    assert by["mcp__github__create_issue"]["server"] == "github" and by["mcp__github__create_issue"]["kind"] == "mcp"
    assert by["Read"]["server"] is None and by["Read"]["kind"] == "function"
    servers = xray.tool_servers(ex)
    assert [(s["server"], s["tools"]) for s in servers] == [("github", 2), ("slack", 1)]
    assert servers[0]["tokens"] == by["mcp__github__create_issue"]["tokens"] + \
        by["mcp__github__list_pull_requests"]["tokens"]


def test_anthropic_server_tools_are_builtin():
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}, tool("Read", 5)]
    kinds = {i["name"]: i["kind"] for i in xray.tool_bloat(mk(anth_req(tools=tools), inp=5000))}
    assert kinds == {"web_search": "builtin", "Read": "function"}


def test_tool_bloat_summary_headlines():
    exs = [mk(anth_req(), ts=T0 + i, inp=400, cr=9000, cw=600) for i in range(3)]
    exs.append(mk(anth_req(tools=[]), ts=T0 + 10, inp=3000))                       # no tools
    exs.append(mk(anth_req(), ts=T0 + 11, inp=400, cr=9000, source="cache"))       # not billed: ignored
    exs.append(mk(anth_req(), ts=T0 + 12, inp=400, status=500))                    # failed: ignored
    s = xray.tool_bloat_summary(exs)
    assert s["requests"] == 4 and s["requests_with_tools"] == 3
    assert s["total_tool_tokens"] == sum(xray.breakdown(e)["parts"]["tools"]["tokens"] for e in exs[:3])
    assert s["avg_tool_tokens"] == round(s["total_tool_tokens"] / 4)
    assert s["total_tool_usd"] > 0 and s["top_tools"][0]["name"] == "Bash" and s["top_tools"][0]["requests"] == 3
    gh = s["servers"][0]
    assert (gh["server"], gh["tools"], gh["requests"]) == ("github", 2, 3)
    assert any("your 2 MCP tools from 'github' add ~" in h and "to 3 of 4 requests" in h for h in s["headlines"])
    assert any("your 1 MCP tool from 'slack' adds ~" in h for h in s["headlines"])
    s3 = xray.tool_bloat_summary(exs[:3])
    assert any("from 'github' add ~" in h and "to every request" in h and "over 3 requests" in h
               for h in s3["headlines"])
    assert xray.tool_bloat_summary([])["requests"] == 0


# ------------------------------------------------------------------ 3. cache-miss explainer


def pair(prev_req: dict, cur_req: dict, *, gap: float = 30, prev_usage: dict | None = None,
         cur_usage: dict | None = None, **kw: Any) -> tuple[Exchange, Exchange]:
    pu = prev_usage or {"inp": 50, "cr": 0, "cw": 10_000}
    cu = cur_usage or {"inp": 50, "cr": 0, "cw": 10_600}
    return mk(prev_req, ts=T0, **pu, **kw), mk(cur_req, ts=T0 + gap, **cu, **kw)


def grow(req: dict) -> dict:
    """The next turn of the same conversation: assistant reply + new user message appended."""
    nxt = copy.deepcopy(req)
    for m in nxt["messages"]:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                b.pop("cache_control", None)
    nxt["messages"] += [{"role": "assistant", "content": [{"type": "text", "text": "Fixed it."}]},
                        {"role": "user", "content": [{"type": "text", "text": "thanks, now run lint",
                                                      "cache_control": {"type": "ephemeral"}}]}]
    return nxt


def test_cache_hit():
    prev, cur = pair(anth_req(), grow(anth_req()), cur_usage={"inp": 30, "cr": 10_000, "cw": 620})
    r = xray.explain_cache(prev, cur)
    assert r["verdict"] == "hit" and r["cause"] is None and r["lost_usd"] == 0 and r["changes"] == []
    assert r["expected_cached_tokens"] == 10_000


def test_timestamp_in_system_prompt():
    cur_req = grow(anth_req(system=SYSTEM.replace("10:31", "10:32")))
    prev, cur = pair(anth_req(), cur_req)
    r = xray.explain_cache(prev, cur)
    assert r["verdict"] == "miss" and r["cause"] == "system_changed"
    pos = SYSTEM.index("10:31") + 4
    assert f"system prompt changed at char {pos:,}:" in r["detail"]
    assert "10:31" in r["detail"] and "10:32" in r["detail"] and "\\n" in r["detail"]    # newlines escaped
    assert "timestamp" in r["detail"]
    assert r["where"] == {"section": "system", "index": 0, "label": "system prompt", "char": pos}
    # 10,000 tokens re-written at the cache-write price instead of read at the cache-read price
    p = price_for(SONNET)
    assert r["lost_tokens"] == 10_000
    assert r["lost_usd"] == pytest.approx(10_000 * (p["cache_write"] - p["cache_read"]) / 1e6)


def test_tool_description_edited():
    tools = copy.deepcopy(TOOLS)
    tools[0]["description"] = tools[0]["description"].replace("carefully", "very carefully", 1)
    prev, cur = pair(anth_req(), grow(anth_req(tools=tools)))
    r = xray.explain_cache(prev, cur)
    assert r["cause"] == "tools_changed"
    assert "tool definitions changed: tool 'Read' description edited" in r["detail"]
    assert "'…" in r["detail"] and "very carefully" in r["detail"]
    assert r["where"]["section"] == "tools" and r["where"]["index"] == 0


def test_tool_schema_edited_names_the_path():
    tools = copy.deepcopy(TOOLS)
    tools[1]["input_schema"]["properties"]["arg_1"]["description"] = "changed"
    r = xray.explain_cache(*pair(anth_req(), grow(anth_req(tools=tools))))
    assert "tool 'Bash' input schema (input_schema.properties.arg_1.description) edited" in r["detail"]


def test_tool_order_changed():
    tools = copy.deepcopy(TOOLS)
    tools[0], tools[2] = tools[2], tools[0]
    r = xray.explain_cache(*pair(anth_req(), grow(anth_req(tools=tools))))
    assert r["cause"] == "tool_order_changed"
    assert "tool order changed: 'Read' moved from position 0 to 2" in r["detail"]


def test_mcp_server_connected():
    tools = copy.deepcopy(TOOLS) + [tool("mcp__linear__create_ticket", 10), tool("mcp__linear__search", 10)]
    r = xray.explain_cache(*pair(anth_req(), grow(anth_req(tools=tools))))
    assert r["cause"] == "tools_changed"
    assert "2 added: 'mcp__linear__create_ticket', 'mcp__linear__search'" in r["detail"]
    assert "MCP server 'linear' connected" in r["detail"]


def test_history_message_edited():
    cur_req = grow(anth_req())
    cur_req["messages"][1]["content"][0]["text"] = "I'll look at the test file first."
    r = xray.explain_cache(*pair(anth_req(), cur_req, cur_usage={"inp": 50, "cr": 6_000, "cw": 4_600}))
    assert r["verdict"] == "partial" and r["cause"] == "history_edited"
    assert "a message earlier in history was edited (message 1, assistant)" in r["detail"]
    assert r["where"]["section"] == "messages" and r["where"]["index"] == 1
    assert 3_800 < r["lost_tokens"] <= 4_000           # the edited message itself could not have been cached


def test_history_truncated():
    cur_req = anth_req(messages=HISTORY[:1] + [NEW_TURN])
    r = xray.explain_cache(*pair(anth_req(), cur_req))
    assert r["cause"] in ("history_edited", "history_truncated")
    cur_req = anth_req(messages=HISTORY[:2])
    r = xray.explain_cache(*pair(anth_req(), cur_req, cur_usage={"inp": 50, "cr": 0, "cw": 8_000}))
    assert r["cause"] == "history_truncated" and "5 -> 2 messages" in r["detail"]


def test_ttl_expired_vs_one_hour_ttl():
    prev, cur = pair(anth_req(), grow(anth_req()), gap=11 * 60)
    r = xray.explain_cache(prev, cur)
    assert r["cause"] == "ttl_expired" and "cache expired: 11 min" in r["detail"] and r["ttl_s"] == 300
    assert r["changes"] == [] and r["lost_usd"] > 0
    # with a 1-hour cache breakpoint, the same 11-minute gap is within the TTL
    req1h = anth_req()
    req1h["system"][0]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    r = xray.explain_cache(*pair(req1h, grow(req1h), gap=11 * 60))
    assert r["ttl_s"] == 3600 and r["cause"] == "unknown"
    r = xray.explain_cache(*pair(req1h, grow(req1h), gap=11 * 60, cur_usage={"inp": 30, "cr": 10_000, "cw": 600}))
    assert r["verdict"] == "hit"


def test_ttl_expired_mentions_prefix_change_too():
    cur_req = grow(anth_req(system=SYSTEM.replace("10:31", "10:52")))
    r = xray.explain_cache(*pair(anth_req(), cur_req, gap=21 * 60))
    assert r["cause"] == "ttl_expired" and "the prefix also changed: system prompt changed" in r["detail"]


def test_model_changed():
    prev, cur = pair(anth_req(), grow(anth_req(model="claude-opus-4-5")))
    cur.model = "claude-opus-4-5"
    r = xray.explain_cache(prev, cur)
    assert r["cause"] == "model_changed" and "claude-sonnet-4-5 -> claude-opus-4-5" in r["detail"]


def test_no_cache_and_none_cases():
    prev, cur = pair(anth_req(), grow(anth_req()), prev_usage={"inp": 9000}, cur_usage={"inp": 9500})
    r = xray.explain_cache(prev, cur)
    assert r["verdict"] == "no_cache" and r["lost_usd"] == 0 and "cache_control" in r["detail"]
    assert xray.explain_cache(None, cur) is None
    assert xray.explain_cache(prev, None) is None
    assert xray.explain_cache(cur, prev) is None                                   # cur earlier than prev
    est = mk(grow(anth_req()), ts=T0 + 30, inp=9500, estimated=True)
    assert xray.explain_cache(prev, est) is None


def test_openai_automatic_caching():
    rules = " ".join(f"Rule {i}: answer questions about topic number {i} politely." for i in range(110))
    req = {"model": "gpt-4.1", "messages": [{"role": "system", "content": rules},       # one long line
                                            {"role": "user", "content": "hi"}]}
    nxt = copy.deepcopy(req)
    nxt["messages"] += [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}]
    kw = {"style": "openai", "model": "gpt-4.1"}
    hit = xray.explain_cache(mk(req, inp=1800, **kw), mk(nxt, ts=T0 + 20, inp=1820, cr=1792, **kw))
    assert hit["verdict"] == "hit"
    changed = copy.deepcopy(nxt)
    changed["messages"][0]["content"] = "Rules! " + changed["messages"][0]["content"]
    r = xray.explain_cache(mk(req, inp=1800, **kw), mk(changed, ts=T0 + 20, inp=1820, cr=0, **kw))
    assert r["verdict"] == "miss" and r["cause"] == "system_changed"
    assert "system prompt changed (message 0) at char 4: 'Rule 0: answer" in r["detail"]
    assert "-> 'Rules! Rule 0: answer" in r["detail"]
    p = price_for("gpt-4.1")
    # ~1,800 tokens (less the edited chunk) billed at the input price instead of the cached price
    assert r["lost_usd"] == pytest.approx(r["lost_tokens"] * (p["input"] - p["cache_read"]) / 1e6)
    assert 1700 < r["lost_tokens"] <= 1800
    small = xray.explain_cache(mk(req, inp=900, **kw), mk(nxt, ts=T0 + 20, inp=920, **kw))
    assert small["verdict"] == "no_cache" and "1,024" in small["detail"]
    ttl = xray.explain_cache(mk(req, inp=1800, **kw), mk(nxt, ts=T0 + 3000, inp=1820, **kw))
    assert ttl["cause"] == "ttl_expired"


def test_unparseable_bodies_still_get_a_verdict():
    prev = mk("{broken", ts=T0, inp=50, cw=10_000)
    cur = mk("{broken too", ts=T0 + 5, inp=50, cw=10_000)
    r = xray.explain_cache(prev, cur)
    assert r["verdict"] == "miss" and r["cause"] == "unknown" and "cannot be located" in r["detail"]


def test_shared_system_prompt_with_different_questions_is_not_a_miss():
    """Single-shot apps: same cached system prompt, a different (large) question each time."""
    system = "You answer questions about the ACME handbook. " * 120
    kw = {"style": "openai", "model": "gpt-4.1"}
    q1 = {"model": "gpt-4.1", "messages": [{"role": "system", "content": system},
                                           {"role": "user", "content": "retrieved context A. " * 400}]}
    q2 = {"model": "gpt-4.1", "messages": [{"role": "system", "content": system},
                                           {"role": "user", "content": "retrieved context B! " * 400}]}
    r = xray.explain_cache(mk(q1, inp=3500, **kw), mk(q2, ts=T0 + 30, inp=3500, cr=1408, **kw))
    assert r["verdict"] == "hit", r
    assert r["expected_cached_tokens"] < 1800                       # only the shared system prompt was cacheable


# ------------------------------------------------------------------ cache_report


def session_sequence() -> list[Exchange]:
    r1 = anth_req()
    r2 = grow(r1)
    r3 = grow(anth_req(system=SYSTEM.replace("10:31", "10:33"), messages=r2["messages"]))
    r4 = grow(r3)
    sub = anth_req(system="You are a file search specialist. " * 30, tools=TOOLS[:2],
                   messages=[{"role": "user", "content": "find config files"}])
    tools_swapped = copy.deepcopy(TOOLS)
    tools_swapped[3], tools_swapped[4] = tools_swapped[4], tools_swapped[3]
    r5 = grow(anth_req(system=SYSTEM.replace("10:31", "10:33"), tools=tools_swapped, messages=r4["messages"]))
    return [
        mk(r1, ts=T0, inp=40, cw=10_000),
        mk(r2, ts=T0 + 20, inp=30, cr=10_000, cw=500),                         # hit
        mk(sub, ts=T0 + 25, inp=10, cw=3_000, model="claude-haiku-4-5"),       # interleaved sub-agent: new thread
        mk(r3, ts=T0 + 40, inp=30, cr=0, cw=10_900),                           # system timestamp miss
        mk(r4, ts=T0 + 60, inp=30, cr=10_900, cw=300),                         # hit
        mk(r5, ts=T0 + 80, inp=30, cr=0, cw=11_300),                           # tool order miss
        mk(grow(r5), ts=T0 + 80 + 1200, inp=30, cr=0, cw=11_500),              # 20 min later: TTL miss
        mk(r1, ts=T0 + 90, inp=40, cw=10_000, source="cassette"),              # replay: ignored
    ]


def test_cache_report_session():
    rep = xray.cache_report(session_sequence())
    assert rep["pairs"] == 5 and rep["hits"] == 2 and rep["total_misses"] == 3
    causes = [m["cause"] for m in rep["misses"]]
    assert sorted(causes) == ["system_changed", "tool_order_changed", "ttl_expired"]
    costs = [m["lost_usd"] for m in rep["misses"]]
    assert costs == sorted(costs, reverse=True) and all(c > 0 for c in costs)
    assert rep["total_lost_usd"] == pytest.approx(sum(costs))
    assert rep["by_cause"]["system_changed"]["count"] == 1
    assert all(m["session"] == "s1" for m in rep["misses"])
    tool_miss = next(m for m in rep["misses"] if m["cause"] == "tool_order_changed")
    assert "'mcp__github__create_issue' moved from position 3 to 4" in tool_miss["detail"]


def test_cache_report_untagged_groups_by_model_and_splits_on_gap():
    r1 = anth_req()
    exs = [mk(r1, ts=T0, inp=40, cw=10_000, session=None),
           mk(grow(r1), ts=T0 + 2 * 3600, inp=40, cw=10_500, session=None),     # > 1 h later: new session
           mk(grow(grow(r1)), ts=T0 + 2 * 3600 + 30, inp=40, cr=0, cw=10_800, session=None)]
    rep = xray.cache_report(exs)
    assert rep["pairs"] == 1 and rep["total_misses"] == 1
    # with an explicit session tag the 2-hour gap is compared (and explained as an expiry)
    exs = [mk(r1, ts=T0, inp=40, cw=10_000), mk(grow(r1), ts=T0 + 2 * 3600, inp=40, cw=10_500)]
    rep = xray.cache_report(exs)
    assert rep["total_misses"] == 1 and rep["misses"][0]["cause"] == "ttl_expired"
    assert "2.0 h" in rep["misses"][0]["detail"]


def test_cache_report_empty_and_garbage():
    assert xray.cache_report([])["pairs"] == 0
    rep = xray.cache_report([mk(b, ts=T0 + i, inp=10, cw=5000) for i, b in enumerate(MALFORMED)])
    assert isinstance(rep["misses"], list)


# ------------------------------------------------------------------ 4. renderers


def test_render_xray_text():
    prev = mk(anth_req(), ts=T0, inp=40, cw=10_000)
    cur = mk(grow(anth_req(system=SYSTEM.replace("10:31", "10:32"))), ts=T0 + 30, inp=40, cw=10_600)
    out = xray.render_xray_text(cur, prev=prev)
    assert "\x1b[" not in out
    for s in ("input 10.6k tokens", "system", "tools", "history", "new input", "tool definitions: 6 tools",
              "Bash", "MCP: github 2 tools", "cache: MISS", "system prompt changed at char"):
        assert s in out, s
    assert "█" in out and "$" in out
    colored = xray.render_xray_text(cur, color=True, prev=prev)
    assert "\x1b[" in colored
    hit = mk(grow(anth_req()), ts=T0 + 30, inp=40, cr=10_000, cw=600)
    assert "cache: HIT" in xray.render_xray_text(hit, prev=prev)
    est = xray.render_xray_text(mk(anth_req()))
    assert "(estimated)" in est and "~" in est


def test_render_xray_text_escapes_control_chars_in_tool_names():
    tools = [tool("evil\x1b]0;pwned\x07", 3)]
    out = xray.render_xray_text(mk(anth_req(tools=tools), inp=3000))
    assert "\x1b" not in out and "\x07" not in out


def test_summary_json_and_text():
    exs = session_sequence()
    data = xray.summary_json(exs)
    json.dumps(data)                                                           # JSON-safe
    assert data["requests"] == len(exs) and data["analyzed"] == 7
    assert set(data["parts"]) == {"system", "tools", "history", "new_input"}
    assert abs(sum(p["share"] for p in data["parts"].values()) - 1) < 1e-9
    assert data["avg_input_tokens"] == round(data["total_input_tokens"] / 7)
    assert data["input_usd"] == pytest.approx(sum(p["usd"] for p in data["parts"].values()))
    assert data["tool_bloat"]["servers"][0]["server"] == "github"
    assert data["cache"]["total_misses"] == 3
    text = xray.render_summary_text(exs)
    for s in ("7 requests", "where the input goes", "tool definitions", "MCP tools from 'github'",
              "prompt cache", "3 misses", "lost", "system prompt changed at char", "tool order changed",
              "cache expired", "ttl_expired"):
        assert s in text, s
    assert "\x1b[" not in text and "\x1b[" in xray.render_summary_text(exs, color=True)
    assert "no billable requests" in xray.render_summary_text([])


def test_works_on_rows_read_back_from_the_store():
    s = Store(":memory:")
    try:
        for ex in session_sequence():
            s.record(ex)
        rows = list(s.iter_light())
        assert xray.cache_report(rows)["total_misses"] == 3
        assert xray.breakdown(s.get(rows[0].id))["parsed"]
    finally:
        s.close()
