"""Tests for llmtap.whatif: repricing recorded usage on other models (cross-style cache accounting, long-context
tiers, skipped calls, unknown models), candidate comparison, the prompt-cache what-if and the renderers."""
from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest

from llmtap import whatif, xray
from llmtap.models import Exchange, Usage
from llmtap.pricing import PRICES, cost_usd, price_for
from llmtap.store import Store
from llmtap.term import strip_ansi

T0 = 1_795_000_000.0
M = 1_000_000


def call(model: str = "claude-opus-5-5", style: str | None = None, *, inp: int = 0, out: int = 0, cr: int = 0,
         cw: int = 0, estimated: bool = False, cost: float | None = None, source: str = "upstream",
         status: int = 200, ts: float = T0, body: Any = "", session: str | None = None, **usage_extra: Any) -> Exchange:
    """An exchange with usage in ``style``'s convention, priced like the proxy would record it."""
    style = style or ("anthropic" if model.startswith("claude") else "openai")
    u = Usage(input_tokens=inp, output_tokens=out, cache_read_tokens=cr, cache_write_tokens=cw, estimated=estimated,
              **usage_extra)
    if cost is None:
        cost = 0.0 if estimated else cost_usd(model, u, style=style)
    return Exchange(ts=ts, provider=style, style=style, model=model, status=status, usage=u, cost_usd=cost,
                    source=source, request_body=body if isinstance(body, str) else json.dumps(body),
                    tags={"session": session} if session else {})


def usd(tokens: int, rate: float) -> float:
    return tokens * rate / M


# ------------------------------------------------------------------ reprice: same model


@pytest.mark.parametrize("ex", [
    call("claude-opus-5-5", inp=1_000, out=500, cr=20_000, cw=2_000),
    call("claude-sonnet-5-5", "bedrock", inp=3_000, out=100, cr=5_000, cw=700),
    call("gpt-5", inp=23_000, out=500, cr=20_000),
    call("gemini-2.5-flash", "gemini", inp=9_000, out=300, cr=4_000),
    call("mistral-large", inp=5_000, out=100, cr=2_000),            # no published cache price: same model keeps it
    call("claude-sonnet-5-5", "openai", inp=23_000, out=500, cr=20_000),   # Claude through an OpenAI-style router
])
def test_reprice_on_the_recorded_model_matches_actual(ex: Exchange) -> None:
    r = whatif.reprice([ex], ex.model)
    assert r["ok"] and r["calls"] == 1
    assert r["whatif_usd"] == pytest.approx(ex.cost_usd) == pytest.approx(r["actual_usd"])
    assert r["diff_usd"] == pytest.approx(0) and r["diff_pct"] == pytest.approx(0)
    assert r["baseline_usd"] == pytest.approx(ex.cost_usd)
    assert not any("differs from the same usage" in n for n in r["notes"])


# ------------------------------------------------------------------ reprice: cross-style accounting


def test_anthropic_usage_on_openai_target() -> None:
    # Anthropic: 1,000 uncached + 20,000 cache reads + 2,000 cache writes (all exclusive)
    ex = call("claude-opus-5-5", inp=1_000, out=500, cr=20_000, cw=2_000)
    r = whatif.reprice([ex], "gpt-5")
    p = price_for("gpt-5")
    # OpenAI-style: prompt 23,000 of which 20,000 cached; the cache writes become normal input
    want = usd(3_000, p["input"]) + usd(20_000, p["cache_read"]) + usd(500, p["output"])
    assert r["whatif_usd"] == pytest.approx(want)
    assert r["accounting"] == "openai" and r["cache_reads_carry_over"]
    assert r["tokens"] == {"input": 1_000, "cache_read": 20_000, "cache_write": 2_000, "output": 500}
    assert r["actual_usd"] == pytest.approx(ex.cost_usd)
    assert r["diff_usd"] == pytest.approx(want - ex.cost_usd)
    assert r["diff_pct"] == pytest.approx(round(100 * (want - ex.cost_usd) / ex.cost_usd, 1))
    assert any("Anthropic cache-write tokens become normal input" in n for n in r["notes"])


def test_openai_usage_on_anthropic_target() -> None:
    # OpenAI: prompt_tokens 23,000 INCLUDING 20,000 cached
    ex = call("gpt-5", inp=23_000, out=500, cr=20_000)
    r = whatif.reprice([ex], "claude-sonnet-5-5")
    p = price_for("claude-sonnet-5-5")
    want = usd(3_000, p["input"]) + usd(20_000, p["cache_read"]) + usd(500, p["output"])   # no writes invented
    assert r["whatif_usd"] == pytest.approx(want)
    assert r["accounting"] == "anthropic"
    assert r["tokens"] == {"input": 3_000, "cache_read": 20_000, "cache_write": 0, "output": 500}
    assert any("OpenAI-style calls report no billable cache writes" in n for n in r["notes"])


def test_round_trip_between_styles_preserves_prompt() -> None:
    """Anthropic -> OpenAI-style target and back: both see the same 23,000-token prompt."""
    a = call("claude-sonnet-5-5", inp=1_000, out=10, cr=20_000, cw=2_000)
    o = call("gpt-5", inp=23_000, out=10, cr=20_000)
    ra, ro = whatif.reprice([a], "gpt-5"), whatif.reprice([o], "gpt-5")
    assert ra["whatif_usd"] == pytest.approx(ro["whatif_usd"])         # writes are unbilled input on OpenAI
    # on Claude, the Anthropic call keeps its 2,000 cache writes; the OpenAI call bills them as plain input
    p = price_for("claude-sonnet-5-5")
    assert whatif.reprice([a], "claude-sonnet-5-5")["whatif_usd"] - whatif.reprice([o], "claude-sonnet-5-5")[
        "whatif_usd"] == pytest.approx(usd(2_000, p["cache_write"] - p["input"]))


def test_openai_style_cache_writes_are_not_billed_on_claude() -> None:
    """OpenRouter-style usage reporting cache writes inside prompt_tokens: never turned into Claude cache writes."""
    ex = call("gpt-5", inp=10_000, out=0, cr=0, cw=4_000)
    r = whatif.reprice([ex], "claude-opus-5-5")
    assert r["tokens"]["cache_write"] == 0 and r["tokens"]["input"] == 10_000
    assert r["whatif_usd"] == pytest.approx(usd(10_000, price_for("claude-opus-5-5")["input"]))


def test_exclusive_cached_count_in_openai_style() -> None:
    """cached > prompt_tokens: the numbers are clearly exclusive (same rule as pricing.cost_usd)."""
    ex = call("gpt-5", inp=1_000, out=0, cr=5_000)
    r = whatif.reprice([ex], "claude-opus-5-5")
    p = price_for("claude-opus-5-5")
    assert r["tokens"]["input"] == 1_000 and r["tokens"]["cache_read"] == 5_000
    assert r["whatif_usd"] == pytest.approx(usd(1_000, p["input"]) + usd(5_000, p["cache_read"]))


def test_cache_reads_billed_as_input_without_published_cache_price() -> None:
    ex = call("gpt-5", inp=23_000, out=0, cr=20_000)
    assert "cache_read" not in whatif.pricing._RAW["mistral-large"]
    r = whatif.reprice([ex], "mistral-large")
    assert not r["cache_reads_carry_over"]
    assert r["whatif_usd"] == pytest.approx(usd(23_000, price_for("mistral-large")["input"]))
    assert any("no published cache-read price" in n for n in r["notes"])
    assert any("1 call(s) had cache reads that were billed as normal input" in n for n in r["notes"])
    # Claude models without an explicit cache price still use Anthropic's universal 10% cache-read rate
    r = whatif.reprice([ex], "claude-haiku-5-5")
    assert r["cache_reads_carry_over"]
    p = price_for("claude-haiku-5-5")
    assert r["whatif_usd"] == pytest.approx(usd(3_000, p["input"]) + usd(20_000, p["cache_read"]))


def test_long_context_tier_applies_on_target() -> None:
    # 150K-token prompt (Anthropic exclusive counts) -> Haiku 5.5's >100K rate card for the whole request
    ex = call("claude-opus-5-5", inp=10_000, out=1_000, cr=140_000)
    r = whatif.reprice([ex], "claude-haiku-5-5")
    threshold, long = whatif.LONG_CONTEXT["claude-haiku-5-5"]
    want = usd(10_000, long["input"]) + usd(140_000, long["cache_read"]) + usd(1_000, long["output"])
    assert r["whatif_usd"] == pytest.approx(want)
    assert any(f"above {threshold:,} tokens" in n for n in r["notes"])
    # the same prompt reported OpenAI-style (cached inside input) crosses the threshold too
    o = call("gpt-5", inp=150_000, out=1_000, cr=140_000)
    assert whatif.reprice([o], "claude-haiku-5-5")["whatif_usd"] == pytest.approx(want)
    # below the threshold: short rates, no note
    small = whatif.reprice([call("claude-opus-5-5", inp=1_000, out=10)], "claude-haiku-5-5")
    assert not any("long-context" in n for n in small["notes"])


def test_free_local_target_costs_nothing() -> None:
    r = whatif.reprice([call("gpt-5", inp=10_000, out=1_000)], "llama3:8b")
    assert r["ok"] and r["whatif_usd"] == 0 and r["diff_pct"] == -100.0
    assert any("$0" in n for n in r["notes"])


def test_overrides_price_target_and_actual() -> None:
    over = {"my-model": {"input": 1.0, "output": 2.0}, "acme-cache": {"input": 1.0, "output": 2.0, "cache_read": 0.1}}
    ex = call("gpt-5", inp=20_000, out=1_000, cr=10_000)
    r = whatif.reprice([ex], "my-model", over)
    assert r["ok"] and r["priced_as"] == "my-model" and not r["cache_reads_carry_over"]
    assert r["whatif_usd"] == pytest.approx(usd(20_000, 1.0) + usd(1_000, 2.0))
    assert any("pricing_overrides" in n for n in r["notes"])
    r = whatif.reprice([ex], "acme-cache", over)
    assert r["cache_reads_carry_over"]
    assert r["whatif_usd"] == pytest.approx(usd(10_000, 1.0) + usd(10_000, 0.1) + usd(1_000, 2.0))
    # without the override the model is unknown
    assert not whatif.reprice([ex], "my-model")["ok"]


def test_priced_as_shows_prefix_match() -> None:
    r = whatif.reprice([call("gpt-5", inp=100, out=10)], "gpt-4o-mini-2024-07-18")
    assert r["ok"] and r["priced_as"] == "gpt-4o-mini"
    assert "priced as gpt-4o-mini" in whatif.render_text(r)


def test_modifiers_and_recorded_cost_mismatch_are_noted() -> None:
    u_kw = {"speed": "fast"}
    ex = call("claude-opus-5-5", inp=1_000, out=1_000, **u_kw)         # recorded at 2x (fast mode)
    assert ex.cost_usd == pytest.approx(2 * cost_usd("claude-opus-5-5", Usage(input_tokens=1_000, output_tokens=1_000)))
    r = whatif.reprice([ex], "claude-opus-5-5")
    assert r["whatif_usd"] == pytest.approx(ex.cost_usd / 2) and r["diff_pct"] == pytest.approx(-50)
    assert any("fast mode" in n and "standard rates" in n for n in r["notes"])
    assert any("differs from the same usage at today's table" in n for n in r["notes"])


def test_zero_recorded_cost_falls_back_to_todays_price() -> None:
    ex = call("claude-opus-5-5", inp=1_000, out=1_000, cost=0.0)
    r = whatif.reprice([ex], "claude-opus-5-5")
    want = cost_usd("claude-opus-5-5", ex.usage)
    assert want > 0 and r["actual_usd"] == pytest.approx(want)
    assert r["diff_usd"] == pytest.approx(0)
    assert any("recorded with a $0 cost" in n for n in r["notes"])


# ------------------------------------------------------------------ reprice: skipped / ignored / errors


def test_skipped_and_ignored_calls() -> None:
    exs = [
        call("claude-opus-5-5", inp=1_000, out=100),                            # included
        call("claude-opus-5-5", inp=1_000, out=100, estimated=True, cost=0.5),  # estimated -> skipped
        call("claude-opus-5-5"),                                                # no usage -> skipped
        call("totally-unknown-model", inp=1_000, out=100, cost=0.0),           # not priced -> skipped
        call("claude-opus-5-5", inp=1_000, out=100, source="cassette"),         # replay -> ignored
        call("claude-opus-5-5", inp=1_000, out=100, status=500),                # failed -> ignored
        call("claude-opus-5-5", inp=1_000, out=100, source="blocked"),
    ]
    r = whatif.reprice(exs, "gpt-5")
    assert r["calls"] == 1
    assert r["skipped"] == {"no_usage": 1, "estimated": 1, "unknown_price": 1} and r["skipped_total"] == 3
    assert r["skipped_actual_usd"] == pytest.approx(0.5)
    assert r["ignored"] == 3
    assert r["actual_usd"] == pytest.approx(exs[0].cost_usd)                # skipped cost excluded
    assert any("3 call(s) skipped" in n for n in r["notes"])
    text = whatif.render_text(r)
    assert "3 calls skipped: 1 no usage, 1 estimated usage, 1 model not in price table" in text


def test_unknown_target_is_an_error_result_with_suggestions() -> None:
    exs = [call("claude-opus-5-5", inp=1_000, out=100)]
    r = whatif.reprice(exs, "claude-opux-5-5")
    assert r["ok"] is False and r["whatif_usd"] is None
    assert "unknown model 'claude-opux-5-5'" in r["error"]
    assert "claude-opus-5-5" in r["suggestions"]
    assert r["calls"] == 1                                                  # still says what was looked at
    r = whatif.reprice(exs, "opus-5.5")
    assert "claude-opus-5-5" in r["suggestions"]
    assert "did you mean" in whatif.render_text(r)
    for bad in (None, "", "   ", 42):
        r = whatif.reprice(exs, bad)                                        # type: ignore[arg-type]
        assert r["ok"] is False and r["error"]
    assert whatif.reprice(exs, "zzzzqqq")["suggestions"] == []


def test_empty_input() -> None:
    for empty in ([], None, iter(())):
        r = whatif.reprice(empty, "gpt-5")                                  # type: ignore[arg-type]
        assert r["ok"] and r["calls"] == 0 and r["actual_usd"] == 0 and r["whatif_usd"] == 0
        assert r["diff_pct"] is None and r["by_model"] == []
    assert "no upstream calls with recorded usage since today" in whatif.render_text(
        whatif.reprice([], "gpt-5"), period="since today")
    res = whatif.compare([])
    assert res and all(x["ok"] and x["whatif_usd"] == 0 and not x["used"] for x in res)
    assert whatif.render_text(res).startswith("what if: no upstream calls")
    c = whatif.cache_savings([])
    assert c["pairs"] == 0 and c["fixable_usd"] == 0 and c["fixable_pct"] is None
    assert "no follow-up requests" in whatif.render_text(c)


def test_odd_data_never_raises() -> None:
    nan_cost = call("claude-opus-5-5", inp=100, out=10)
    nan_cost.cost_usd = float("nan")
    str_cost = call("claude-opus-5-5", inp=100, out=10)
    str_cost.cost_usd = "abc"                                               # type: ignore[assignment]
    dict_usage = call("claude-opus-5-5")
    dict_usage.usage = {"input_tokens": "500", "output_tokens": 5.7, "bogus": 1}   # type: ignore[assignment]
    neg = call("claude-opus-5-5", inp=-100, out=-5, cr=-1)
    none_usage = call("claude-opus-5-5")
    none_usage.usage = None                                                 # type: ignore[assignment]
    weird_status = call("gpt-5", inp=100, out=1)
    weird_status.status = "200"                                             # type: ignore[assignment]
    no_model = call("gpt-5", inp=100, out=1)
    no_model.model = None
    exs: list[Any] = [nan_cost, str_cost, dict_usage, neg, none_usage, weird_status, no_model, None, object(), 7]
    r = whatif.reprice(exs, "gpt-5")
    assert r["ok"] and math.isfinite(r["whatif_usd"]) and math.isfinite(r["actual_usd"])
    assert r["skipped"]["unknown_price"] >= 1 and r["skipped"]["no_usage"] >= 2
    assert r["by_model"][0]["tokens"]["input"] >= 500                       # dict usage was read
    whatif.compare(exs)
    whatif.cache_savings(exs)
    json.loads(whatif.to_json(r))
    whatif.render_text(r)
    assert whatif.render_text(None) == "what if: nothing to show"
    assert whatif.render_text("garbage") == "what if: nothing to show"     # type: ignore[arg-type]


def test_accepts_a_generator_once() -> None:
    exs = [call("claude-opus-5-5", inp=1_000, out=100) for _ in range(3)]
    r = whatif.compare((e for e in exs), ["gpt-5", "claude-opus-5-5"])
    assert [x["calls"] for x in r] == [3, 3]


# ------------------------------------------------------------------ compare


def mixed_period() -> list[Exchange]:
    return ([call("claude-opus-5-5", inp=1_000, out=500, cr=20_000, cw=2_000, ts=T0 + i) for i in range(38)]
            + [call("claude-haiku-5-5", inp=3_000, out=200, ts=T0 + 100 + i) for i in range(3)]
            + [call("claude-opus-5-5", inp=1_000, out=10, estimated=True)])


def test_compare_default_candidates_sorted_and_marked() -> None:
    exs = mixed_period()
    res = whatif.compare(exs)
    costs = [r["whatif_usd"] for r in res]
    assert costs == sorted(costs)
    assert all(r["ok"] for r in res)
    names = [r["target"] for r in res]
    assert len(names) == len(set(names))
    for n in names:                                                         # all from the price table
        assert price_for(n) is not None and whatif._target_meta(n, None)["key"] in PRICES
    used = {r["target"]: r["used_calls"] for r in res if r["used"]}
    assert used == {"claude-opus-5-5": 38, "claude-haiku-5-5": 3}
    # same base for every row
    assert len({(r["calls"], r["actual_usd"]) for r in res}) == 1 and res[0]["calls"] == 41
    # cheaper models in the default list: each family's latest version only
    assert "claude-sonnet-5-5" in names and "claude-sonnet-4-5" not in names and "gpt-4o" not in names


def test_default_candidates_follow_the_price_table() -> None:
    fams = whatif._family_latest()
    assert fams and all(k in PRICES for k in fams)
    assert "claude-opus-5-5" in fams and "claude-opus-4-8" not in fams and "claude-opus-5" not in fams
    assert "gpt-5.2" in fams and "gpt-5-mini" in fams and "gpt-5-nano" in fams and "gpt-5" not in fams
    assert "gemini-3-pro" in fams and "gemini-2.5-flash" in fams and "gemini-2.5-flash-lite" in fams
    # peers or cheaper: a gpt-5-nano user is not shown Opus
    cands = whatif.default_candidates([call("gpt-5-nano", inp=1_000, out=100)])
    assert cands[0] == "gpt-5-nano" and "claude-opus-5-5" not in cands
    assert "gemini-2.5-flash-lite" in cands
    assert whatif.default_candidates() == fams


def test_compare_dated_model_id_is_deduplicated() -> None:
    exs = [call("claude-opus-5-5-20261001", inp=1_000, out=100)]
    res = whatif.compare(exs)
    names = [r["target"] for r in res]
    assert "claude-opus-5-5-20261001" in names and "claude-opus-5-5" not in names
    used = [r for r in res if r["used"]]
    assert len(used) == 1 and used[0]["diff_usd"] == pytest.approx(0)


def test_compare_explicit_models_with_unknown_last() -> None:
    exs = mixed_period()
    res = whatif.compare(exs, ["claude-opus-5-5", "nope-model", "gpt-5-nano", "claude-sonnet-5-5", "gpt-5-nano"])
    assert [r["target"] for r in res] == ["gpt-5-nano", "claude-sonnet-5-5", "claude-opus-5-5", "nope-model"]
    assert res[-1]["ok"] is False
    text = whatif.render_text(res)
    assert "nope-model" in text and "unknown model" in text
    assert whatif.compare(exs, []) == []


def test_compare_render_matches_the_example_layout() -> None:
    exs = mixed_period()
    text = whatif.render_text(whatif.compare(exs, ["claude-sonnet-5-5", "claude-opus-5-5"]),
                              period="since today")
    lines = text.splitlines()
    actual = sum(e.cost_usd for e in exs[:-1])
    assert lines[0] == (f"what if: 41 calls since today, actual {whatif.fmt_usd(actual)} "
                        f"(claude-opus-5-5 x38, claude-haiku-5-5 x3)")
    assert lines[1].lstrip().startswith("claude-sonnet-5-5") and "%" in lines[1]
    assert "(used x38)" in lines[2]
    assert "1 call skipped: 1 estimated usage" in lines[3]
    assert "\x1b[" not in text
    colored = whatif.render_text(whatif.compare(exs, ["claude-sonnet-5-5"]), color=True)
    assert "\x1b[" in colored and strip_ansi(colored).startswith("what if:")


def test_render_escapes_control_characters_in_model_names() -> None:
    evil = call("gpt-5\x1b]0;pwned\x07", inp=100, out=10)
    text = whatif.render_text(whatif.reprice([evil], "gpt-5"))
    assert "\x1b" not in text and "\\x1b" in text


# ------------------------------------------------------------------ cache what-if

SYSTEM = (
    "You are a coding assistant that helps users with software engineering tasks.\n"
    + "Follow the repository conventions and keep changes minimal. " * 40
    + "\n<env>\nWorking directory: /home/dev/project\nToday's date: 2026-11-24 10:31\n</env>\n"
    + "IMPORTANT: never commit secrets. " * 10
)


def tool(name: str, words: int) -> dict[str, Any]:
    return {"name": name, "description": f"{name}: " + "Performs the operation described carefully. " * words,
            "input_schema": {"type": "object", "properties": {"arg_0": {"type": "string"}}, "required": ["arg_0"]}}


TOOLS = [tool("Read", 30), tool("Bash", 60), tool("Edit", 20), tool("Grep", 25), tool("Glob", 15)]
HISTORY = [
    {"role": "user", "content": "Please fix the failing test in tests/test_api.py"},
    {"role": "assistant", "content": [{"type": "text", "text": "I'll look at the test first."}]},
    {"role": "user", "content": [{"type": "text", "text": "def test_api():\n    assert f() == 2\n" * 40,
                                  "cache_control": {"type": "ephemeral"}}]},
]


def req(system: str = SYSTEM, tools: list | None = None, messages: list | None = None) -> dict[str, Any]:
    return {"model": "claude-sonnet-5-5", "max_tokens": 4096,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools": copy.deepcopy(TOOLS if tools is None else tools),
            "messages": copy.deepcopy(HISTORY if messages is None else messages)}


def grow(r: dict[str, Any], text: str = "next") -> dict[str, Any]:
    nxt = copy.deepcopy(r)
    for m in nxt["messages"]:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                b.pop("cache_control", None)
    nxt["messages"] += [{"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
                        {"role": "user", "content": [{"type": "text", "text": text,
                                                      "cache_control": {"type": "ephemeral"}}]}]
    return nxt


def cache_session() -> list[Exchange]:
    def c(body: dict, ts: float, inp: int, cr: int, cw: int, **kw: Any) -> Exchange:
        return call("claude-sonnet-5-5", inp=inp, out=100, cr=cr, cw=cw, ts=ts, body=body, session="s1", **kw)

    r1 = req()
    r2 = grow(r1)
    r3 = grow(req(system=SYSTEM.replace("10:31", "10:33"), messages=r2["messages"]))     # timestamp changed
    r4 = grow(r3)
    swapped = copy.deepcopy(TOOLS)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    r5 = grow(req(system=SYSTEM.replace("10:31", "10:33"), tools=swapped, messages=r4["messages"]))  # tool order
    return [
        c(r1, T0, 40, 0, 10_000),
        c(r2, T0 + 20, 30, 10_000, 500),               # hit
        c(r3, T0 + 40, 30, 0, 10_900),                 # system prompt timestamp miss
        c(r4, T0 + 60, 30, 10_900, 300),               # hit
        c(r5, T0 + 80, 30, 0, 11_300),                 # tool order miss
        c(grow(r5), T0 + 80 + 1200, 30, 0, 11_500),    # 20 min later: TTL expiry
        c(r1, T0 + 90, 40, 0, 10_000, source="cassette"),   # replay: ignored
    ]


def test_cache_savings_fixable_vs_expiry() -> None:
    exs = cache_session()
    rep = xray.cache_report(exs)
    by_cause = {m["cause"]: m["lost_usd"] for m in rep["misses"]}
    assert set(by_cause) == {"system_changed", "tool_order_changed", "ttl_expired"}

    c = whatif.cache_savings(exs)
    assert c["calls"] == 6 and c["pairs"] == 5 and c["hits"] == 2 and c["misses"] == 3
    assert set(c["fixable"]) == {"system_changed", "tool_order_changed"}
    assert c["fixable"]["system_changed"]["count"] == 1 and c["fixable"]["system_changed"]["volatile"] == 1
    assert c["fixable_usd"] == pytest.approx(by_cause["system_changed"] + by_cause["tool_order_changed"])
    assert c["expiries"] == {"count": 1, "lost_usd": pytest.approx(by_cause["ttl_expired"])}
    assert c["expiry_usd"] == pytest.approx(by_cause["ttl_expired"]) and c["other"] == {}
    actual = sum(e.cost_usd for e in exs if e.source == "upstream")
    assert c["actual_usd"] == pytest.approx(actual)
    assert c["whatif_usd"] == pytest.approx(actual - c["fixable_usd"])      # expiries are NOT subtracted
    assert c["fixable_pct"] == pytest.approx(round(100 * c["fixable_usd"] / actual, 1))
    # the timestamp miss: everything the previous request had cached (10,000 read + 500 written) was re-written
    p = price_for("claude-sonnet-5-5")
    assert c["fixable"]["system_changed"]["lost_usd"] == pytest.approx(usd(10_500, p["cache_write"] - p["cache_read"]))
    assert [m["cause"] for m in c["top_fixable"]] == sorted(
        (m["cause"] for m in c["top_fixable"]), key=lambda k: -by_cause[k])
    assert any("10:31" in m["detail"] for m in c["top_fixable"])
    assert any("not fixable by changing the prompt" in n for n in c["notes"])


def test_cache_savings_render() -> None:
    c = whatif.cache_savings(cache_session())
    line = whatif.render_cache_text(c)
    assert line.startswith("prompt cache: ~$")
    assert "lost to fixable misses (" in line
    assert "system prompt changed x1 incl. 1 timestamp/ID" in line and "tool order x1" in line
    assert "to expiries (x1, not fixable in the prompt)" in line
    detail = whatif.render_text(c)
    assert len(detail.splitlines()) == 3                                    # line + the two fixable misses
    combined = whatif.render_text(whatif.compare(cache_session(), ["claude-haiku-5-5"]), cache=c,
                                  period="this week")
    assert combined.splitlines()[0].startswith("what if: 6 calls this week, actual $")
    assert combined.splitlines()[-1] == line


def test_cache_savings_other_causes_and_all_hits() -> None:
    r1 = req()
    edited = grow(r1)
    edited["messages"][1]["content"][0]["text"] = "I'll look at the test file first, carefully."
    exs = [call("claude-sonnet-5-5", inp=40, out=10, cw=10_000, ts=T0, body=r1, session="s"),
           call("claude-sonnet-5-5", inp=50, out=10, cr=3_000, cw=7_000, ts=T0 + 10, body=edited, session="s")]
    c = whatif.cache_savings(exs)
    assert c["misses"] == 1 and set(c["other"]) == {"history_edited"}
    assert c["fixable_usd"] == 0 and c["other_usd"] > 0 and c["whatif_usd"] == pytest.approx(c["actual_usd"])
    assert whatif.render_cache_text(c) == (f"prompt cache: no fixable misses; {whatif.fmt_usd(c['other_usd'])} "
                                           f"other (x1: history edited)")
    hits = [call("claude-sonnet-5-5", inp=40, out=10, cw=10_000, ts=T0, body=r1, session="s"),
            call("claude-sonnet-5-5", inp=30, out=10, cr=10_000, cw=500, ts=T0 + 10, body=grow(r1), session="s")]
    c = whatif.cache_savings(hits)
    assert c["pairs"] == 1 and c["misses"] == 0 and c["fixable_usd"] == 0
    assert "no misses (1 follow-up requests)" in whatif.render_cache_text(c)


def test_cache_savings_garbage_bodies() -> None:
    bodies = ["", "not json", '{"messages": "x"', json.dumps({"system": 5, "tools": "y"}), "[]"]
    exs = [call("claude-sonnet-5-5", inp=10, out=1, cw=5_000, ts=T0 + i, body=b, session="s")
           for i, b in enumerate(bodies)]
    c = whatif.cache_savings(exs)
    assert isinstance(c["fixable"], dict) and math.isfinite(c["whatif_usd"])
    whatif.render_text(c)


# ------------------------------------------------------------------ json + store integration


def test_to_json_shapes() -> None:
    exs = mixed_period()
    res = whatif.compare(exs, ["gpt-5", "nope"])
    c = whatif.cache_savings(exs)
    doc = json.loads(whatif.to_json(res, cache=c, period="since today"))
    assert doc["prices_as_of"] == whatif.PRICES_AS_OF and doc["period"] == "since today"
    assert [r["target"] for r in doc["compare"]] == ["gpt-5", "nope"] and "cache" in doc
    doc = json.loads(whatif.to_json(whatif.reprice(exs, "gpt-5")))
    assert doc["reprice"]["ok"] and isinstance(doc["reprice"]["notes"], list)
    assert set(json.loads(whatif.to_json(c))) == {"prices_as_of", "cache"}
    assert json.loads(whatif.to_json({"x": float("nan")}))["reprice"]["x"] is None


def test_reprice_from_store_iter_light(tmp_path) -> None:
    st = Store(str(tmp_path / "t.db"))
    try:
        for ex in mixed_period():
            st.record(ex)
        st.record(call("gpt-5", inp=23_000, out=500, cr=20_000, ts=T0 + 500))
        rows = list(st.iter_light())
    finally:
        st.close()
    r = whatif.reprice(rows, "claude-sonnet-5-5")
    assert r["calls"] == 42 and r["skipped"]["estimated"] == 1
    direct = whatif.reprice(mixed_period() + [call("gpt-5", inp=23_000, out=500, cr=20_000)], "claude-sonnet-5-5")
    assert r["whatif_usd"] == pytest.approx(direct["whatif_usd"])
    assert r["actual_usd"] == pytest.approx(direct["actual_usd"])
    assert [m["model"] for m in r["by_model"]] == ["claude-opus-5-5", "gpt-5", "claude-haiku-5-5"]
