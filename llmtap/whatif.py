"""What-if pricing: what the recorded calls would have cost on another model, and what prompt-cache misses cost.

Pure functions over :class:`~llmtap.models.Exchange` records, plus plain-text / JSON renderers:

* ``reprice(exchanges, target_model)`` re-prices every recorded upstream call's token usage at ``target_model``'s
  list prices and compares it with the recorded cost.
* ``compare(exchanges, models=None)`` does that for several candidates (default: the latest model of each
  main family in the price table that is no pricier than ~2x what was used, plus the models actually used),
  sorted by cost.
* ``cache_savings(exchanges)`` uses :func:`llmtap.xray.cache_report` to price prompt-cache misses whose cause can be
  fixed in the prompt (system prompt / timestamp changed, tool order changed, tool definitions edited), and
  reports TTL expiries and other misses separately.
* ``render_text`` / ``to_json`` present the results.

Token accounting (the same rules as :func:`llmtap.pricing.cost_usd` and :mod:`llmtap.xray`): Anthropic and Bedrock
report ``input_tokens`` EXCLUDING cache reads/writes; OpenAI-style APIs (OpenAI, Gemini, DeepSeek, routers...)
report cached tokens as a SUBSET of ``input_tokens`` and do not bill cache writes. Each recorded call is first
split into ``uncached input + cache reads + cache writes + output`` (prompt size preserved), then expressed in the
target's convention:

* Claude targets (Anthropic accounting): cache reads carry over as cache reads; cache writes carry over only where
  the source reported billable writes (Anthropic/Bedrock calls). Writes are never invented for OpenAI-style calls.
* Other targets (OpenAI-style accounting): cache writes are not billed separately, so they become normal input.
  Cache reads carry over as cache reads when the target has a published cache-read price (or is the model that
  served the call); otherwise they are billed as normal input (no unpublished discount is assumed).
* Output tokens carry over unchanged. Long-context rate cards apply to the target from the preserved prompt size.
  Fast mode, the flex tier, 1-hour cache writes and web searches are not kept in the database; the what-if uses
  standard-tier rates.

Calls that were not upstream calls (cache/cassette replays, blocked, failed requests) are ignored. Calls with no
usage, estimated usage, or a model that is not in the price table are skipped and counted. Nothing here raises on
odd data: an unknown target model gives an error result with suggestions.
"""
from __future__ import annotations

import difflib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from llmtap import pricing
from llmtap.models import Exchange, Usage
from llmtap.pricing import LONG_CONTEXT, PRICES, PRICES_AS_OF, cost_usd, normalize_model, price_for
from llmtap.term import clean, fmt_usd, pad, paint

__all__ = ["reprice", "compare", "cache_savings", "default_candidates", "render_text", "render_reprice_text",
           "render_compare_text", "render_cache_text", "to_json", "FIXABLE_CAUSES"]

# xray cause -> label. Fixable: the prompt prefix changed in a way the caller controls.
FIXABLE_CAUSES: dict[str, str] = {
    "system_changed": "system prompt changed",
    "tool_order_changed": "tool order",
    "tools_changed": "tool definitions edited",
}
EXPIRY_CAUSES = frozenset({"ttl_expired"})
OTHER_LABELS: dict[str, str] = {
    "history_edited": "history edited",
    "history_truncated": "history shortened",
    "model_changed": "model switched",
    "unknown": "unexplained",
}
_VOLATILE_MARK = "looks like a timestamp/ID"

PEER_FACTOR = 2.0          # default candidates: at most this x the priciest used model's input+output rate
MAX_USED_IN_COMPARE = 5    # actually-used models added to the default candidate list
_SKIP_REASONS = ("no_usage", "estimated", "unknown_price")
_SKIP_LABELS = {"no_usage": "no usage", "estimated": "estimated usage", "unknown_price": "model not in price table"}


# --------------------------------------------------------------------------------------------
# safe coercion


def _int(v: Any) -> int:
    try:
        if isinstance(v, bool):
            return int(v)
        f = float(v)
        if not math.isfinite(f):
            return 0
        return max(int(f), 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _float(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _str(v: Any) -> str:
    return v if isinstance(v, str) else ("" if v is None else str(v))


def _usage(ex: Any) -> Usage:
    u = getattr(ex, "usage", None)
    if isinstance(u, Usage):
        return u
    if isinstance(u, dict):
        try:
            return Usage(**{k: v for k, v in u.items() if k in Usage.__dataclass_fields__})
        except Exception:
            return Usage()
    return Usage()


def _r(x: float) -> float:
    return round(_float(x), 10)


def _anthropic_style(style: str, model: str) -> bool:
    """Whether the usage counts input EXCLUSIVE of cache tokens (cost_usd's rule: the wire style, else the name)."""
    if style:
        return style in ("anthropic", "bedrock")
    return normalize_model(model).startswith("claude")


def _safe_price(model: str | None, overrides: dict | None) -> dict[str, float] | None:
    try:
        return price_for(model, overrides)
    except Exception:
        return None


def _safe_cost(model: str, usage: Usage, overrides: dict | None, style: str) -> float:
    try:
        return _float(cost_usd(model, usage, overrides, style=style))
    except Exception:
        return 0.0


# --------------------------------------------------------------------------------------------
# collecting the recorded calls


@dataclass
class _Call:
    model: str
    key: str            # price-table key the recorded model resolves to
    anth_src: bool
    plain: int          # uncached input (includes OpenAI-style cache "writes", which are not billed)
    cr: int             # cache reads
    cw: int             # billable cache writes (Anthropic/Bedrock only)
    out: int
    actual: float       # recorded cost
    baseline: float     # same usage on its own model at today's table, standard tier

    @property
    def prompt(self) -> int:
        return self.plain + self.cr + self.cw


@dataclass
class _Collected:
    calls: list[_Call]
    skipped: dict[str, int]
    skipped_actual: float
    ignored: int
    modifiers: int      # calls whose in-memory usage had fast mode / flex / 1h writes / web searches
    styles: set[str]
    zero_cost: int = 0  # calls recorded at $0 that today's table prices (their actual = today's price)


def _billable(ex: Any) -> bool:
    status = _int(getattr(ex, "status", 0))
    return _str(getattr(ex, "source", "upstream")) == "upstream" and (status == 0 or 200 <= status < 300)


def _collect(exchanges: Iterable[Exchange] | None, overrides: dict | None) -> _Collected:
    col = _Collected([], {k: 0 for k in _SKIP_REASONS}, 0.0, 0, 0, set())
    keys: dict[str, str | None] = {}
    try:
        items = list(exchanges or [])
    except Exception:
        items = []
    for ex in items:
        try:
            if ex is None or not _billable(ex):
                col.ignored += 1
                continue
            u = _usage(ex)
            inp, out = _int(u.input_tokens), _int(u.output_tokens)
            cr, cw = _int(u.cache_read_tokens), _int(u.cache_write_tokens)
            actual = _float(getattr(ex, "cost_usd", 0.0))
            model = _str(getattr(ex, "model", None)).strip()
            reason = None
            if inp + out + cr + cw == 0:
                reason = "no_usage"
            elif u.estimated:
                reason = "estimated"
            else:
                if model and model not in keys:
                    meta = _target_meta(model, overrides)
                    keys[model] = meta["key"] if meta else None
                if not model or keys.get(model) is None:
                    reason = "unknown_price"
            if reason:
                col.skipped[reason] += 1
                col.skipped_actual += actual
                continue
            style = _str(getattr(ex, "style", "")).strip().lower()
            anth = _anthropic_style(style, model)
            col.styles.add("anthropic" if anth else "openai")
            if anth:
                plain, cw_b = inp, cw
            else:
                plain = inp - cr if cr <= inp else inp      # cached > input: the numbers are clearly exclusive
                cw_b = 0                                    # OpenAI-style writes are not billed (they are in input)
            baseline = _safe_cost(model, Usage(input_tokens=inp, output_tokens=out, cache_read_tokens=cr,
                                               cache_write_tokens=cw), overrides, style or ("anthropic" if anth
                                                                                            else "openai"))
            if actual <= 0 < baseline:
                actual = baseline
                col.zero_cost += 1
            if (u.speed == "fast" or u.service_tier == "flex" or _int(u.web_search_requests)
                    or _int(u.cache_write_1h_tokens)):
                col.modifiers += 1
            col.calls.append(_Call(model, str(keys[model]), anth, plain, cr, cw_b, out, actual, baseline))
        except Exception:  # pragma: no cover - defensive
            col.ignored += 1
    return col


# --------------------------------------------------------------------------------------------
# target model metadata


def _norm_overrides(overrides: dict | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            if isinstance(k, str) and isinstance(v, dict):
                try:
                    out[normalize_model(k)] = v
                except Exception:
                    continue
    return out


def _target_meta(target: str, overrides: dict | None) -> dict[str, Any] | None:
    """Prices, matched key, accounting and whether the target publishes a cache-read price; None if unknown."""
    try:
        prices, long_key = pricing._lookup(target, overrides)
    except Exception:
        prices, long_key = _safe_price(target, overrides), None
    if prices is None:
        return None
    name = normalize_model(target)
    key: str | None = None
    explicit_cr = True
    try:
        _, free = pricing._normalize(target)
        builtin = pricing._best_key(name, PRICES, pricing._LOCAL)
        norm_over = _norm_overrides(overrides)
        okey = pricing._best_key(name, norm_over) if norm_over else None
        if okey is not None and (builtin is None or len(okey) >= len(builtin) or free):
            key = okey
            base = builtin if (builtin is not None and okey.startswith(builtin) and not free) else None
            explicit_cr = (norm_over[okey].get("cache_read") is not None
                           or (base is not None and "cache_read" in pricing._RAW.get(base, {})))
        elif free:
            key = name
        else:
            key = builtin
            explicit_cr = builtin is not None and "cache_read" in pricing._RAW.get(builtin, {})
    except Exception:  # pragma: no cover - private helpers changed: fall back to llmtap's default cache prices
        key, explicit_cr = name, True
    anth = name.startswith("claude")
    long = LONG_CONTEXT.get(long_key) if long_key else None
    return {"prices": dict(prices), "key": key or name, "norm": name, "anthropic": anth,
            "cache_read_published": bool(explicit_cr or anth),
            "long_context": {"threshold": long[0], "prices": dict(long[1])} if long else None,
            "free": all(_float(v) == 0 for v in prices.values())}


def _all_keys(overrides: dict | None) -> list[str]:
    return list(dict.fromkeys(list(_norm_overrides(overrides)) + [k for k, v in PRICES.items()
                                                                   if any(_float(x) for x in v.values())]))


def _suggest(target: str, overrides: dict | None, n: int = 5) -> list[str]:
    try:
        name = normalize_model(target) or target.strip().lower()
    except Exception:
        name = _str(target).strip().lower()
    if not name:
        return []
    keys = _all_keys(overrides)
    variants = {name, name.replace(".", "-"), name.replace("-", ".")}
    hits = sorted({k for k in keys for v in variants if v and v in k}, key=lambda k: (len(k), k))
    close = difflib.get_close_matches(name, keys, n=n, cutoff=0.55)
    return list(dict.fromkeys(close + hits))[:n]


# --------------------------------------------------------------------------------------------
# 1. reprice


def _price_call(c: _Call, target: str, meta: dict[str, Any], overrides: dict | None) -> tuple[float, bool]:
    """(what-if cost, whether cache reads were billed as normal input)."""
    same = c.key == meta["key"]
    carry = meta["cache_read_published"] or same
    cr = c.cr if carry else 0
    plain = c.plain + (c.cr - cr)
    if meta["anthropic"]:
        u = Usage(input_tokens=plain, output_tokens=c.out, cache_read_tokens=cr, cache_write_tokens=c.cw)
        style = "anthropic"
    else:
        u = Usage(input_tokens=plain + c.cw + cr, output_tokens=c.out, cache_read_tokens=cr)
        style = "openai"
    return _safe_cost(target, u, overrides, style), (c.cr > 0 and not carry)


def _usd_rate(x: float) -> str:
    return f"${_float(x):g}/M"


def _base_notes(col: _Collected, overrides: dict | None) -> list[str]:
    notes = [f"Prices: list prices per 1M tokens as of {PRICES_AS_OF}, standard tier"
             + (", with pricing_overrides applied" if overrides else "") + "."]
    notes.append("Token accounting: Anthropic/Bedrock report input_tokens excluding cache tokens; OpenAI-style APIs "
                 "(OpenAI, Gemini, DeepSeek, routers) include cached tokens in input_tokens, so cache reads are "
                 "subtracted first. Each call's prompt size is preserved when moving between providers.")
    notes.append("Output tokens are assumed unchanged; another model would write a different amount, and tokenizers "
                 "count the same text differently (often 10-30%).")
    if col.modifiers:
        notes.append(f"{col.modifiers} call(s) used fast mode, the flex tier, 1-hour cache writes or web searches; "
                     f"the what-if prices them at standard rates without those.")
    if col.zero_cost:
        notes.append(f"{col.zero_cost} call(s) were recorded with a $0 cost (model unpriced when recorded); their "
                     f"actual cost is taken from today's price table.")
    sk = sum(col.skipped.values())
    if sk:
        parts = ", ".join(f"{n} {_SKIP_LABELS[k]}" for k, n in col.skipped.items() if n)
        notes.append(f"{sk} call(s) skipped ({parts}); their recorded cost ({fmt_usd(col.skipped_actual)}) is "
                     f"excluded from both totals.")
    return notes


def _error_result(target: Any, col: _Collected, error: str, suggestions: list[str]) -> dict[str, Any]:
    return {"ok": False, "target": _str(target), "error": error, "suggestions": suggestions,
            "calls": len(col.calls), "actual_usd": _r(sum(c.actual for c in col.calls)), "whatif_usd": None,
            "diff_usd": None, "diff_pct": None, "skipped": dict(col.skipped), "skipped_total": sum(col.skipped.values()),
            "ignored": col.ignored, "notes": []}


def _reprice(col: _Collected, target_model: Any, overrides: dict | None) -> dict[str, Any]:
    if not isinstance(target_model, str) or not target_model.strip():
        return _error_result(target_model, col, "no target model given", [])
    target = target_model.strip()
    meta = _target_meta(target, overrides)
    if meta is None:
        sugg = _suggest(target, overrides)
        msg = f"unknown model '{clean(target)}': not in the price table"
        if sugg:
            msg += " (did you mean " + ", ".join(sugg) + "?)"
        return _error_result(target, col, msg, sugg)
    by_model: dict[str, dict[str, Any]] = {}
    tok = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    actual = whatif = baseline = 0.0
    reads_as_input = long_calls = 0
    for c in col.calls:
        cost, as_input = _price_call(c, target, meta, overrides)
        reads_as_input += as_input
        if meta["long_context"] and c.prompt > meta["long_context"]["threshold"]:
            long_calls += 1
        actual += c.actual
        whatif += cost
        baseline += c.baseline
        tok["input"] += c.plain
        tok["cache_read"] += c.cr
        tok["cache_write"] += c.cw
        tok["output"] += c.out
        m = by_model.setdefault(c.model, {"model": c.model, "calls": 0, "actual_usd": 0.0, "whatif_usd": 0.0,
                                          "tokens": {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}})
        m["calls"] += 1
        m["actual_usd"] += c.actual
        m["whatif_usd"] += cost
        for k, v in (("input", c.plain), ("cache_read", c.cr), ("cache_write", c.cw), ("output", c.out)):
            m["tokens"][k] += v
    for m in by_model.values():
        m["actual_usd"], m["whatif_usd"] = _r(m["actual_usd"]), _r(m["whatif_usd"])
        m["diff_usd"] = _r(m["whatif_usd"] - m["actual_usd"])
    diff = whatif - actual
    p = meta["prices"]
    notes = _base_notes(col, overrides)
    has_anth = "anthropic" in col.styles
    has_oai = "openai" in col.styles
    if meta["free"]:
        notes.append(f"{target} is priced at $0 (a local / free model): the what-if is $0.")
    elif meta["anthropic"]:
        s = (f"{target} is priced with Anthropic accounting: cache reads carry over as cache reads at "
             f"{_usd_rate(p['cache_read'])}")
        if has_anth:
            s += f"; cache writes reported by Anthropic/Bedrock calls carry over at {_usd_rate(p['cache_write'])}"
        if has_oai:
            s += ("; OpenAI-style calls report no billable cache writes, so none are added (on Claude the first use of "
                  "each cached prefix would also pay ~1.25x input and needs cache_control breakpoints)")
        notes.append(s + ".")
    else:
        s = f"{target} is priced with OpenAI-style accounting: "
        if meta["cache_read_published"]:
            s += f"cache reads carry over at its cache-read price ({_usd_rate(p['cache_read'])})"
        else:
            s += "it has no published cache-read price, so cache reads are billed as normal input"
        if has_anth:
            s += ("; cache writes are not billed separately, so Anthropic cache-write tokens become normal input "
                  "(assumes its automatic prompt caching hits the same prefixes)")
        notes.append(s + ".")
    if reads_as_input and meta["cache_read_published"] is False:
        notes.append(f"{reads_as_input} call(s) had cache reads that were billed as normal input on {target}.")
    if long_calls:
        notes.append(f"{long_calls} call(s) have prompts above {meta['long_context']['threshold']:,} tokens and are "
                     f"priced at {target}'s long-context rates.")
    if col.calls and abs(baseline - actual) > max(0.0001, 0.01 * abs(actual)):
        notes.append(f"The recorded cost ({fmt_usd(actual)}) differs from the same usage at today's table "
                     f"({fmt_usd(baseline)}): prices changed since recording, or fast mode / flex / 1-hour cache "
                     f"writes / web searches applied (not kept in the database). The difference is against the "
                     f"recorded cost.")
    return {
        "ok": True, "target": target, "priced_as": meta["key"], "accounting": "anthropic" if meta["anthropic"]
        else "openai", "target_prices": p, "cache_reads_carry_over": bool(meta["cache_read_published"]),
        "calls": len(col.calls), "actual_usd": _r(actual), "whatif_usd": _r(whatif), "diff_usd": _r(diff),
        "diff_pct": round(100.0 * diff / actual, 1) if actual > 0 else None, "baseline_usd": _r(baseline),
        "tokens": tok,
        "by_model": sorted(by_model.values(), key=lambda m: (-m["actual_usd"], -m["calls"], m["model"])),
        "skipped": dict(col.skipped), "skipped_total": sum(col.skipped.values()),
        "skipped_actual_usd": _r(col.skipped_actual), "ignored": col.ignored, "notes": notes,
    }


def reprice(exchanges: Iterable[Exchange], target_model: str, overrides: dict | None = None) -> dict[str, Any]:
    """What the recorded upstream calls would have cost on ``target_model``. Never raises.

    Returns ``{"ok": True, "target", "priced_as" (price-table key), "accounting" ("anthropic"|"openai"),
    "target_prices", "cache_reads_carry_over", "calls", "actual_usd" (recorded cost of the included calls),
    "whatif_usd", "diff_usd" (what-if minus actual), "diff_pct" (None when actual is 0), "baseline_usd" (the same
    usage on the recorded models at today's table), "tokens": {"input" (uncached), "cache_read", "cache_write",
    "output"}, "by_model": [{"model", "calls", "actual_usd", "whatif_usd", "diff_usd", "tokens"}], "skipped":
    {"no_usage", "estimated", "unknown_price"}, "skipped_total", "skipped_actual_usd", "ignored" (not upstream /
    failed), "notes": [assumptions]}``.

    An unknown target gives ``{"ok": False, "target", "error", "suggestions": [close price-table keys], ...}``.
    """
    try:
        return _reprice(_collect(exchanges, overrides), target_model, overrides)
    except Exception as e:  # pragma: no cover - defensive
        return _error_result(target_model, _Collected([], {k: 0 for k in _SKIP_REASONS}, 0.0, 0, 0, set()),
                             f"what-if failed: {type(e).__name__}", [])


# --------------------------------------------------------------------------------------------
# 2. compare

# (family regex over price-table keys, version groups) -> the latest key of each family is a candidate
_FAMILIES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^claude-(opus|sonnet|haiku)-(\d+)(?:-(\d+))?$"), "claude"),
    (re.compile(r"^gpt-(\d+)(?:\.(\d+))?(?:-(mini|nano))?$"), "gpt"),
    (re.compile(r"^gemini-(\d+)(?:\.(\d+))?-(pro|flash-lite|flash)$"), "gemini"),
    (re.compile(r"^deepseek-(chat)$"), "deepseek"),
]


def _family_latest() -> list[str]:
    best: dict[tuple[str, str], tuple[tuple[int, ...], str]] = {}
    for key in PRICES:
        for rx, fam in _FAMILIES:
            m = rx.match(key)
            if not m:
                continue
            g = m.groups()
            if fam == "claude":
                tier, ver = g[0], (int(g[1]), int(g[2] or 0))
            elif fam == "gpt":
                tier, ver = g[2] or "", (int(g[0]), int(g[1] or 0))
            elif fam == "gemini":
                tier, ver = g[2], (int(g[0]), int(g[1] or 0))
            else:
                tier, ver = g[0], (0,)
            cur = best.get((fam, tier))
            if cur is None or ver > cur[0]:
                best[(fam, tier)] = (ver, key)
            break
    return [k for _, k in sorted(best.values(), key=lambda vk: vk[1])]


def _rate(prices: dict[str, float] | None) -> float:
    return _float((prices or {}).get("input")) + _float((prices or {}).get("output"))


def default_candidates(exchanges: Iterable[Exchange] | None = None, overrides: dict | None = None) -> list[str]:
    """Latest model of each main family in the price table (Claude Opus/Sonnet/Haiku, GPT base/mini/nano, Gemini
    Pro/Flash/Flash-Lite, DeepSeek) that is at most ``PEER_FACTOR`` x the priciest used model's rate, plus the
    (up to ``MAX_USED_IN_COMPARE``) most-used models."""
    try:
        col = _collect(exchanges, overrides) if exchanges is not None else None
        return _candidates(col, overrides)
    except Exception:  # pragma: no cover - defensive
        return _family_latest()


def _candidates(col: _Collected | None, overrides: dict | None) -> list[str]:
    used: dict[str, int] = {}
    for c in (col.calls if col else []):
        used[c.model] = used.get(c.model, 0) + 1
    used_models = [m for m, _ in sorted(used.items(), key=lambda kv: (-kv[1], kv[0]))][:MAX_USED_IN_COMPARE]
    cap = max((_rate(_safe_price(m, overrides)) for m in used_models), default=0.0)
    fams = _family_latest()
    if cap > 0:
        fams = [k for k in fams if _rate(_safe_price(k, overrides)) <= PEER_FACTOR * cap + 1e-12]
    out: list[str] = []
    seen: set[str] = set()
    for m in used_models + fams:
        meta = _target_meta(m, overrides)
        ident = meta["key"] if meta else normalize_model(m)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(m)
    return out


def compare(exchanges: Iterable[Exchange], models: list[str] | None = None,
            overrides: dict | None = None) -> list[dict[str, Any]]:
    """``reprice`` for several candidate models, cheapest first (unknown models last, as error results). Each
    result also has ``"used"`` (the model served some of the calls) and ``"used_calls"``. Never raises."""
    try:
        col = _collect(exchanges, overrides)
        if models is None:
            names = _candidates(col, overrides)
        else:
            names = list(dict.fromkeys(m.strip() if isinstance(m, str) else m for m in (models or [])))
        used: dict[str, int] = {}
        for c in col.calls:
            used[c.key] = used.get(c.key, 0) + 1
        out: list[dict[str, Any]] = []
        for m in names:
            r = _reprice(col, m, overrides)
            n = 0
            if r.get("ok"):
                n = used.get(r["priced_as"], 0)
            r["used"], r["used_calls"] = n > 0, n
            out.append(r)
        out.sort(key=lambda r: (not r.get("ok"), _float(r.get("whatif_usd")), _str(r.get("target"))))
        return out
    except Exception:  # pragma: no cover - defensive
        return []


# --------------------------------------------------------------------------------------------
# 3. cache what-if


def _actual_cost(ex: Exchange, overrides: dict | None) -> float:
    """Recorded cost; a call recorded at $0 with reported usage is priced at today's table."""
    rec = _float(getattr(ex, "cost_usd", 0.0))
    if rec > 0:
        return rec
    u = _usage(ex)
    if u.estimated:
        return 0.0
    style = _str(getattr(ex, "style", "")).strip().lower()
    model = _str(getattr(ex, "model", None)).strip()
    return _safe_cost(model, Usage(input_tokens=_int(u.input_tokens), output_tokens=_int(u.output_tokens),
                                   cache_read_tokens=_int(u.cache_read_tokens),
                                   cache_write_tokens=_int(u.cache_write_tokens)),
                      overrides, style or ("anthropic" if _anthropic_style("", model) else "openai")) if model else 0.0


def cache_savings(exchanges: Iterable[Exchange], overrides: dict | None = None, top: int = 5) -> dict[str, Any]:
    """What the recorded calls would have cost if every prompt-cache miss with a fixable cause had been a hit.

    Misses come from :func:`llmtap.xray.cache_report`; each miss's ``lost_usd`` (tokens re-written / re-read at full
    price instead of a cache read) is what a hit would have saved. Fixable causes are ``FIXABLE_CAUSES``; TTL
    expiries and other causes (history edits, model switches, unexplained) are reported but not counted.

    Returns ``{"calls", "actual_usd", "whatif_usd" (actual - fixable), "fixable_usd", "fixable_pct",
    "expiry_usd", "other_usd", "pairs", "hits", "misses", "fixable": {cause: {"label", "count", "lost_usd"
    (+ "volatile" for system_changed: misses where a timestamp/ID changed)}}, "expiries": {"count",
    "lost_usd"}, "other": {cause: {"label", "count", "lost_usd"}}, "top_fixable": [{"id", "ts", "model",
    "cause", "detail", "lost_usd"}], "notes"}``. Never raises."""
    out: dict[str, Any] = {"calls": 0, "actual_usd": 0.0, "whatif_usd": 0.0, "fixable_usd": 0.0, "fixable_pct": None,
                           "expiry_usd": 0.0, "other_usd": 0.0, "pairs": 0, "hits": 0, "misses": 0, "fixable": {},
                           "expiries": {"count": 0, "lost_usd": 0.0}, "other": {}, "top_fixable": [], "notes": []}
    try:
        items = [e for e in list(exchanges or []) if isinstance(e, Exchange)]
    except Exception:
        items = []
    billable = [e for e in items if _billable(e)]
    out["calls"] = len(billable)
    actual = sum(_actual_cost(e, overrides) for e in billable)
    try:
        from llmtap import xray
        rep = xray.cache_report(billable, overrides)
    except Exception:  # pragma: no cover - defensive
        rep = {"pairs": 0, "hits": 0, "misses": []}
    out["pairs"], out["hits"] = _int(rep.get("pairs")), _int(rep.get("hits"))
    misses = [m for m in (rep.get("misses") or []) if isinstance(m, dict)]
    out["misses"] = len(misses)
    fixable_list = []
    for m in misses:
        cause = _str(m.get("cause")) or "unknown"
        lost = _float(m.get("lost_usd"))
        if cause in FIXABLE_CAUSES:
            d = out["fixable"].setdefault(cause, {"label": FIXABLE_CAUSES[cause], "count": 0, "lost_usd": 0.0})
            d["count"] += 1
            d["lost_usd"] += lost
            if cause == "system_changed":
                d["volatile"] = d.get("volatile", 0) + (_VOLATILE_MARK in _str(m.get("detail")))
            out["fixable_usd"] += lost
            fixable_list.append(m)
        elif cause in EXPIRY_CAUSES:
            out["expiries"]["count"] += 1
            out["expiries"]["lost_usd"] += lost
            out["expiry_usd"] += lost
        else:
            d = out["other"].setdefault(cause, {"label": OTHER_LABELS.get(cause, cause), "count": 0, "lost_usd": 0.0})
            d["count"] += 1
            d["lost_usd"] += lost
            out["other_usd"] += lost
    for group in (out["fixable"], out["other"]):
        for d in group.values():
            d["lost_usd"] = _r(d["lost_usd"])
    out["fixable"] = dict(sorted(out["fixable"].items(), key=lambda kv: -kv[1]["lost_usd"]))
    out["other"] = dict(sorted(out["other"].items(), key=lambda kv: -kv[1]["lost_usd"]))
    out["expiries"]["lost_usd"] = _r(out["expiries"]["lost_usd"])
    fixable_list.sort(key=lambda m: -_float(m.get("lost_usd")))
    out["top_fixable"] = [{"id": m.get("id"), "ts": m.get("ts"), "model": m.get("model"), "cause": m.get("cause"),
                           "detail": _str(m.get("detail")), "lost_usd": _r(_float(m.get("lost_usd")))}
                          for m in fixable_list[:max(0, _int(top))]]
    out["actual_usd"] = _r(actual)
    out["fixable_usd"] = _r(out["fixable_usd"])
    out["expiry_usd"] = _r(out["expiry_usd"])
    out["other_usd"] = _r(out["other_usd"])
    out["whatif_usd"] = _r(max(actual - out["fixable_usd"], 0.0))
    out["fixable_pct"] = round(100.0 * out["fixable_usd"] / actual, 1) if actual > 0 else None
    out["notes"] = [
        "Each request is paired with its predecessor in the same session (xray.cache_report); a miss's cost is the "
        "tokens re-written or re-read at full price instead of the cache-read price.",
        "Fixable: the prompt prefix changed (system prompt or a timestamp in it, tool order, tool definitions); "
        "the what-if assumes each such miss would have been a full hit.",
        "TTL expiries are not fixable by changing the prompt (only a longer cache TTL or more frequent requests "
        "help); history edits, model switches and unexplained misses are reported as other.",
    ]
    return out


# --------------------------------------------------------------------------------------------
# 4. renderers


def _pct(diff: float | None, actual: float | None) -> str:
    a = _float(actual)
    if diff is None or a <= 0:
        return "n/a"
    p = 100.0 * _float(diff) / a
    if abs(p) < 0.5:
        return "±0%"
    return f"{p:+.0f}%"


def _models_summary(by_model: list[dict[str, Any]], limit: int = 3) -> str:
    ms = sorted(by_model, key=lambda m: (-m["calls"], m["model"]))
    s = ", ".join(f"{clean(m['model'])} x{m['calls']:,}" for m in ms[:limit])
    if len(ms) > limit:
        s += f", +{len(ms) - limit} more"
    return s


def _header(r: dict[str, Any], period: str, color: bool) -> str:
    p = lambda s, *st: paint(s, *st, color=color)  # noqa: E731
    n = _int(r.get("calls"))
    when = f" {clean(period)}" if period else ""
    if not n:
        return p("what if:", "bold") + f" no upstream calls with recorded usage{when}"
    s = (p("what if:", "bold") + f" {n:,} call{'s' if n != 1 else ''}{when}, actual "
         + p(fmt_usd(r.get("actual_usd")), "yellow"))
    if r.get("by_model"):
        s += f" ({_models_summary(r['by_model'])})"
    return s


def _skipped_line(r: dict[str, Any], color: bool) -> str | None:
    sk = r.get("skipped") or {}
    total = sum(_int(v) for v in sk.values())
    if not total:
        return None
    parts = ", ".join(f"{_int(sk[k])} {_SKIP_LABELS.get(k, k)}" for k in _SKIP_REASONS if _int(sk.get(k)))
    return paint(f"  ({total:,} call{'s' if total != 1 else ''} skipped: {parts})", "dim", color=color)


def _diff_hue(diff: float | None) -> str:
    d = _float(diff)
    return "green" if d < 0 else ("red" if d > 0 else "gray")


def render_cache_text(cache: dict[str, Any] | None, color: bool = False, detail: bool = False) -> str:
    """One line: ``prompt cache: ~$1.90 lost to fixable misses (system prompt changed x14, tool order x2); $0.40 to
    expiries``; with ``detail`` the costliest fixable misses follow."""
    p = lambda s, *st: paint(s, *st, color=color)  # noqa: E731
    head = p("prompt cache:", "bold")
    if not isinstance(cache, dict) or not _int(cache.get("pairs")):
        return head + p(" no follow-up requests to compare", "dim")
    if not _int(cache.get("misses")):
        return head + p(f" no misses ({_int(cache['pairs']):,} follow-up requests)", "green")
    bits = []
    fx = cache.get("fixable") or {}
    if fx:
        causes = []
        for cause, d in fx.items():
            lab = f"{d.get('label', cause)} x{_int(d.get('count')):,}"
            if _int(d.get("volatile")):
                lab += f" incl. {_int(d['volatile']):,} timestamp/ID"
            causes.append(lab)
        bits.append(p(f"~{fmt_usd(cache.get('fixable_usd'))}", "yellow") + " lost to fixable misses ("
                    + ", ".join(causes) + ")")
    else:
        bits.append(p("no fixable misses", "green"))
    exp = cache.get("expiries") or {}
    if _int(exp.get("count")):
        bits.append(f"{fmt_usd(exp.get('lost_usd'))} to expiries (x{_int(exp['count']):,}, not fixable in the prompt)")
    if cache.get("other"):
        n = sum(_int(d.get("count")) for d in cache["other"].values())
        bits.append(f"{fmt_usd(cache.get('other_usd'))} other (x{n:,}: "
                    + ", ".join(d.get("label", c) for c, d in cache["other"].items()) + ")")
    lines = [head + " " + "; ".join(bits)]
    if detail:
        for m in cache.get("top_fixable") or []:
            cost = f"{fmt_usd(m.get('lost_usd')):>9}"
            lines.append(f"  {p(cost, 'yellow')}  {clean(m.get('model') or '-')}  {clean(m.get('detail') or '')}")
    return "\n".join(lines)


def _notes_lines(notes: list[str], color: bool, max_notes: int) -> list[str]:
    if max_notes <= 0 or not notes:
        return []
    shown = notes[:max_notes]
    lines = [paint(f"  note: {clean(n)}", "dim", color=color) for n in shown]
    if len(notes) > len(shown):
        lines.append(paint(f"  (+{len(notes) - len(shown)} more notes in --json)", "dim", color=color))
    return lines


def render_reprice_text(result: dict[str, Any], cache: dict[str, Any] | None = None, color: bool = False,
                        period: str = "", max_notes: int = 3) -> str:
    """Header line, the what-if total vs actual, one row per source model, skipped calls, notes."""
    p = lambda s, *st: paint(s, *st, color=color)  # noqa: E731
    if not isinstance(result, dict):
        return p("what if: nothing to show", "dim")
    if not result.get("ok"):
        return p("what if:", "bold") + " " + p(clean(result.get("error") or "unknown error"), "red")
    lines = [_header(result, period, color)]
    if result.get("calls"):
        tgt = clean(result["target"])
        if result.get("priced_as") and normalize_model(result["target"]) != normalize_model(result["priced_as"]):
            tgt += p(f" (priced as {clean(result['priced_as'])})", "dim")
        diff = result.get("diff_usd")
        lines.append(f"  on {tgt}: " + p(fmt_usd(result.get("whatif_usd")), "bold") + "  "
                     + p(_pct(diff, result.get("actual_usd")), _diff_hue(diff))
                     + f"  ({'+' if _float(diff) > 0 else ''}{fmt_usd(diff)})")
        rows = result.get("by_model") or []
        if len(rows) > 1:
            w = max(len(clean(m["model"])) for m in rows)
            for m in rows:
                lines.append(f"    {pad(clean(m['model']), w)}  x{m['calls']:<6,} {fmt_usd(m['actual_usd']):>9} -> "
                             f"{fmt_usd(m['whatif_usd']):>9}  "
                             + p(f"{_pct(m['diff_usd'], m['actual_usd']):>5}", _diff_hue(m["diff_usd"])))
    sk = _skipped_line(result, color)
    if sk:
        lines.append(sk)
    lines += _notes_lines([n for n in result.get("notes") or [] if "skipped (" not in n], color, max_notes)
    if cache is not None:
        lines.append(render_cache_text(cache, color=color))
    return "\n".join(lines)


def render_compare_text(results: list[dict[str, Any]], cache: dict[str, Any] | None = None, color: bool = False,
                        period: str = "", max_notes: int = 0) -> str:
    """The comparison table::

        what if: 412 calls since today, actual $4.10 (claude-opus-5-5 x380, claude-haiku-5-5 x32)
          claude-haiku-5-5   $0.30   -93%
          ...
        prompt cache: ~$1.90 lost to fixable misses (system prompt changed x14, tool order x2); $0.40 to expiries
    """
    p = lambda s, *st: paint(s, *st, color=color)  # noqa: E731
    results = [r for r in (results or []) if isinstance(r, dict)]
    base = next((r for r in results if r.get("ok")), results[0] if results else {"calls": 0})
    lines = [_header(base, period, color)]
    if _int(base.get("calls")):
        names = [clean(r.get("target") or "?") for r in results]
        w = max((len(n) for n in names), default=0)
        for r, name in zip(results, names):
            if not r.get("ok"):
                sugg = r.get("suggestions") or []
                lines.append(f"  {pad(name, w)}  " + p("unknown model" + (f" (did you mean {', '.join(sugg[:3])}?)"
                                                                          if sugg else ""), "dim"))
                continue
            diff = r.get("diff_usd")
            row = (f"  {pad(p(name, 'bold') if r.get('used') else name, w)}  {fmt_usd(r.get('whatif_usd')):>9}  "
                   + p(f"{_pct(diff, r.get('actual_usd')):>5}", _diff_hue(diff)))
            tags = []
            if r.get("used"):
                tags.append(f"used x{_int(r.get('used_calls')):,}")
            if r.get("priced_as") and normalize_model(r["target"]) != normalize_model(r["priced_as"]):
                tags.append(f"priced as {clean(r['priced_as'])}")
            if not r.get("cache_reads_carry_over", True) and _int((r.get("tokens") or {}).get("cache_read")):
                tags.append("no cache discount")
            if tags:
                row += "  " + p("(" + ", ".join(tags) + ")", "dim")
            lines.append(row)
    sk = _skipped_line(base, color)
    if sk:
        lines.append(sk)
    if max_notes:
        notes = list(dict.fromkeys(n for r in results for n in (r.get("notes") or []) if "skipped (" not in n))
        lines += _notes_lines(notes, color, max_notes)
    if cache is not None:
        lines.append(render_cache_text(cache, color=color))
    return "\n".join(lines)


def render_text(result: Any, cache: dict[str, Any] | None = None, color: bool = False, period: str = "",
                max_notes: int | None = None) -> str:
    """Render a ``compare`` list, a ``reprice`` dict or a ``cache_savings`` dict (optionally with ``cache``)."""
    try:
        if isinstance(result, list):
            return render_compare_text(result, cache, color=color, period=period,
                                       max_notes=0 if max_notes is None else max_notes)
        if isinstance(result, dict) and "fixable_usd" in result:
            return render_cache_text(result, color=color, detail=True)
        if isinstance(result, dict):
            return render_reprice_text(result, cache, color=color, period=period,
                                       max_notes=3 if max_notes is None else max_notes)
        if result is None and cache is not None:
            return render_cache_text(cache, color=color, detail=True)
    except Exception:  # pragma: no cover - defensive
        pass
    return paint("what if: nothing to show", "dim", color=color)


def _json_safe(obj: Any, depth: int = 0) -> Any:
    if depth > 32:
        return None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v, depth + 1) for v in obj]
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def to_json(result: Any = None, cache: dict[str, Any] | None = None, period: str | None = None,
            indent: int | None = 2) -> str:
    """JSON text for ``--json``: ``{"prices_as_of", "period", "compare" | "reprice", "cache"}`` (keys only for the
    parts given)."""
    doc: dict[str, Any] = {"prices_as_of": PRICES_AS_OF}
    if period is not None:
        doc["period"] = period
    if isinstance(result, list):
        doc["compare"] = result
    elif isinstance(result, dict) and "fixable_usd" in result:
        doc["cache"] = result
    elif isinstance(result, dict):
        doc["reprice"] = result
    if cache is not None:
        doc["cache"] = cache
    return json.dumps(_json_safe(doc), indent=indent, ensure_ascii=False)
