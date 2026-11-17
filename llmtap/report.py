"""Cost & safety report: aggregation (pure functions) and renderers (HTML / Markdown / JSON).

``build_report(store, since) -> dict`` does all the maths in one streaming pass over the store,
``render_html(data)`` / ``render_markdown(data)`` turn that dict into a shareable artifact. The HTML is a
single self-contained file (inline CSS + SVG, no JS, no network) and every piece of untrusted text
(prompts, models, tags, findings) goes through :func:`html.escape`.

``potential_savings(store, since)`` is the cheap "waste finder" used by ``llmtap stats``.
"""
from __future__ import annotations

import hashlib
import heapq
import html
import json
import re
import time
from datetime import date, timedelta
from typing import Any, Iterable, Iterator

from llmtap import __version__
from llmtap.models import Exchange

__all__ = ["build_report", "render_html", "render_markdown", "render_json", "potential_savings",
           "find_waste", "percentile"]

PAGE = 500                       # exchanges fetched per store page
TOP_N = 10
_BLOCK_SOURCES = ("blocked", "budget", "loop")
_PREVIEW_CHARS = 100
_MAX_DAYS = 400

# --------------------------------------------------------------------------------------------
# small helpers


def percentile(values: Iterable[float], p: float) -> float | None:
    """Linear-interpolated percentile (``p`` in 0..100); ``None`` for no data."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * max(0.0, min(100.0, p)) / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _usd(x: float | None) -> str:
    from llmtap.pricing import format_usd
    return format_usd(float(x or 0.0))


def _tok(n: float | int | None) -> str:
    from llmtap.term import fmt_tokens
    return fmt_tokens(n)


def _ms(x: float | None) -> str:
    from llmtap.term import fmt_ms
    return fmt_ms(x) if x is not None else "-"


def _n(x: float | int | None) -> str:
    return f"{int(x or 0):,}"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%" if x and x < 0.1 else f"{100 * x:.0f}%"


def _one_line(s: Any) -> str:
    return " ".join(str(s or "").split())


def _clip(s: str, n: int) -> str:
    s = _one_line(s)
    return s if len(s) <= n else s[: max(0, n - 1)].rstrip() + "…"


def _content_text(c: Any) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        if isinstance(c.get("text"), str):
            return c["text"]
        t = c.get("type")
        if t in ("tool_use", "function_call"):
            return f"[tool call {c.get('name', '?')}]"
        if t == "tool_result":
            return "[tool result] " + _content_text(c.get("content"))
        if t in ("image", "image_url", "input_image"):
            return "[image]"
        if "content" in c:
            return _content_text(c["content"])
        return ""
    if isinstance(c, list):
        return " ".join(x for x in (_content_text(i) for i in c) if x)
    return str(c)


def prompt_preview(req_body: str, limit: int = _PREVIEW_CHARS) -> str:
    """Best-effort one-line preview of the last user message in a request body."""
    if not req_body:
        return ""
    try:
        req = json.loads(req_body)
    except ValueError:
        return _clip(req_body, limit)
    if not isinstance(req, dict):
        return _clip(req_body, limit)
    texts: list[tuple[str, str]] = []
    if req.get("system"):
        texts.append(("system", _content_text(req["system"])))
    if req.get("instructions"):
        texts.append(("system", _content_text(req["instructions"])))
    msgs = req.get("messages")
    for m in msgs if isinstance(msgs, list) else []:
        if isinstance(m, dict):
            texts.append((str(m.get("role", "")), _content_text(m.get("content"))))
    inp = req.get("input")
    if isinstance(inp, str):
        texts.append(("user", inp))
    elif isinstance(inp, list):
        for m in inp:
            if isinstance(m, dict) and "role" in m:
                texts.append((str(m["role"]), _content_text(m.get("content"))))
            elif isinstance(m, str):
                texts.append(("user", m))
    if isinstance(req.get("prompt"), str):
        texts.append(("user", req["prompt"]))
    for role, text in reversed(texts):
        if role == "user" and text.strip():
            return _clip(text, limit)
    for _role, text in reversed(texts):
        if text.strip():
            return _clip(text, limit)
    return ""


def safe_preview(p: Any) -> str:
    """Findings previews are already masked by the guard; re-mask defensively so a report never leaks a secret."""
    s = _one_line(p)
    if not s:
        return ""
    if "…" in s or "[REDACTED" in s or "*" in s:
        return s[:48]
    if len(s) <= 6:
        return s[:1] + "…"
    return s[:3] + "…" + s[-2:]


def _request_key(ex: Exchange, memo: dict[bytes, str] | None = None) -> str:
    """Same key `--cache` would use (provider + upstream path + canonical body), so 'savings' are real, and so
    rows recorded before keys were stored group with newer rows of the same request.

    ``memo`` caches keys by a digest of the raw request, so identical requests (the common case for
    waste) are canonicalized once instead of JSON-parsed and re-serialized every time."""
    body = (ex.request_body or "").encode("utf-8")
    digest = b""
    if memo is not None:
        h = hashlib.blake2b(digest_size=16)
        for part in (ex.provider or "", "\n", ex.path or "", "\n"):
            h.update(part.encode("utf-8"))
        h.update(body)
        digest = h.digest()
        hit = memo.get(digest)
        if hit is not None:
            return hit
    try:
        from llmtap.cache import cache_key
        from llmtap.providers import upstream_path_for
        key = cache_key(ex.provider, upstream_path_for(ex.provider, ex.path or ""), body)
    except Exception:  # pragma: no cover - cache.py unavailable
        key = f"{ex.provider}\n{ex.path}\n{ex.request_body}"
    if memo is not None:
        memo[digest] = key
    return key


def _iter_exchanges(store: Any, since: float | None) -> Iterator[Exchange]:
    """Stream every exchange since ``since`` (newest first) so big DBs don't blow memory. Reports never
    read response bodies, so a Store skips them (``iter_light``, keyset-paged: rows a running proxy
    records meanwhile are newer than the cursor and never re-read).

    Otherwise pages with OFFSET. A running proxy can record new (newest) rows between pages, shifting every
    OFFSET by one; skip the rows that shift onto the next page, or they would be counted twice (and look
    like duplicate prompts)."""
    if hasattr(store, "iter_light"):
        yield from store.iter_light(since)
        return
    offset = 0
    seen: set[str] = set()
    while True:
        page = store.list(limit=PAGE, offset=offset, since=since)
        if not page:
            return
        for ex in page:
            if ex.id not in seen:
                seen.add(ex.id)
                yield ex
        if len(page) < PAGE:
            return
        offset += PAGE


def _is_ok(ex: Exchange) -> bool:
    return ex.source == "upstream" and 200 <= int(ex.status or 0) < 300


# --------------------------------------------------------------------------------------------
# waste finder


def _waste_from(records: dict[str, list[tuple[float, float, str, str | None, str]]],
                preview_of: Any = None) -> dict[str, Any]:
    """records: key -> [(ts, cost, preview, model, provider)] of successful upstream calls.

    With ``preview_of``, the third field is an exchange id and ``preview_of(id)`` is called only for the
    groups that are returned (previews are costly and most records are never shown)."""
    groups: list[dict[str, Any]] = []
    dup_requests = 0
    dup_groups = 0
    total_savings = 0.0
    for key, calls in records.items():
        if len(calls) < 2:
            continue
        calls = sorted(calls, key=lambda c: c[0])
        wasted = calls[1:]                       # the first call is not waste
        savings = sum(c[1] for c in wasted)
        dup_requests += len(wasted)
        dup_groups += 1
        total_savings += savings
        models = sorted({c[3] for c in calls if c[3]})
        groups.append({
            "key": key[:12],
            "preview": next((c[2] for c in calls if c[2]), ""),
            "model": ", ".join(models) if models else None,
            "provider": calls[0][4],
            "count": len(calls),
            "duplicates": len(wasted),
            "total_cost_usd": sum(c[1] for c in calls),
            "savings_usd": savings,
            "first_ts": calls[0][0],
            "last_ts": calls[-1][0],
        })
    groups.sort(key=lambda g: (-g["savings_usd"], -g["count"], g["key"]))
    if preview_of is not None:
        for g in groups[:TOP_N]:
            g["preview"] = preview_of(g["preview"]) if g["preview"] else ""
    return {
        "duplicate_requests": dup_requests,
        "duplicate_groups": dup_groups,
        "savings_usd": total_savings,
        "groups": groups[:TOP_N],
    }


def find_waste(store: Any, since: float | None = None) -> dict[str, Any]:
    """Waste finder on its own: duplicated upstream prompts and what ``--cache`` would have saved."""
    records: dict[str, list[tuple[float, float, str, str | None, str]]] = {}
    memo: dict[bytes, str] = {}
    if hasattr(store, "waste_groups"):
        return _find_waste_indexed(store, since)
    for ex in _iter_exchanges(store, since):
        if _is_ok(ex) and ex.request_body:
            records.setdefault(ex.request_key or _request_key(ex, memo), []).append(
                (ex.ts, float(ex.cost_usd or 0.0), ex.id, ex.model, ex.provider))
    return _waste_from(records, _preview_lookup(store))


def _find_waste_indexed(store: Any, since: float | None) -> dict[str, Any]:
    """find_waste on a Store with stored request keys: duplicate groups come from an index-only SQL
    aggregate; only rows recorded before keys were stored are read and hashed in Python."""
    groups = store.waste_groups(since)
    legacy = store.iter_legacy_waste(since)
    aggs: dict[str, dict[str, Any]] = {}
    for key, n, total, first_ts, first_cost, last_ts in groups:
        aggs[key] = {"n": int(n), "total": float(total or 0.0), "first_ts": float(first_ts),
                     "first_cost": float(first_cost or 0.0), "last_ts": float(last_ts),
                     "first_id": None, "provider": None, "models": None}
    memo: dict[bytes, str] = {}
    legacy_aggs: dict[str, dict[str, Any]] = {}
    # rows recorded without a key, streamed in pages
    for ex_id, ts, cost, provider, path, model, body in legacy:
        key = _request_key(Exchange(id=ex_id, provider=provider, path=path, request_body=body), memo)
        cost = float(cost or 0.0)
        a = legacy_aggs.get(key)
        if a is None:
            legacy_aggs[key] = {"n": 1, "total": cost, "first_ts": ts, "first_cost": cost, "last_ts": ts,
                                "first_id": ex_id, "provider": provider, "models": {model} if model else set()}
            continue
        a["n"] += 1
        a["total"] += cost
        if model:
            a["models"].add(model)
        if ts < a["first_ts"]:
            a.update(first_ts=ts, first_cost=cost, first_id=ex_id, provider=provider)
        a["last_ts"] = max(a["last_ts"], ts)
    # merge with keyed groups; a legacy row may duplicate a keyed request seen only once
    singles = [k for k in legacy_aggs if k not in aggs]
    if singles and hasattr(store, "rows_for_keys"):
        for key, ts, cost, ex_id, model, provider in store.rows_for_keys(singles, since):
            a = legacy_aggs[key]
            a["n"] += 1
            a["total"] += float(cost or 0.0)
            if model:
                a["models"].add(model)
            if ts < a["first_ts"]:
                a.update(first_ts=ts, first_cost=float(cost or 0.0), first_id=ex_id, provider=provider)
            a["last_ts"] = max(a["last_ts"], ts)
    for key, la in legacy_aggs.items():
        a = aggs.get(key)
        if a is None:
            aggs[key] = la
            continue
        a["n"] += la["n"]
        a["total"] += la["total"]
        a["last_ts"] = max(a["last_ts"], la["last_ts"])
        a["legacy_models"] = la["models"]
        if la["first_ts"] < a["first_ts"]:
            a.update(first_ts=la["first_ts"], first_cost=la["first_cost"], first_id=la["first_id"],
                     provider=la["provider"])

    out: list[dict[str, Any]] = []
    dup_requests = 0
    total_savings = 0.0
    for key, a in aggs.items():
        if a["n"] < 2:
            continue
        savings = a["total"] - a["first_cost"]
        dup_requests += a["n"] - 1
        total_savings += savings
        out.append({"key": key[:12], "_key": key, "_a": a, "count": a["n"], "duplicates": a["n"] - 1,
                    "total_cost_usd": a["total"], "savings_usd": savings,
                    "first_ts": a["first_ts"], "last_ts": a["last_ts"]})
    out.sort(key=lambda g: (-g["savings_usd"], -g["count"], g["key"]))
    top = out[:TOP_N]
    preview_of = _preview_lookup(store)
    for g in top:
        a = g.pop("_a")
        full_key = g.pop("_key")
        models = set(a["models"] or ())
        if a["models"] is None or a["first_id"] is None:      # keyed: look the details up now
            first_id, provider, keyed_models = store.waste_detail(full_key, since)
            models.update(keyed_models)
            models.update(a.get("legacy_models") or ())
            if a["first_id"] is None:
                a["first_id"], a["provider"] = first_id, provider
        g["preview"] = preview_of(a["first_id"]) if a["first_id"] else ""
        g["model"] = ", ".join(sorted(models)) if models else None
        g["provider"] = a["provider"]
    ordered = ["key", "preview", "model", "provider", "count", "duplicates", "total_cost_usd", "savings_usd",
               "first_ts", "last_ts"]
    return {
        "duplicate_requests": dup_requests,
        "duplicate_groups": sum(1 for _ in out),
        "savings_usd": total_savings,
        "groups": [{k: g[k] for k in ordered} for g in top],
    }


def _preview_lookup(store: Any) -> Any:
    def preview_of(ex_id: str) -> str:
        ex = store.get(ex_id) if hasattr(store, "get") else None
        return prompt_preview(ex.request_body) if ex is not None else ""
    return preview_of


def potential_savings(store: Any, since: float | None = None) -> tuple[float, int]:
    """(USD that ``--cache`` would have saved, number of duplicate upstream requests)."""
    w = find_waste(store, since)
    return float(w["savings_usd"]), int(w["duplicate_requests"])


# --------------------------------------------------------------------------------------------
# aggregation


def _new_group() -> dict[str, Any]:
    return {"requests": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "saved_usd": 0.0}


def _add_group(groups: dict[str, dict[str, Any]], key: str, ex: Exchange) -> None:
    g = groups.setdefault(key, _new_group())
    g["requests"] += 1
    g["cost_usd"] += float(ex.cost_usd or 0.0)
    g["input_tokens"] += int(ex.usage.input_tokens or 0)
    g["output_tokens"] += int(ex.usage.output_tokens or 0)
    g["saved_usd"] += float(ex.saved_usd or 0.0)


def _finish_groups(groups: dict[str, dict[str, Any]], by_key: bool = False) -> list[dict[str, Any]]:
    rows = [{"key": k, **v} for k, v in groups.items()]
    if by_key:
        rows.sort(key=lambda r: r["key"])
    else:
        rows.sort(key=lambda r: (-r["cost_usd"], -r["requests"], r["key"]))
    return rows


def _fill_days(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < 2:
        return rows
    try:
        first = date.fromisoformat(rows[0]["key"])
        last = date.fromisoformat(rows[-1]["key"])
    except ValueError:
        return rows
    if (last - first).days + 1 > _MAX_DAYS:
        return rows
    have = {r["key"]: r for r in rows}
    out = []
    d = first
    while d <= last:
        k = d.isoformat()
        out.append(have.get(k) or {"key": k, **_new_group()})
        d += timedelta(days=1)
    return out


def build_report(store: Any, since: float | None = None, now: float | None = None) -> dict[str, Any]:
    """Aggregate everything the report shows into one JSON-serializable dict (exact numbers, no rounding)."""
    now = time.time() if now is None else float(now)
    from llmtap.guard import KINDS, PII_KINDS

    totals = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
              "cost_usd": 0.0, "saved_usd": 0.0, "errors": 0, "attempts": 0, "cache_hits": 0, "blocked": 0}
    by_model: dict[str, dict[str, Any]] = {}
    by_provider: dict[str, dict[str, Any]] = {}
    by_day: dict[str, dict[str, Any]] = {}
    by_source: dict[str, int] = {}
    by_tag: dict[str, dict[str, dict[str, Any]]] = {}
    latencies: list[float] = []
    ttfts: list[float] = []
    waste_records: dict[str, list[tuple[float, float, str, str | None, str]]] = {}
    key_cost: dict[str, list[float]] = {}          # key -> [sum cost, n] over ok upstream calls
    heap: list[tuple[float, float, int, Exchange]] = []
    guard_kinds: dict[str, dict[str, Any]] = {}
    guard_total = 0
    guard_exchanges = 0
    blocked_by: dict[str, int] = {"blocked": 0, "budget": 0, "loop": 0}
    loops: dict[str, dict[str, Any]] = {}
    first_ts: float | None = None
    last_ts: float | None = None
    seq = 0
    key_memo: dict[bytes, str] = {}

    for ex in _iter_exchanges(store, since):
        seq += 1
        totals["requests"] += 1
        totals["input_tokens"] += int(ex.usage.input_tokens or 0)
        totals["output_tokens"] += int(ex.usage.output_tokens or 0)
        totals["cache_read_tokens"] += int(ex.usage.cache_read_tokens or 0)
        totals["cost_usd"] += float(ex.cost_usd or 0.0)
        totals["saved_usd"] += float(ex.saved_usd or 0.0)
        first_ts = ex.ts if first_ts is None else min(first_ts, ex.ts)
        last_ts = ex.ts if last_ts is None else max(last_ts, ex.ts)
        by_source[ex.source] = by_source.get(ex.source, 0) + 1
        blocked = ex.source in _BLOCK_SOURCES
        if blocked:
            totals["blocked"] += 1
            blocked_by[ex.source] += 1
        if ex.source in ("cache", "cassette"):
            totals["cache_hits"] += 1
        if ex.source in ("upstream", "error"):
            totals["attempts"] += 1
        if (int(ex.status or 0) >= 400 or ex.source == "error") and not blocked:
            totals["errors"] += 1

        _add_group(by_model, ex.model or "unknown", ex)
        _add_group(by_provider, ex.provider or "unknown", ex)
        _add_group(by_day, time.strftime("%Y-%m-%d", time.localtime(ex.ts)), ex)
        for tname, tval in (ex.tags or {}).items():
            _add_group(by_tag.setdefault(str(tname), {}), str(tval), ex)

        if ex.source == "upstream" and int(ex.status or 0) < 400:
            if ex.latency_ms:
                latencies.append(float(ex.latency_ms))
            if ex.ttft_ms is not None:
                ttfts.append(float(ex.ttft_ms))

        if float(ex.cost_usd or 0.0) > 0:
            item = (float(ex.cost_usd), ex.ts, seq, ex)
            if len(heap) < TOP_N:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

        key = None
        if ex.request_body and (_is_ok(ex) or ex.source == "loop"):
            key = ex.request_key or _request_key(ex, key_memo)
        if key is not None and _is_ok(ex):
            waste_records.setdefault(key, []).append(
                (ex.ts, float(ex.cost_usd or 0.0), ex.id, ex.model, ex.provider))
            kc = key_cost.setdefault(key, [0.0, 0.0])
            kc[0] += float(ex.cost_usd or 0.0)
            kc[1] += 1
        if key is not None and ex.source == "loop":
            lp = loops.setdefault(key, {"preview": prompt_preview(ex.request_body), "model": ex.model,
                                        "blocked": 0, "first_ts": ex.ts, "last_ts": ex.ts})
            lp["blocked"] += 1
            lp["first_ts"] = min(lp["first_ts"], ex.ts)
            lp["last_ts"] = max(lp["last_ts"], ex.ts)
        elif ex.source == "loop":
            lp = loops.setdefault("?" + ex.id, {"preview": prompt_preview(ex.request_body), "model": ex.model,
                                                "blocked": 0, "first_ts": ex.ts, "last_ts": ex.ts})
            lp["blocked"] += 1

        if ex.guard_findings:
            guard_exchanges += 1
            for f in ex.guard_findings:
                if not isinstance(f, dict):
                    continue
                guard_total += 1
                kind = str(f.get("kind") or "unknown")
                gk = guard_kinds.setdefault(kind, {"kind": kind, "count": 0, "exchanges": 0, "previews": [],
                                                   "_ex": set()})
                gk["count"] += 1
                if ex.id not in gk["_ex"]:
                    gk["_ex"].add(ex.id)
                    gk["exchanges"] += 1
                pv = safe_preview(f.get("preview"))
                if pv and pv not in gk["previews"] and len(gk["previews"]) < 3:
                    gk["previews"].append(pv)

    # ---- derived
    waste = _waste_from(waste_records, _preview_lookup(store))

    loop_rows = []
    prevented = 0.0
    for key, lp in loops.items():
        kc = key_cost.get(key)
        per_call = (kc[0] / kc[1]) if kc and kc[1] else 0.0
        est = per_call * lp["blocked"]
        prevented += est
        loop_rows.append({**lp, "est_prevented_usd": est})
    loop_rows.sort(key=lambda r: (-r["blocked"], -r["est_prevented_usd"], r["preview"]))

    kinds_out = []
    for kind, gk in guard_kinds.items():
        kinds_out.append({"kind": kind, "count": gk["count"], "exchanges": gk["exchanges"],
                          "previews": gk["previews"], "description": KINDS.get(kind, ""),
                          "pii": kind in PII_KINDS})
    kinds_out.sort(key=lambda r: (-r["count"], r["kind"]))

    expensive = []
    for cost, _ts, _seq, ex in sorted(heap, reverse=True):
        expensive.append({
            "id": ex.id, "ts": ex.ts, "model": ex.model, "provider": ex.provider, "cost_usd": cost,
            "input_tokens": int(ex.usage.input_tokens or 0), "output_tokens": int(ex.usage.output_tokens or 0),
            "latency_ms": float(ex.latency_ms or 0.0), "source": ex.source, "status": int(ex.status or 0),
            "preview": prompt_preview(ex.request_body),
        })

    tag_out: dict[str, list[dict[str, Any]]] = {}
    for tname in sorted(by_tag, key=lambda t: (t != "feature", t)):
        tag_out[tname] = _finish_groups(by_tag[tname])

    attempts = totals["attempts"]
    total_tokens = totals["input_tokens"] + totals["output_tokens"]
    spend = totals["cost_usd"]
    waste_pct = (waste["savings_usd"] / spend) if spend > 0 else 0.0

    insights: list[str] = []
    if waste["savings_usd"] > 0:
        insights.append(
            f"{_usd(waste['savings_usd'])} ({_pct(waste_pct)} of spend) went to {waste['duplicate_requests']} duplicate "
            f"request{'s' if waste['duplicate_requests'] != 1 else ''}; `llmtap serve --cache` would have served them free.")
    if totals["saved_usd"] > 0:
        insights.append(f"Cache/replay already saved {_usd(totals['saved_usd'])} across {totals['cache_hits']} request(s).")
    if guard_total:
        insights.append(f"{guard_total} secret/PII finding{'s' if guard_total != 1 else ''} in prompts "
                        f"({len(kinds_out)} kind{'s' if len(kinds_out) != 1 else ''}).")
    if blocked_by["loop"]:
        insights.append(f"{blocked_by['loop']} runaway-loop request{'s' if blocked_by['loop'] != 1 else ''} blocked"
                        + (f" (~{_usd(prevented)} prevented)." if prevented > 0 else "."))

    return {
        "version": __version__,
        "generated_at": now,
        "period": {"since": since, "start": since if since is not None else first_ts, "end": now,
                   "first_request": first_ts, "last_request": last_ts},
        "totals": {
            "requests": totals["requests"],
            "upstream_requests": by_source.get("upstream", 0),
            "input_tokens": totals["input_tokens"],
            "output_tokens": totals["output_tokens"],
            "total_tokens": total_tokens,
            "cache_read_tokens": totals["cache_read_tokens"],
            "cost_usd": spend,
            "saved_usd": totals["saved_usd"],
            "cache_hits": totals["cache_hits"],
            "blocked": totals["blocked"],
            "errors": totals["errors"],
            "error_rate": (totals["errors"] / attempts) if attempts else 0.0,
            "guard_findings": guard_total,
            "guard_exchanges": guard_exchanges,
        },
        "latency": {
            "samples": len(latencies),
            "p50_ms": percentile(latencies, 50), "p95_ms": percentile(latencies, 95),
            "ttft_samples": len(ttfts),
            "ttft_p50_ms": percentile(ttfts, 50), "ttft_p95_ms": percentile(ttfts, 95),
        },
        "by_model": _finish_groups(by_model),
        "by_provider": _finish_groups(by_provider),
        "by_day": _fill_days(_finish_groups(by_day, by_key=True)),
        "by_tag": tag_out,
        "by_source": by_source,
        "waste": {**waste, "waste_pct": waste_pct},
        "expensive": expensive,
        "guard": {"total": guard_total, "exchanges": guard_exchanges, "blocked_requests": blocked_by["blocked"],
                  "kinds": kinds_out},
        "loops": {"blocked": blocked_by["loop"], "budget_blocked": blocked_by["budget"],
                  "est_prevented_usd": prevented, "groups": loop_rows[:TOP_N]},
        "insights": insights,
    }


# --------------------------------------------------------------------------------------------
# rendering helpers

def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _fmt_day(ts: float | None) -> str:
    try:
        return time.strftime("%b %d, %Y", time.localtime(ts)) if ts else "-"
    except (OverflowError, OSError, ValueError):  # outside the platform's time_t range
        return "-"


def _period_label(data: dict[str, Any]) -> str:
    p = data.get("period") or {}
    start, end = p.get("start"), p.get("end")
    if not data["totals"]["requests"]:
        return "no requests in this period"
    if start is None:
        return _fmt_day(end)
    a, b = _fmt_day(start), _fmt_day(end)
    return a if a == b else f"{a} → {b}"


def _gen_label(data: dict[str, Any]) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(data.get("generated_at") or time.time()))


_PALETTE = ["#8b8dff", "#22d3ee", "#34d399", "#c084fc", "#fbbf24", "#fb923c", "#fb7185", "#60a5fa"]

_CSS = """
:root{--bg:#08090c;--panel:#101219;--panel2:#151821;--panel3:#1b1f2a;--border:#1f2330;--border2:#2a2f3e;
--text:#e7e9ef;--text2:#b4b9c6;--muted:#7d8496;--dim:#535a6b;--accent:#8b8dff;--accent2:#22d3ee;
--good:#34d399;--warn:#fbbf24;--bad:#f87171;--rose:#fb7185;
--mono:ui-monospace,"SF Mono","JetBrains Mono","Cascadia Code",Menlo,Consolas,monospace;
--sans:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
*{box-sizing:border-box}html,body{margin:0;padding:0}
body{background:radial-gradient(1000px 420px at 10% -8%,rgba(139,141,255,.13),transparent 60%),
radial-gradient(800px 380px at 96% -10%,rgba(34,211,238,.08),transparent 60%),var(--bg);
color:var(--text);font:14px/1.5 var(--sans);-webkit-font-smoothing:antialiased;min-height:100vh}
.wrap{max-width:1120px;margin:0 auto;padding:30px 24px 48px}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:22px}
.logo{width:38px;height:38px;flex:none}
.brand{font-size:21px;font-weight:700;letter-spacing:-.02em}
.brand .tap{background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--muted);font-size:13px}
.spacer{flex:1}
.chip{display:inline-flex;align-items:center;gap:6px;height:26px;padding:0 11px;border-radius:999px;border:1px solid var(--border);background:var(--panel);font-size:12px;color:var(--text2);white-space:nowrap}
.hero{position:relative;overflow:hidden;border:1px solid rgba(52,211,153,.28);border-radius:16px;padding:24px 28px;margin-bottom:18px;
background:linear-gradient(135deg,rgba(52,211,153,.12),rgba(34,211,238,.06) 55%,rgba(139,141,255,.08)),var(--panel)}
.hero.neutral{border-color:var(--border2);background:linear-gradient(135deg,rgba(139,141,255,.10),rgba(34,211,238,.05)),var(--panel)}
.hero .kicker{font-size:11.5px;text-transform:uppercase;letter-spacing:.1em;color:#a7f3d0;font-weight:650}
.hero.neutral .kicker{color:var(--muted)}
.hero .big{font-size:46px;line-height:1.1;font-weight:760;letter-spacing:-.03em;margin:6px 0 4px;
background:linear-gradient(90deg,#6ee7b7,#22d3ee);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero.neutral .big{background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text}
.hero p{margin:0;color:var(--text2);max-width:760px}
.hero code{font-family:var(--mono);font-size:12.5px;background:rgba(255,255,255,.07);padding:1px 7px;border-radius:6px;color:var(--text)}
.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:18px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:14px 16px;min-width:0}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:600}
.card .v{font-size:27px;font-weight:700;letter-spacing:-.02em;margin-top:3px;font-variant-numeric:tabular-nums;white-space:nowrap}
.card .d{font-size:12px;color:var(--muted);margin-top:1px}
.card.good .v{color:var(--good)}.card.bad .v{color:var(--bad)}.card.warn .v{color:var(--warn)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:18px 20px;margin-bottom:14px;min-width:0}
.grid .panel{margin-bottom:0}
.tw{overflow-x:auto}
h2{font-size:12px;font-weight:650;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin:0 0 14px;display:flex;align-items:center;gap:8px}
h2 .n{color:var(--dim);font-weight:500;letter-spacing:0;text-transform:none}
h2 .tag{margin-left:auto;font-size:11px;text-transform:none;letter-spacing:0;font-weight:600;padding:2px 9px;border-radius:999px;border:1px solid var(--border2);color:var(--text2)}
h2 .tag.good{color:#a7f3d0;border-color:rgba(52,211,153,.3);background:rgba(52,211,153,.07)}
h2 .tag.bad{color:#fecaca;border-color:rgba(248,113,113,.3);background:rgba(248,113,113,.07)}
.bars{display:flex;flex-direction:column;gap:11px}
.bar .row{display:flex;justify-content:space-between;gap:12px;font-size:13px;margin-bottom:4px}
.bar .nm{font-family:var(--mono);font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.bar .val{font-variant-numeric:tabular-nums;white-space:nowrap;font-weight:600}
.bar .val small{color:var(--muted);font-weight:500;margin-left:6px}
.track{height:8px;border-radius:99px;background:var(--panel3);overflow:hidden}
.fill{height:100%;border-radius:99px}
table{width:100%;min-width:520px;border-collapse:collapse;font-size:13px}
th{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:600;text-align:left;padding:0 10px 8px 0;border-bottom:1px solid var(--border)}
td{padding:9px 10px 9px 0;border-bottom:1px solid var(--border);vertical-align:top}
tr:last-child td{border-bottom:0}
th.r,td.r{text-align:right}td.r{font-variant-numeric:tabular-nums;white-space:nowrap}
td.p{color:var(--text2);max-width:420px;word-break:break-word}
.mono{font-family:var(--mono);font-size:12.5px}
.save{color:var(--good);font-weight:700}
.cnt{display:inline-block;min-width:34px;text-align:center;padding:1px 8px;border-radius:999px;background:rgba(251,191,36,.12);color:#fde68a;font-weight:650;font-size:12px}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:650;border:1px solid var(--border2);color:var(--text2)}
.badge.bad{color:#fecaca;border-color:rgba(248,113,113,.35);background:rgba(248,113,113,.08)}
.badge.warn{color:#fde68a;border-color:rgba(251,191,36,.35);background:rgba(251,191,36,.08)}
code.pv{font-family:var(--mono);font-size:12px;padding:1px 7px;border-radius:6px;background:var(--panel3);color:var(--text2);margin-right:6px;display:inline-block;margin-bottom:2px}
.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:12px 18px}
.stat .k{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:600}
.stat .v{font-size:21px;font-weight:680;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.stat .d{font-size:11.5px;color:var(--dim)}
.empty{color:var(--muted);padding:6px 0;font-size:13px}
.ok{color:var(--good)}
svg.chart{width:100%;height:auto;display:block}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:12px;font-size:12px;color:var(--text2)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:6px;vertical-align:baseline}
.mix{display:flex;height:10px;border-radius:99px;overflow:hidden;background:var(--panel3);margin:6px 0 2px}
footer{margin-top:26px;color:var(--dim);font-size:12px;text-align:center}
footer b{color:var(--text2);font-weight:600}
@media (max-width:860px){.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.hero .big{font-size:36px}.wrap{padding:20px 16px 36px}}
@media print{body{background:#fff;color:#111}}
"""

_LOGO = ('<svg class="logo" viewBox="0 0 32 32" aria-hidden="true"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">'
         '<stop offset="0" stop-color="#8b8dff"/><stop offset="1" stop-color="#22d3ee"/></linearGradient></defs>'
         '<rect width="32" height="32" rx="8" fill="url(#lg)"/><path d="M5 17h5l3-7 4 13 3-6h7" fill="none" '
         'stroke="white" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _bars(rows: list[dict[str, Any]], limit: int = 8, metric: str = "cost_usd") -> str:
    if not rows:
        return '<div class="empty">No data.</div>'
    rows = rows[:limit]
    mx = max((r[metric] for r in rows), default=0) or 0
    total = sum(r["cost_usd"] for r in rows) or 0
    out = ['<div class="bars">']
    for i, r in enumerate(rows):
        frac = (r[metric] / mx) if mx > 0 else 0
        width = max(1.5, 100 * frac) if r[metric] > 0 else 0
        share = (f" · {max(1, round(100 * r['cost_usd'] / total))}%" if total > 0 and r["cost_usd"] > 0 else "")
        color = _PALETTE[i % len(_PALETTE)]
        out.append(
            f'<div class="bar"><div class="row"><span class="nm" title="{_e(r["key"])}">{_e(r["key"])}</span>'
            f'<span class="val">{_e(_usd(r["cost_usd"]))}<small>{_e(_n(r["requests"]))} req · '
            f'{_e(_tok(r["input_tokens"] + r["output_tokens"]))} tok{_e(share)}</small></span></div>'
            f'<div class="track"><div class="fill" style="width:{width:.1f}%;background:{color}"></div></div></div>')
    out.append("</div>")
    return "".join(out)


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    import math
    exp = math.floor(math.log10(v))
    f = v / (10 ** exp)
    for step in (1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10):
        if f <= step:
            return step * (10 ** exp)
    return 10 ** (exp + 1)


def _day_chart(days: list[dict[str, Any]]) -> str:
    if not days or all(d["cost_usd"] <= 0 for d in days):
        if days and any(d["requests"] for d in days):
            return '<div class="empty">No paid spend recorded in this period (local or free models?).</div>'
        return '<div class="empty">No data.</div>'
    W, H, L, R, T, B = 560, 230, 52, 10, 14, 30
    pw, ph = W - L - R, H - T - B
    n = len(days)
    ymax = _nice_max(max(d["cost_usd"] for d in days))
    slot = pw / n
    bw = max(3.0, min(54.0, slot * 0.68))
    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Cost by day">'
             '<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8b8dff"/>'
             '<stop offset="1" stop-color="#22d3ee" stop-opacity=".75"/></linearGradient></defs>']
    for frac in (0, 0.5, 1):
        y = T + ph * (1 - frac)
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#1f2330" stroke-width="1"/>')
        parts.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10.5" fill="#7d8496" '
                     f'font-family="ui-monospace,monospace">{_e(_usd(ymax * frac) if frac else "$0")}</text>')
    step = max(1, -(-n // 7))
    for i, d in enumerate(days):
        x = L + slot * i + (slot - bw) / 2
        h = ph * (d["cost_usd"] / ymax) if ymax else 0
        y = T + ph - h
        tip = f'{d["key"]}: {_usd(d["cost_usd"])} · {d["requests"]} requests'
        if d["cost_usd"] > 0:
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h, 1.5):.1f}" rx="{min(5, bw / 3):.1f}" '
                         f'fill="url(#bg)"><title>{_e(tip)}</title></rect>')
        else:
            parts.append(f'<rect x="{x:.1f}" y="{T + ph - 1.5:.1f}" width="{bw:.1f}" height="1.5" fill="#2a2f3e">'
                         f'<title>{_e(tip)}</title></rect>')
        if i % step == 0 or i == n - 1 and n <= 7:
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{H - 10}" text-anchor="middle" font-size="10.5" fill="#7d8496" '
                         f'font-family="ui-monospace,monospace">{_e(d["key"][5:] if len(d["key"]) >= 10 else d["key"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


_SOURCE_COLORS = {"upstream": "#8b8dff", "cache": "#34d399", "cassette": "#c084fc", "blocked": "#f87171",
                  "loop": "#fb923c", "budget": "#fbbf24", "error": "#fb7185"}


def _source_mix(by_source: dict[str, int]) -> str:
    total = sum(by_source.values())
    if not total:
        return ""
    segs = []
    legend = []
    for name in sorted(by_source, key=lambda k: -by_source[k]):
        c = _SOURCE_COLORS.get(name, "#7d8496")
        pct = 100 * by_source[name] / total
        segs.append(f'<div style="width:{pct:.2f}%;background:{c}" title="{_e(name)}: {by_source[name]}"></div>')
        legend.append(f'<span><i style="background:{c}"></i>{_e(name)} {_e(_n(by_source[name]))}</span>')
    return f'<div class="mix">{"".join(segs)}</div><div class="legend">{"".join(legend)}</div>'


def _stat(k: str, v: str, d: str = "") -> str:
    return (f'<div class="stat"><div class="k">{_e(k)}</div><div class="v">{_e(v)}</div>'
            + (f'<div class="d">{_e(d)}</div>' if d else "") + "</div>")


def render_html(data: dict[str, Any]) -> str:
    """Render the report dict as one self-contained, XSS-safe HTML document (inline CSS/SVG, no JS)."""
    t = data["totals"]
    w = data["waste"]
    g = data["guard"]
    lp = data["loops"]
    lat = data["latency"]
    out: list[str] = []
    add = out.append

    add('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="dark"><meta name="generator" content="llmtap">'
        f'<title>llmtap report · {_e(_period_label(data))}</title><style>{_CSS}</style></head><body><div class="wrap">')

    # header
    add(f'<header>{_LOGO}<div><div class="brand">llm<span class="tap">tap</span> '
        f'<span style="font-weight:500;color:var(--text2)">cost &amp; safety report</span></div>'
        f'<div class="sub">{_e(_period_label(data))}</div></div><div class="spacer"></div>'
        f'<span class="chip">{_e(_n(t["requests"]))} requests</span>'
        f'<span class="chip">generated {_e(_gen_label(data))}</span></header>')

    # hero
    if w["savings_usd"] > 0:
        add('<section class="hero"><div class="kicker">Waste finder</div>'
            f'<div class="big">{_e(_usd(w["savings_usd"]))} wasted</div>'
            f'<p>{_e(_n(w["duplicate_requests"]))} duplicate request{"s" if w["duplicate_requests"] != 1 else ""} across '
            f'{_e(_n(w["duplicate_groups"]))} repeated prompt{"s" if w["duplicate_groups"] != 1 else ""}'
            + (f' — {_e(_pct(w["waste_pct"]))} of your total spend' if w["waste_pct"] > 0 else "")
            + '. Run the proxy with <code>llmtap serve --cache</code> and these come back instantly, for free.</p></section>')
    elif t["saved_usd"] > 0:
        add('<section class="hero"><div class="kicker">Savings</div>'
            f'<div class="big">{_e(_usd(t["saved_usd"]))} saved</div>'
            f'<p>Served from cache/replay across {_e(_n(t["cache_hits"]))} request(s) — nothing left to deduplicate.</p></section>')
    else:
        add('<section class="hero neutral"><div class="kicker">Waste finder</div><div class="big">No waste found</div>'
            '<p>No duplicated upstream prompts in this period. Nice and lean.</p></section>')

    # cards
    gcls = "bad" if g["total"] else "good"
    add('<div class="cards">'
        f'<div class="card"><div class="k">Spend</div><div class="v">{_e(_usd(t["cost_usd"]))}</div>'
        f'<div class="d">{_e(_n(t["upstream_requests"]))} paid requests</div></div>'
        f'<div class="card"><div class="k">Requests</div><div class="v">{_e(_n(t["requests"]))}</div>'
        f'<div class="d">{_e(_pct(t["error_rate"]))} error rate</div></div>'
        f'<div class="card"><div class="k">Tokens</div><div class="v">{_e(_tok(t["total_tokens"]))}</div>'
        f'<div class="d">{_e(_tok(t["input_tokens"]))} in · {_e(_tok(t["output_tokens"]))} out</div></div>'
        f'<div class="card good"><div class="k">Saved</div><div class="v">{_e(_usd(t["saved_usd"]))}</div>'
        f'<div class="d">{_e(_n(t["cache_hits"]))} from cache/replay</div></div>'
        f'<div class="card {gcls}"><div class="k">Guard findings</div><div class="v">{_e(_n(g["total"]))}</div>'
        f'<div class="d">{_e(_n(t["blocked"]))} requests blocked</div></div></div>')

    # model + day
    add('<div class="grid"><section class="panel"><h2>Cost by model</h2>' + _bars(data["by_model"]) + "</section>"
        '<section class="panel"><h2>Cost by day</h2>' + _day_chart(data["by_day"]) + "</section></div>")

    # provider / reliability / tags
    lat_html = ('<div class="stats">'
                + _stat("Latency p50", _ms(lat["p50_ms"]), f'{_n(lat["samples"])} upstream calls')
                + _stat("Latency p95", _ms(lat["p95_ms"]))
                + _stat("TTFT p50", _ms(lat["ttft_p50_ms"]), f'{_n(lat["ttft_samples"])} streamed' if lat["ttft_samples"] else "no streaming data")
                + _stat("Error rate", _pct(t["error_rate"]), f'{_n(t["errors"])} failed')
                + "</div>" + _source_mix(data["by_source"]))
    add('<div class="grid"><section class="panel"><h2>Latency &amp; reliability</h2>' + lat_html + "</section>"
        '<section class="panel"><h2>Cost by provider</h2>' + _bars(data["by_provider"]) + "</section></div>")

    for tname, rows in list(data["by_tag"].items())[:3]:
        add(f'<section class="panel"><h2>Cost by tag <span class="n">{_e(tname)}</span></h2>' + _bars(rows, limit=10) + "</section>")

    # waste finder
    tag = (f'<span class="tag good">would save {_e(_usd(w["savings_usd"]))}</span>' if w["savings_usd"] > 0 else "")
    add(f'<section class="panel"><h2>Waste finder <span class="n">identical prompts sent upstream more than once</span>{tag}</h2>')
    if w["groups"]:
        add('<div class="tw"><table><thead><tr><th>Prompt</th><th>Model</th><th class="r">Calls</th><th class="r">Total cost</th>'
            '<th class="r">Saved with <span class="mono">--cache</span></th></tr></thead><tbody>')
        for r in w["groups"]:
            add(f'<tr><td class="p">{_e(r["preview"] or "(no text)")}</td><td class="mono">{_e(r["model"] or "-")}</td>'
                f'<td class="r"><span class="cnt">×{_e(r["count"])}</span></td><td class="r">{_e(_usd(r["total_cost_usd"]))}</td>'
                f'<td class="r save">{_e(_usd(r["savings_usd"]))}</td></tr>')
        add("</tbody></table></div>")
    else:
        add('<div class="empty"><span class="ok">✓</span> No prompt was sent upstream twice. The first call of a prompt is never counted as waste.</div>')
    add("</section>")

    # most expensive
    add('<section class="panel"><h2>Most expensive requests <span class="n">top 10</span></h2>')
    if data["expensive"]:
        add('<div class="tw"><table><thead><tr><th>Prompt</th><th>Model</th><th class="r">Tokens</th><th class="r">Latency</th>'
            '<th class="r">Cost</th></tr></thead><tbody>')
        for r in data["expensive"]:
            add(f'<tr><td class="p">{_e(r["preview"] or "(no text)")}</td><td class="mono">{_e(r["model"] or "-")}</td>'
                f'<td class="r">{_e(_tok(r["input_tokens"]))} → {_e(_tok(r["output_tokens"]))}</td>'
                f'<td class="r">{_e(_ms(r["latency_ms"] or None))}</td><td class="r"><b>{_e(_usd(r["cost_usd"]))}</b></td></tr>')
        add("</tbody></table></div>")
    else:
        add('<div class="empty">No priced requests yet.</div>')
    add("</section>")

    # guard
    tag = (f'<span class="tag bad">{_e(_n(g["total"]))} finding{"s" if g["total"] != 1 else ""}</span>' if g["total"]
           else '<span class="tag good">clean</span>')
    add(f'<section class="panel"><h2>Guard findings <span class="n">secrets &amp; PII in prompts · previews are masked</span>{tag}</h2>')
    if g["kinds"]:
        add('<div class="tw"><table><thead><tr><th>Kind</th><th>Severity</th><th class="r">Count</th><th>Masked preview</th></tr></thead><tbody>')
        for k in g["kinds"]:
            sev = '<span class="badge warn">PII</span>' if k["pii"] else '<span class="badge bad">secret</span>'
            pv = "".join(f'<code class="pv">{_e(p)}</code>' for p in k["previews"]) or "-"
            desc = f'<div class="sub">{_e(k["description"])}</div>' if k["description"] else ""
            add(f'<tr><td><span class="mono">{_e(k["kind"])}</span>{desc}</td><td>{sev}</td>'
                f'<td class="r"><b>{_e(k["count"])}</b></td><td>{pv}</td></tr>')
        add("</tbody></table></div>")
    else:
        add('<div class="empty"><span class="ok">✓</span> No secrets or PII found in any prompt.</div>')
    add("</section>")

    # loops
    tag = (f'<span class="tag good">{_e(_n(lp["blocked"]))} blocked'
           + (f' · ~{_e(_usd(lp["est_prevented_usd"]))} prevented' if lp["est_prevented_usd"] > 0 else "") + "</span>"
           if lp["blocked"] else "")
    add(f'<section class="panel"><h2>Runaway loops blocked <span class="n">identical request repeated in a tight window</span>{tag}</h2>')
    if lp["groups"]:
        add('<div class="tw"><table><thead><tr><th>Looping prompt</th><th>Model</th><th class="r">Blocked</th><th class="r">Est. prevented</th></tr></thead><tbody>')
        for r in lp["groups"]:
            est = _usd(r["est_prevented_usd"]) if r["est_prevented_usd"] > 0 else "-"
            add(f'<tr><td class="p">{_e(r["preview"] or "(no text)")}</td><td class="mono">{_e(r["model"] or "-")}</td>'
                f'<td class="r"><b>{_e(r["blocked"])}</b></td><td class="r save">{_e(est)}</td></tr>')
        add("</tbody></table></div>")
    else:
        add('<div class="empty"><span class="ok">✓</span> No runaway loops were detected.'
            + (f' ({_e(_n(lp["budget_blocked"]))} request(s) stopped by a budget cap.)' if lp["budget_blocked"] else "") + "</div>")
    add("</section>")

    add(f'<footer>Generated by <b>llmtap v{_e(data.get("version", __version__))}</b> · {_e(_gen_label(data))} · '
        'self-contained report: no scripts, no network requests</footer></div></body></html>')
    return "".join(out)


# --------------------------------------------------------------------------------------------
# markdown / json


_MD_SPECIAL = re.compile(r"([\\`*_\[\]|~])")


def _md(s: Any, limit: int | None = None) -> str:
    """Untrusted text as inert Markdown: no links/images, emphasis, HTML, table breaks or @mentions."""
    s = _one_line(s)
    if limit:
        s = _clip(s, limit)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _MD_SPECIAL.sub(r"\\\1", s).replace("@", "@​")  # zero-width space breaks the mention


def _code(s: Any) -> str:
    """Untrusted text as an inline code span (nothing inside is rendered)."""
    return "`" + _one_line(s).replace("`", "'").replace("|", "\\|") + "`"


def render_markdown(data: dict[str, Any]) -> str:
    """Compact Markdown (PR comments, Slack): headline numbers plus the top tables."""
    t, w, g, lp, lat = data["totals"], data["waste"], data["guard"], data["loops"], data["latency"]
    L: list[str] = []
    L.append(f"## llmtap report — {_md(_period_label(data))}")
    L.append("")
    L.append(f"**Spend {_usd(t['cost_usd'])}** · {_n(t['requests'])} requests · {_tok(t['total_tokens'])} tokens · "
             f"saved {_usd(t['saved_usd'])} by cache/replay · {_n(g['total'])} guard finding{'s' if g['total'] != 1 else ''}")
    if data.get("insights"):
        L.append("")
    for line in data.get("insights", []):
        L.append(f"- {line}")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Latency p50 / p95 | {_ms(lat['p50_ms'])} / {_ms(lat['p95_ms'])} |")
    L.append(f"| TTFT p50 | {_ms(lat['ttft_p50_ms'])} |")
    L.append(f"| Error rate | {_pct(t['error_rate'])} ({_n(t['errors'])} failed) |")
    L.append(f"| Blocked requests | {_n(t['blocked'])} |")
    if data["by_model"]:
        L += ["", "### Cost by model", "", "| Model | Requests | Tokens | Cost |", "|---|---:|---:|---:|"]
        for r in data["by_model"][:8]:
            L.append(f"| {_md(r['key'])} | {_n(r['requests'])} | {_tok(r['input_tokens'] + r['output_tokens'])} | {_usd(r['cost_usd'])} |")
    if len(data["by_day"]) > 1:
        L += ["", "### Cost by day", "", "| Day | Requests | Cost |", "|---|---:|---:|"]
        for r in data["by_day"][-14:]:
            L.append(f"| {r['key']} | {_n(r['requests'])} | {_usd(r['cost_usd'])} |")
    for tname, rows in list(data["by_tag"].items())[:2]:
        L += ["", f"### Cost by tag {_code(tname)}", "", "| Value | Requests | Cost |", "|---|---:|---:|"]
        for r in rows[:8]:
            L.append(f"| {_md(r['key'])} | {_n(r['requests'])} | {_usd(r['cost_usd'])} |")
    L += ["", f"### Waste finder — {_usd(w['savings_usd'])} avoidable"]
    if w["groups"]:
        L += ["", "| Prompt | Calls | Total | Saved with `--cache` |", "|---|---:|---:|---:|"]
        for r in w["groups"][:5]:
            L.append(f"| {_md(r['preview'] or '(no text)', 60)} | ×{r['count']} | {_usd(r['total_cost_usd'])} | {_usd(r['savings_usd'])} |")
    else:
        L += ["", "No duplicated prompts."]
    if data["expensive"]:
        L += ["", "### Most expensive requests", "", "| Model | Cost | Tokens | Prompt |", "|---|---:|---:|---|"]
        for r in data["expensive"][:5]:
            L.append(f"| {_md(r['model'] or '-')} | {_usd(r['cost_usd'])} | "
                     f"{_tok(r['input_tokens'])}→{_tok(r['output_tokens'])} | {_md(r['preview'], 60)} |")
    L += ["", "### Guard findings"]
    if g["kinds"]:
        L += ["", "| Kind | Count | Masked preview |", "|---|---:|---|"]
        for k in g["kinds"]:
            L.append(f"| {_md(k['kind'])} | {k['count']} | {' '.join(_code(p) for p in k['previews']) or '-'} |")
    else:
        L += ["", "None."]
    L += ["", f"### Runaway loops blocked: {_n(lp['blocked'])}"
          + (f" (~{_usd(lp['est_prevented_usd'])} prevented)" if lp["est_prevented_usd"] > 0 else "")]
    L += ["", f"_Generated by llmtap v{data.get('version', __version__)}_", ""]
    return "\n".join(L)


def render_json(data: dict[str, Any]) -> str:
    """Pretty JSON of the report dict."""
    return json.dumps(data, indent=2, default=str, ensure_ascii=False) + "\n"
