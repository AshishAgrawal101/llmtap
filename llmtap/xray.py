"""Context X-ray: where a request's input tokens (and dollars) go, which tool definitions bloat it, and why the
prompt cache missed.

Pure functions over :class:`~llmtap.models.Exchange` records, plus plain-text renderers:

* ``breakdown(ex)`` splits one request's input into ``system`` / ``tools`` / ``history`` / ``new_input`` /
  ``other``, estimates tokens per part from the serialized size (UTF-8 bytes / 4, images and documents at a
  flat ``MEDIA_TOKENS``) and scales the estimates so they sum to the input size the provider reported.
* ``tool_bloat(ex)`` ranks tool definitions; ``tool_servers(ex)`` groups MCP tools (``mcp__<server>__<tool>``)
  by server; ``tool_bloat_summary(exchanges)`` aggregates over a period.
* ``explain_cache(prev, cur)`` finds the first point where ``cur``'s prompt prefix diverged from ``prev`` in
  prompt-cache order (tools, system, messages) and prices the miss; ``cache_report(exchanges)`` walks a
  period, pairs each request with its predecessor in the same session and totals the dollars lost.
* ``render_xray_text`` / ``render_summary_text`` / ``summary_json`` present the results.

Reported usage semantics (kept consistent with :mod:`llmtap.providers` and :func:`llmtap.pricing.cost_usd`):
Anthropic and Bedrock report ``input_tokens`` EXCLUDING cache reads/writes, so the prompt size is
``input + cache_read + cache_write``; OpenAI, Gemini and other OpenAI-compatible APIs report cached tokens as a
subset of ``input_tokens``. Per-part dollars are the request's input cost split cache-aware: the leading
``cache_read`` tokens (tools first, then system, history, new input) are weighted at the cache-read price, the
next ``cache_write`` tokens at the cache-write price and the rest at the input price.

Nothing here raises on malformed, truncated or unknown request bodies: they degrade to an ``other`` part.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from llmtap.models import Exchange, Usage
from llmtap.pricing import cost_usd, normalize_model, price_for
from llmtap.term import clean, fmt_tokens, fmt_usd, paint

__all__ = ["breakdown", "tool_bloat", "tool_servers", "tool_bloat_summary", "explain_cache", "cache_report",
           "render_xray_text", "render_summary_text", "summary_json", "PARTS"]

PARTS = ("system", "tools", "history", "new_input", "other")      # display order
CACHE_ORDER = ("tools", "system", "other", "history", "new_input")  # prompt-cache (prefix) order
PART_LABELS = {"system": "system", "tools": "tools", "history": "history", "new_input": "new input",
               "other": "other"}
_PART_HUE = {"system": "blue", "tools": "magenta", "history": "cyan", "new_input": "green", "other": "gray"}

MEDIA_TOKENS = 1500            # flat estimate per image / document block (billed by size, not by base64 length)
MESSAGE_OVERHEAD = 4           # role / turn separator tokens per message
SESSION_GAP_S = 3600.0         # auto-grouped (untagged) requests further apart than this start a new session
ANTHROPIC_TTL_S = 300.0        # default ephemeral cache lifetime
ANTHROPIC_TTL_1H_S = 3600.0    # cache_control {"ttl": "1h"}
OPENAI_TTL_S = 600.0           # OpenAI keeps prefixes 5-10 minutes of inactivity (best effort)
OPENAI_TTL_24H_S = 86400.0     # prompt_cache_retention: "24h"
OTHER_TTL_S = 300.0            # Gemini implicit caching and unknown providers
MIN_AUTO_CACHE_TOKENS = 1024   # OpenAI / Gemini cache nothing below this prompt size
MIN_LOST_TOKENS = 256          # smaller shortfalls are noise, not a miss
HIT_RATIO = 0.8                # cache_read >= 80% of the expected prefix counts as a hit
PARENT_CANDIDATES = 8          # recent requests per session considered as the "previous" request
_MEMO_SIZE = 32

_TRUNC_MARK = "[llmtap: truncated,"
_B64_RE = re.compile(r"[A-Za-z0-9+/=_-]+")
_VOLATILE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-|\b\d{10,13}\b|"
                          r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\b")
_TTL_1H_RE = re.compile(r'"ttl"\s*:\s*"1h"')
_RETENTION_24H_RE = re.compile(r'"prompt_cache_retention"\s*:\s*"24h"')

_MEDIA_TYPES = frozenset({"image", "image_url", "input_image", "document", "input_file", "file", "input_audio",
                          "audio", "video"})
_STRUCT_KEYS = frozenset({"type", "role", "content", "text", "parts", "source", "id", "media_type", "mimeType"})
_SKIP_KEYS = frozenset({"cache_control", "cachePoint"})
_CONSUMED_KEYS = frozenset({"system", "instructions", "systemInstruction", "system_instruction", "messages",
                            "contents", "input", "prompt", "inputText", "tools", "functions", "toolConfig"})
# Request settings that are not prompt text (they are not tokenized as input).
_CONFIG_KEYS = frozenset({
    "model", "max_tokens", "max_completion_tokens", "max_output_tokens", "temperature", "top_p", "top_k",
    "stream", "stream_options", "stop", "stop_sequences", "n", "seed", "user", "metadata", "store",
    "service_tier", "parallel_tool_calls", "tool_choice", "function_call", "logprobs", "top_logprobs",
    "presence_penalty", "frequency_penalty", "logit_bias", "reasoning", "reasoning_effort", "thinking",
    "anthropic_version", "anthropic_beta", "betas", "inferenceConfig", "generationConfig", "generation_config",
    "safetySettings", "safety_settings", "previous_response_id", "truncation", "include", "modalities", "audio",
    "prompt_cache_key", "prompt_cache_retention", "safety_identifier", "verbosity", "speed", "effort",
    "container", "context_management", "mcp_servers", "additionalModelRequestFields",
    "additionalModelResponseFieldPaths", "performanceConfig", "guardrailConfig", "promptVariables",
    "requestMetadata", "cachedContent", "cached_content", "labels", "background", "max_tool_calls",
    "conversation", "tool_config", "echo", "best_of", "suffix", "web_search_options",
})
_ASSISTANT_ROLES = frozenset({"assistant", "model"})
_SYSTEM_ROLES = frozenset({"system", "developer"})


# --------------------------------------------------------------------------------------------
# token estimation


def _text_tokens(s: str) -> float:
    """~UTF-8 bytes / 4: chars/4 for ASCII, a bit more per char for accented / CJK text."""
    if not s:
        return 0.0
    if s.isascii():
        return len(s) / 4.0
    return len(s.encode("utf-8", "ignore")) / 4.0


def _is_blob(s: str) -> bool:
    """Inline binary payloads (base64 images/PDFs, data: URLs) are billed by content size, not text length."""
    return s.startswith("data:") or (len(s) > 512 and _B64_RE.fullmatch(s[:512]) is not None)


def _walk_tokens(obj: Any, acc: list[float], depth: int = 0) -> None:
    """Add the estimated tokens of a message-like value to ``acc[0]`` (media blocks counted in ``acc[1]``)."""
    if depth > 64:
        return
    if isinstance(obj, str):
        if _is_blob(obj):
            acc[0] += MEDIA_TOKENS
            acc[1] += 1
        else:
            acc[0] += _text_tokens(obj)
    elif isinstance(obj, dict):
        t = obj.get("type")
        if isinstance(t, str) and t in _MEDIA_TYPES:
            acc[0] += MEDIA_TOKENS
            acc[1] += 1
            return
        acc[0] += 1.0
        for k, v in obj.items():
            if k in _SKIP_KEYS:
                continue
            if k in _STRUCT_KEYS:
                if isinstance(v, str) and k != "text" and k != "content":
                    continue                       # "type": "text", "role": "user" ... are framing, not text
            else:
                acc[0] += _text_tokens(str(k))
            _walk_tokens(v, acc, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_tokens(v, acc, depth + 1)
    elif obj is not None:
        acc[0] += 1.0


def _canon(obj: Any) -> str:
    """Stable JSON text of a (parsed) value, for size estimates and prefix comparison."""
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError, RecursionError):
        return str(obj)


def _strip_cc(obj: Any, depth: int = 0) -> Any:
    """Drop cache breakpoints (``cache_control`` keys, Converse ``cachePoint`` blocks): moving one does not change
    the prompt."""
    if depth > 64:
        return obj
    if isinstance(obj, dict):
        return {k: _strip_cc(v, depth + 1) for k, v in obj.items() if k not in _SKIP_KEYS}
    if isinstance(obj, list):
        return [_strip_cc(v, depth + 1) for v in obj
                if not (isinstance(v, dict) and v and set(v) <= {"cachePoint"})]
    return obj


# --------------------------------------------------------------------------------------------
# request analysis (memoized per exchange body)


@dataclass
class _Tool:
    name: str
    server: str | None      # MCP server ("mcp__<server>__<tool>" or a Responses-API mcp tool), else None
    kind: str               # "mcp" | "function" | "builtin"
    text: str               # canonical JSON of the definition (cache breakpoints stripped)
    obj: Any
    raw: float              # estimated tokens (unscaled)


@dataclass
class _Analysis:
    shape: str = "unknown"
    parsed: bool = False
    truncated: bool = False
    error: str | None = None
    req: dict | None = None
    raw: dict[str, float] = field(default_factory=lambda: {p: 0.0 for p in PARTS})
    tools: list[_Tool] = field(default_factory=list)
    system: list[str] = field(default_factory=list)                    # top-level system texts
    conv: list[tuple[str, Any]] = field(default_factory=list)          # (role, item) incl. system-role messages
    new_start: int = 0                                                  # index in conv where new_input starts
    media: int = 0
    msg_tokens: list[float] = field(default_factory=list)              # estimated tokens per conv item
    fp: Any = None                                                      # _FP prompt fingerprint (cache_report)


_MEMO: OrderedDict[tuple[Any, ...], _Analysis] = OrderedDict()
_MEMO_LOCK = threading.Lock()


def _parse_body(text: Any) -> tuple[dict | None, bool, str | None]:
    """(request object, truncated, error)."""
    if not isinstance(text, str) or not text.strip():
        return None, False, "empty request body"
    truncated = _TRUNC_MARK in text[-200:]
    try:
        obj = json.loads(text)
    except Exception:
        if truncated:
            return None, True, "request body was truncated in llmtap storage (2 MiB cap)"
        if text.startswith("[") and "not stored" in text[:200]:
            return None, False, "request body was not stored (binary / multipart)"
        return None, False, "request body is not valid JSON"
    if not isinstance(obj, dict):
        return None, truncated, "request body is not a JSON object"
    return obj, truncated, None


def _analyze(ex: Exchange) -> _Analysis:
    body = ex.request_body if isinstance(ex.request_body, str) else ""
    key = (getattr(ex, "id", None), hash(body), str(getattr(ex, "style", "") or ""))
    with _MEMO_LOCK:
        hit = _MEMO.get(key)
        if hit is not None:
            _MEMO.move_to_end(key)
            return hit
    an = _build(body, key[2])
    with _MEMO_LOCK:
        _MEMO[key] = an
        while len(_MEMO) > _MEMO_SIZE:
            _MEMO.popitem(last=False)
    return an


def _is_converse(req: dict) -> bool:
    if any(k in req for k in ("toolConfig", "inferenceConfig", "additionalModelRequestFields")):
        return True
    for m in req.get("messages") or []:
        if isinstance(m, dict) and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict):
                    return "type" not in b and bool(set(b) & {"text", "toolUse", "toolResult", "image", "document",
                                                             "reasoningContent", "cachePoint"})
    return False


def _shape(req: dict, style: str) -> str:
    if isinstance(req.get("contents"), list):
        return "gemini"
    if isinstance(req.get("messages"), list):
        if _is_converse(req):
            return "bedrock-converse"
        if "anthropic_version" in req or "system" in req or style in ("anthropic", "bedrock"):
            return "anthropic"
        return "openai-chat"
    if "input" in req and isinstance(req.get("input"), (str, list)):
        return "openai-responses"
    if isinstance(req.get("prompt"), (str, list)) or isinstance(req.get("inputText"), str):
        return "completion"
    return "unknown"


def _system_texts(v: Any, depth: int = 0) -> list[str]:
    if v is None or depth > 16:
        return []
    if isinstance(v, str):
        return [v] if v else []
    if isinstance(v, dict):
        if isinstance(v.get("parts"), list):
            return _system_texts(v["parts"], depth + 1)
        if isinstance(v.get("text"), str):
            return [v["text"]]
        if "content" in v:
            return _system_texts(v["content"], depth + 1)
        v = _strip_cc(v)
        return [_canon(v)] if v else []
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            out.extend(_system_texts(x, depth + 1))
        return out
    return [str(v)]


def _mcp_split(name: str) -> str | None:
    if name.startswith("mcp__"):
        rest = name[5:]
        return rest.split("__", 1)[0] if "__" in rest else (rest or None)
    return None


def _tools_of(req: dict, shape: str) -> list[_Tool]:
    raw_items: list[Any] = []
    for key in ("tools", "functions"):
        if isinstance(req.get(key), list):
            raw_items.extend(req[key])
    tc = req.get("toolConfig")
    if shape == "bedrock-converse" and isinstance(tc, dict) and isinstance(tc.get("tools"), list):
        raw_items.extend(tc["tools"])
    out: list[_Tool] = []

    def add(name: Any, obj: Any, kind: str, server: str | None = None) -> None:
        name = str(name) if name not in (None, "") else "?"
        obj = _strip_cc(obj)
        if server is None:
            server = _mcp_split(name)
        if server is not None:
            kind = "mcp"
        text = _canon(obj)
        out.append(_Tool(name=name, server=server, kind=kind, text=text, obj=obj, raw=_text_tokens(text) + 2.0))

    for t in raw_items:
        if not isinstance(t, dict):
            if t is not None:
                add(str(t)[:40], t, "function")
            continue
        decls = t.get("functionDeclarations", t.get("function_declarations"))
        if isinstance(decls, list):                                     # Gemini
            for d in decls:
                add(d.get("name") if isinstance(d, dict) else None, d, "function")
            rest = {k: v for k, v in t.items() if k not in ("functionDeclarations", "function_declarations")}
            for k, v in rest.items():
                add(k, {k: v}, "builtin")
            continue
        if isinstance(t.get("toolSpec"), dict):                         # Bedrock Converse
            add(t["toolSpec"].get("name"), t, "function")
            continue
        if set(t) <= {"cachePoint"}:
            continue
        if isinstance(t.get("function"), dict):                         # OpenAI chat
            add(t["function"].get("name"), t, "function")
            continue
        typ = t.get("type")
        if typ == "mcp":                                                # OpenAI Responses remote MCP server
            label = str(t.get("server_label") or t.get("server_url") or "mcp")
            add(f"mcp:{label}", t, "mcp", server=label)
            continue
        if "name" in t:
            schema = any(k in t for k in ("input_schema", "parameters", "inputSchema"))
            builtin = isinstance(typ, str) and typ not in ("custom", "function") and not schema
            add(t.get("name"), t, "builtin" if builtin else "function")
            continue
        if len(t) == 1 and shape == "gemini":                           # {"googleSearch": {}}
            k = next(iter(t))
            add(k, t, "builtin")
            continue
        add(typ or "?", t, "builtin")
    return out


def _is_assistant(role: str, item: Any) -> bool:
    if role in _ASSISTANT_ROLES:
        return True
    if role == "" and isinstance(item, dict):
        typ = str(item.get("type") or "")
        return typ.endswith("_call") or typ == "reasoning"
    return False


def _conversation(req: dict, shape: str) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if shape == "gemini":
        for c in req.get("contents") or []:
            role = str(c.get("role") or "user") if isinstance(c, dict) else "user"
            out.append((role, c))
    elif shape in ("anthropic", "bedrock-converse", "openai-chat"):
        for m in req.get("messages") or []:
            role = str(m.get("role") or "") if isinstance(m, dict) else "user"
            out.append((role, m))
    elif shape == "openai-responses":
        inp = req.get("input")
        if isinstance(inp, str):
            out.append(("user", inp))
        elif isinstance(inp, list):
            for it in inp:
                if isinstance(it, dict):
                    out.append((str(it.get("role") or ""), it))
                else:
                    out.append(("user", it))
    elif shape == "completion":
        p = req.get("prompt") if req.get("prompt") is not None else req.get("inputText")
        out.append(("user", p))
    return out


def _build(body: str, style: str) -> _Analysis:
    an = _Analysis()
    req, an.truncated, an.error = _parse_body(body)
    if req is None:
        an.shape = "unparsed"
        an.raw["other"] = _text_tokens(body) if an.error and "not stored" not in an.error else 0.0
        return an
    an.req, an.parsed = req, True
    an.shape = _shape(req, style)
    an.tools = _tools_of(req, an.shape)
    an.raw["tools"] = sum(t.raw for t in an.tools)
    for key in ("system", "instructions", "systemInstruction", "system_instruction"):
        if key in req and not (key == "instructions" and an.shape not in ("openai-responses", "unknown")):
            an.system.extend(_system_texts(req[key]))
    an.raw["system"] = sum(_text_tokens(s) for s in an.system) + (2.0 if an.system else 0.0)
    an.conv = _conversation(req, an.shape)
    turns = [i for i, (role, _) in enumerate(an.conv) if role not in _SYSTEM_ROLES]
    last_asst = max((i for i in turns if _is_assistant(an.conv[i][0], an.conv[i][1])), default=-1)
    start = last_asst + 1
    if turns and start > turns[-1]:              # ends with an assistant prefill: new input = last non-assistant turn
        start = max((i for i in turns if not _is_assistant(an.conv[i][0], an.conv[i][1])), default=len(an.conv))
    an.new_start = start
    for i, (role, item) in enumerate(an.conv):
        acc = [0.0, 0]
        _walk_tokens(item, acc)
        an.media += int(acc[1])
        part = "system" if role in _SYSTEM_ROLES else ("new_input" if i >= start else "history")
        an.msg_tokens.append(acc[0] + MESSAGE_OVERHEAD)
        an.raw[part] += acc[0] + MESSAGE_OVERHEAD
    for k, v in req.items():
        if k in _CONSUMED_KEYS or k in _CONFIG_KEYS:
            continue
        acc = [0.0, 0]
        _walk_tokens(v, acc)
        an.media += int(acc[1])
        an.raw["other"] += acc[0] + _text_tokens(str(k))
    return an


# --------------------------------------------------------------------------------------------
# usage / pricing helpers


def _anthropic_usage(ex: Exchange) -> bool:
    """Anthropic and Bedrock report input_tokens EXCLUDING cache reads/writes (see pricing.cost_usd)."""
    return str(getattr(ex, "style", "") or "") in ("anthropic", "bedrock")


def _u(ex: Exchange) -> Usage:
    u = getattr(ex, "usage", None)
    return u if isinstance(u, Usage) else Usage()


def _cache_counts(ex: Exchange) -> tuple[int, int, int]:
    u = _u(ex)
    return (max(int(u.input_tokens or 0), 0), max(int(u.cache_read_tokens or 0), 0),
            max(int(u.cache_write_tokens or 0), 0))


def _prompt_total(ex: Exchange) -> int:
    inp, cr, cw = _cache_counts(ex)
    if _anthropic_usage(ex):
        return inp + cr + cw
    return inp if cr <= inp else inp + cr          # cached > input: the numbers are clearly exclusive


def _reported_prompt(ex: Exchange) -> int | None:
    """The input size the provider reported, or None when usage is missing or was estimated by llmtap."""
    if _u(ex).estimated:
        return None
    n = _prompt_total(ex)
    return n if n > 0 else None


def _billable(ex: Exchange) -> bool:
    status = int(getattr(ex, "status", 0) or 0)
    return getattr(ex, "source", "upstream") == "upstream" and (status == 0 or 200 <= status < 300)


def _multiplier(ex: Exchange) -> float:
    u = _u(ex)
    m = 2.0 if u.speed == "fast" else 1.0
    return m * (0.5 if u.service_tier == "flex" else 1.0)


def _apportion(raw: dict[str, float], total: int) -> dict[str, int]:
    """Integer split of ``total`` proportional to ``raw`` (largest remainder), summing exactly to ``total``."""
    keys = list(raw)
    s = sum(max(v, 0.0) for v in raw.values())
    if total <= 0:
        return {k: 0 for k in keys}
    if s <= 0:
        out = {k: 0 for k in keys}
        out["other" if "other" in out else keys[-1]] = total
        return out
    exact = {k: max(raw[k], 0.0) * total / s for k in keys}
    out = {k: int(exact[k]) for k in keys}
    rest = total - sum(out.values())
    for k in sorted(keys, key=lambda k: exact[k] - out[k], reverse=True)[:rest]:
        out[k] += 1
    return out


def _split_cost(tokens: dict[str, int], total_usd: float, rates: dict[str, float] | None, cr: int, cw: int,
                anthropic: bool) -> dict[str, float]:
    """Split the input cost over parts in cache order: cache reads first, then cache writes, then uncached."""
    r_in = rates["input"] if rates else 1.0
    r_cr = rates["cache_read"] if rates else 1.0
    r_cw = rates["cache_write"] if rates else 1.0
    left_cr, left_cw = cr, (cw if anthropic else 0)
    weights: dict[str, float] = {}
    for p in CACHE_ORDER:
        n = tokens.get(p, 0)
        a = min(n, left_cr)
        left_cr -= a
        b = min(n - a, left_cw)
        left_cw -= b
        weights[p] = a * r_cr + b * r_cw + (n - a - b) * r_in
    w = sum(weights.values())
    if w <= 0 or total_usd <= 0:
        return {p: 0.0 for p in tokens}
    return {p: round(total_usd * weights.get(p, 0.0) / w, 10) for p in tokens}


def _safe_cost(model: str | None, usage: Usage, overrides: dict | None, style: str | None) -> float:
    try:
        return float(cost_usd(model, usage, overrides, style=style or None))
    except Exception:
        return 0.0


def _safe_price(model: str | None, overrides: dict | None) -> dict[str, float] | None:
    try:
        return price_for(model, overrides)
    except Exception:
        return None


# --------------------------------------------------------------------------------------------
# 1. context X-ray


def breakdown(ex: Exchange, overrides: dict | None = None) -> dict[str, Any]:
    """Split one request's input into parts with tokens, share and estimated dollars. Never raises.

    Returns ``{"parts": {part: {"tokens", "share", "usd", "raw_tokens"}}, "input_tokens", "scaled",
    "cache_read_tokens", "cache_write_tokens", "input_cost_usd", "shape", "parsed", "truncated", "notes", ...}``.
    ``parts`` always holds system / tools / history / new_input, plus ``other`` when it is non-empty. When the
    provider reported usage (``scaled``), part tokens are the estimates scaled to sum to the reported prompt
    size (Anthropic: input + cache_read + cache_write; OpenAI-style: input); otherwise they are raw estimates.
    """
    try:
        return _breakdown(ex, overrides)
    except Exception as e:  # pragma: no cover - defensive: analysis must never break a caller
        return {"id": getattr(ex, "id", None), "parts": {p: {"tokens": 0, "share": 0.0, "usd": 0.0, "raw_tokens": 0}
                                                         for p in PARTS[:4]},
                "input_tokens": 0, "scaled": False, "estimated": True, "parsed": False, "truncated": False,
                "error": f"analysis failed: {e}", "notes": [f"analysis failed: {e}"], "shape": "unparsed",
                "cache_read_tokens": 0, "cache_write_tokens": 0, "uncached_input_tokens": 0, "input_cost_usd": 0.0,
                "scale": 1.0, "raw_estimate_tokens": 0, "reported_input_tokens": None, "counts": {}}


def _breakdown(ex: Exchange, overrides: dict | None) -> dict[str, Any]:
    an = _analyze(ex)
    raw = dict(an.raw)
    est_total = sum(raw.values())
    reported = _reported_prompt(ex)
    inp, cr, cw = _cache_counts(ex)
    anth = _anthropic_usage(ex)
    notes: list[str] = []
    if an.error:
        notes.append(an.error + ("; everything is counted as 'other'" if not an.parsed else ""))
    elif an.truncated:
        notes.append("request body was truncated in llmtap storage")
    if an.parsed and an.shape == "unknown":
        notes.append("unrecognized request shape; everything is counted as 'other'")
    if reported:
        target = reported
        scale = reported / est_total if est_total > 0 else 1.0
        in_usage = replace(_u(ex), output_tokens=0, web_search_requests=0)
        if est_total > 0 and (scale > 3.0 or scale < 0.33):
            notes.append(f"estimates scaled x{scale:.2f} to match reported usage (images, documents or "
                         f"provider-side prompt text make the estimate rough)")
    else:
        target = int(round(est_total))
        scale = 1.0
        cr = cw = 0
        in_usage = Usage(input_tokens=target, estimated=True)
        if est_total > 0:
            notes.append("no reported usage: token counts are size-based estimates")
    tokens = _apportion(raw, target)
    input_cost = _safe_cost(ex.model, in_usage, overrides, getattr(ex, "style", None))
    usd = _split_cost(tokens, input_cost, _safe_price(ex.model, overrides), cr, cw, anth)
    parts: dict[str, dict[str, Any]] = {}
    for p in PARTS:
        if p == "other" and tokens[p] == 0 and raw[p] <= 0:
            continue
        parts[p] = {"tokens": tokens[p], "share": (tokens[p] / target) if target else 0.0, "usd": usd.get(p, 0.0),
                    "raw_tokens": int(round(raw[p]))}
    uncached = (inp if anth else max(inp - cr, 0)) if reported else target
    return {
        "id": ex.id, "model": ex.model, "provider": ex.provider, "style": ex.style, "shape": an.shape,
        "parsed": an.parsed, "truncated": an.truncated, "error": an.error, "notes": notes,
        "scaled": bool(reported), "estimated": not reported, "scale": round(scale, 4),
        "input_tokens": target, "reported_input_tokens": reported, "raw_estimate_tokens": int(round(est_total)),
        "cache_read_tokens": cr, "cache_write_tokens": cw, "uncached_input_tokens": uncached,
        "input_cost_usd": round(input_cost, 10), "parts": parts,
        "counts": {"tools": len(an.tools), "messages": sum(1 for r, _ in an.conv if r not in _SYSTEM_ROLES),
                   "new_input_messages": max(len(an.conv) - an.new_start, 0), "system_blocks": len(an.system)
                   + sum(1 for r, _ in an.conv if r in _SYSTEM_ROLES), "media": an.media},
    }


# --------------------------------------------------------------------------------------------
# 2. tool / MCP bloat


def _tool_items(ex: Exchange, bd: dict[str, Any]) -> list[dict[str, Any]]:
    an = _analyze(ex)
    if not an.tools:
        return []
    part = bd["parts"].get("tools") or {"tokens": 0, "usd": 0.0}
    alloc = _apportion({str(i): t.raw for i, t in enumerate(an.tools)}, int(part["tokens"]))
    raw_sum = sum(t.raw for t in an.tools) or 1.0
    total = bd["input_tokens"] or 0
    items = []
    for i, t in enumerate(an.tools):
        n = alloc[str(i)]
        items.append({"name": t.name, "server": t.server, "kind": t.kind, "tokens": n,
                      "share": (n / total) if total else 0.0, "usd": round(part["usd"] * t.raw / raw_sum, 10)})
    items.sort(key=lambda d: (-d["tokens"], d["name"]))
    return items


def tool_bloat(ex: Exchange, overrides: dict | None = None) -> list[dict[str, Any]]:
    """Tool definitions of one request ranked by estimated tokens.

    Each item: ``{"name", "server", "kind", "tokens", "share", "usd"}``. ``server`` is the MCP server for
    ``mcp__<server>__<tool>`` names (and OpenAI Responses ``{"type": "mcp"}`` tools), else None; ``share`` is
    the fraction of the whole request input; ``usd`` this request's estimated cost of the definition."""
    try:
        return _tool_items(ex, breakdown(ex, overrides))
    except Exception:  # pragma: no cover - defensive
        return []


def _group_servers(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for it in items:
        if not it.get("server"):
            continue
        g = groups.setdefault(it["server"], {"server": it["server"], "tools": 0, "tokens": 0, "share": 0.0,
                                             "usd": 0.0})
        g["tools"] += 1
        g["tokens"] += it["tokens"]
        g["share"] += it["share"]
        g["usd"] = round(g["usd"] + it["usd"], 10)
    return sorted(groups.values(), key=lambda g: (-g["tokens"], g["server"]))


def tool_servers(ex: Exchange, overrides: dict | None = None) -> list[dict[str, Any]]:
    """MCP tools of one request grouped by server: ``[{"server", "tools", "tokens", "share", "usd"}]``."""
    return _group_servers(tool_bloat(ex, overrides))


class _Acc:
    """One pass over a period: per-part totals and tool-definition totals (each body parsed once)."""

    def __init__(self, overrides: dict | None) -> None:
        self.overrides = overrides
        self.requests = 0
        self.analyzed = 0
        self.unparsed = 0
        self.truncated = 0
        self.scaled = 0
        self.input_tokens = 0
        self.input_usd = 0.0
        self.parts = {p: {"tokens": 0, "usd": 0.0} for p in PARTS}
        self.with_tools = 0
        self.tool_tokens = 0
        self.tool_usd = 0.0
        self.tool_count = 0
        self.tools: dict[str, dict[str, Any]] = {}
        self.servers: dict[str, dict[str, Any]] = {}

    def add(self, ex: Exchange) -> None:
        self.requests += 1
        if not _billable(ex):
            return
        bd = breakdown(ex, self.overrides)
        if bd["truncated"]:
            self.truncated += 1
        if not bd["parsed"]:
            self.unparsed += 1
            return
        self.analyzed += 1
        self.scaled += 1 if bd["scaled"] else 0
        self.input_tokens += bd["input_tokens"]
        self.input_usd += bd["input_cost_usd"]
        for p, d in bd["parts"].items():
            self.parts[p]["tokens"] += d["tokens"]
            self.parts[p]["usd"] += d["usd"]
        items = _tool_items(ex, bd)
        if not items:
            return
        self.with_tools += 1
        self.tool_count += len(items)
        seen: dict[str, list[int | float]] = {}
        for it in items:
            self.tool_tokens += it["tokens"]
            self.tool_usd += it["usd"]
            t = self.tools.setdefault(it["name"], {"name": it["name"], "server": it["server"], "requests": 0,
                                                   "total_tokens": 0, "usd": 0.0})
            t["requests"] += 1
            t["total_tokens"] += it["tokens"]
            t["usd"] += it["usd"]
            if it["server"]:
                s = seen.setdefault(it["server"], [0, 0.0])
                s[0] += it["tokens"]
                s[1] += it["usd"]
                srv = self.servers.setdefault(it["server"], {"server": it["server"], "names": set(), "requests": 0,
                                                             "total_tokens": 0, "usd": 0.0})
                srv["names"].add(it["name"])
        for name, (tok, usd) in seen.items():
            srv = self.servers[name]
            srv["requests"] += 1
            srv["total_tokens"] += int(tok)
            srv["usd"] += float(usd)

    def tool_summary(self, top: int = 10) -> dict[str, Any]:
        n = self.analyzed
        tools = sorted(self.tools.values(), key=lambda t: (-t["total_tokens"], t["name"]))
        top_tools = [{**t, "avg_tokens": int(round(t["total_tokens"] / t["requests"])), "usd": round(t["usd"], 10)}
                     for t in tools[:top]]
        servers = []
        for s in sorted(self.servers.values(), key=lambda s: (-s["total_tokens"], s["server"])):
            servers.append({"server": s["server"], "tools": len(s["names"]), "requests": s["requests"],
                            "avg_tokens": int(round(s["total_tokens"] / s["requests"])) if s["requests"] else 0,
                            "total_tokens": s["total_tokens"], "usd": round(s["usd"], 10)})
        headlines: list[str] = []
        if n and self.tool_tokens:
            share = self.tool_tokens / self.input_tokens if self.input_tokens else 0.0
            headlines.append(f"tool definitions average ~{fmt_tokens(self.tool_tokens / n)} tokens per request "
                             f"({share:.0%} of input), ~{fmt_usd(self.tool_usd)} over {n} request"
                             f"{'s' if n != 1 else ''}")
        for s in servers[:3]:
            every = "every request" if s["requests"] == n else f"{s['requests']} of {n} requests"
            one = s["tools"] == 1
            headlines.append(f"your {s['tools']} MCP tool{'' if one else 's'} from '{clean(s['server'])}'"
                             f" add{'s' if one else ''} ~{fmt_tokens(s['avg_tokens'])} tokens to {every} "
                             f"(~{fmt_usd(s['usd'])} over {s['requests']} request{'s' if s['requests'] != 1 else ''})")
        return {
            "requests": n, "requests_with_tools": self.with_tools,
            "avg_tool_tokens": int(round(self.tool_tokens / n)) if n else 0,
            "avg_tools_per_request": round(self.tool_count / n, 2) if n else 0.0,
            "avg_tool_share": (self.tool_tokens / self.input_tokens) if self.input_tokens else 0.0,
            "total_tool_tokens": self.tool_tokens, "total_tool_usd": round(self.tool_usd, 10),
            "top_tools": top_tools, "servers": servers, "headlines": headlines,
        }


def tool_bloat_summary(exchanges: Iterable[Exchange], overrides: dict | None = None, top: int = 10) -> dict[str, Any]:
    """Period summary of tool-definition overhead over billable (upstream, 2xx) requests.

    ``{"requests", "requests_with_tools", "avg_tool_tokens", "avg_tool_share", "total_tool_tokens",
    "total_tool_usd", "top_tools": [...], "servers": [...], "headlines": [str]}`` where ``servers`` groups MCP
    tools by server and ``headlines`` reads like "your 14 MCP tools from 'github' add ~18k tokens to every
    request (~$0.71 over 120 requests)"."""
    acc = _Acc(overrides)
    for ex in exchanges:
        try:
            acc.add(ex)
        except Exception:  # pragma: no cover - defensive
            continue
    return acc.tool_summary(top)


# --------------------------------------------------------------------------------------------
# 3. cache-miss explainer


@dataclass
class _Unit:
    section: str     # "system" | "messages"
    index: int       # message index (or 0 for the top-level system prompt)
    role: str
    text: str


def _item_text(item: Any) -> str:
    """Comparable text of one message: plain text when it is only text, else canonical JSON."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        item = _strip_cc(item)
        c = item.get("content", item.get("parts"))
        extra = set(item) - {"role", "content", "parts", "name"}
        if not extra:
            if isinstance(c, str):
                return c
            if isinstance(c, list) and c and all(isinstance(b, dict) and isinstance(b.get("text"), str)
                                                 and set(b) <= {"type", "text"} for b in c):
                return "\n\n".join(b["text"] for b in c)
    return _canon(item)


def _units(an: _Analysis) -> tuple[str | None, list[_Unit]]:
    system = "\n\n".join(an.system) if an.system else None
    msgs = [_Unit("system" if role in _SYSTEM_ROLES else "messages", i, role or "item", _item_text(item))
            for i, (role, item) in enumerate(an.conv)]
    return system, msgs


def _common_prefix(a: str, b: str) -> int:
    hi = min(len(a), len(b))
    if a[:hi] == b[:hi]:
        return hi
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _common_suffix(a: str, b: str, limit: int) -> int:
    lo, hi = 0, max(limit, 0)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[len(a) - mid:] == b[len(b) - mid:]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _esc(s: str) -> str:
    return clean(s.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t"))


def _snippet(a: str, b: str, ctx: int = 20, span: int = 40) -> tuple[int, str, str, bool]:
    """(first differing char, before, after, looks-volatile) with a little context around the change."""
    i = _common_prefix(a, b)
    j = _common_suffix(a, b, min(len(a), len(b)) - i)
    left = max(0, i - ctx)

    def show(s: str) -> str:
        end = len(s) - j
        mid = s[i:end]
        if len(mid) > span:
            mid = mid[:span] + "…"
        right = s[end:end + ctx // 2]
        return (("…" if left > 0 else "") + _esc(s[left:i] + mid + right)
                + ("…" if end + ctx // 2 < len(s) else ""))

    window_a = a[max(0, i - 12):len(a) - j + 12]
    window_b = b[max(0, i - 12):len(b) - j + 12]
    volatile = bool(_VOLATILE_RE.search(window_a)) and bool(_VOLATILE_RE.search(window_b))
    return i, show(a), show(b), volatile


_VOLATILE_HINT = " (looks like a timestamp/ID that changes every request: keep dynamic values after the last cache " \
                 "breakpoint)"


def _first_diff(a: Any, b: Any, path: str = "", depth: int = 0) -> tuple[str, Any, Any] | None:
    if a == b:
        return None
    if depth < 32 and isinstance(a, dict) and isinstance(b, dict):
        for k in list(a) + [k for k in b if k not in a]:
            if k not in a or k not in b:
                return (f"{path}.{k}" if path else str(k)), a.get(k), b.get(k)
            d = _first_diff(a[k], b[k], f"{path}.{k}" if path else str(k), depth + 1)
            if d:
                return d
        return None
    if depth < 32 and isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for n, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{n}]", depth + 1)
            if d:
                return d
        return None
    return path, a, b


def _describe_path(path: str) -> str:
    for pre in ("function.", "toolSpec.", "toolSpec"):
        if path.startswith(pre):
            path = path[len(pre):]
    if path in ("description",):
        return "description"
    if path.split(".")[0].split("[")[0] in ("input_schema", "parameters", "inputSchema"):
        return f"input schema ({path})" if "." in path or "[" in path else "input schema"
    return f"'{path}'" if path else "definition"


def _diff_tools(pt: list[_Tool], ct: list[_Tool]) -> dict[str, Any] | None:
    if [t.text for t in pt] == [t.text for t in ct]:
        return None
    pn, cn = [t.name for t in pt], [t.name for t in ct]
    if pn == cn:
        for i, (a, b) in enumerate(zip(pt, ct)):
            if a.text != b.text:
                d = _first_diff(a.obj, b.obj)
                what = _describe_path(d[0]) if d else "definition"
                detail = f"tool definitions changed: tool '{clean(a.name)}' {what} edited"
                if d and isinstance(d[1], str) and isinstance(d[2], str):
                    pos, x, y, vol = _snippet(d[1], d[2])
                    detail += f" at char {pos:,}: '{x}' -> '{y}'" + (_VOLATILE_HINT if vol else "")
                return {"cause": "tools_changed", "section": "tools", "index": i, "label": a.name, "detail": detail}
    first = next((i for i, (a, b) in enumerate(zip(pn, cn)) if a != b), min(len(pn), len(cn)))
    if sorted(pn) == sorted(cn):
        name = pn[first]
        detail = (f"tool order changed: '{clean(name)}' moved from position {first} to {cn.index(name)}"
                  f" (sort tool definitions deterministically)")
        return {"cause": "tool_order_changed", "section": "tools", "index": first, "label": name, "detail": detail}
    added = [n for n in cn if n not in set(pn)]
    removed = [n for n in pn if n not in set(cn)]

    def names(xs: list[str]) -> str:
        shown = ", ".join(f"'{clean(x)}'" for x in xs[:3])
        return shown + (f" (+{len(xs) - 3} more)" if len(xs) > 3 else "")

    bits = []
    if added:
        bits.append(f"{len(added)} added: {names(added)}")
    if removed:
        bits.append(f"{len(removed)} removed: {names(removed)}")
    if not bits:                                       # same names, different multiplicity / edits + reorder
        bits.append(f"tool #{first} changed")
    detail = "tool definitions changed: " + "; ".join(bits)
    servers = {_mcp_split(n) for n in added + removed}
    if len(servers) == 1 and None not in servers:
        detail += f" (MCP server '{clean(next(iter(servers)))}' {'connected' if added and not removed else 'changed'})"
    label = cn[first] if first < len(cn) else (pn[first] if first < len(pn) else "")
    return {"cause": "tools_changed", "section": "tools", "index": first, "label": label, "detail": detail}


def _prefix_changes(prev: Exchange, cur: Exchange) -> tuple[list[dict[str, Any]], str | None]:
    """Every changed section of ``cur``'s prompt prefix vs ``prev``, in prompt-cache order (tools, system,
    messages); only the first changed message is reported. Second item: why the comparison was impossible."""
    pa, ca = _analyze(prev), _analyze(cur)
    if not pa.parsed or not ca.parsed:
        return [], "request body unavailable (truncated or not JSON): the changed part cannot be located"
    changes: list[dict[str, Any]] = []
    t = _diff_tools(pa.tools, ca.tools)
    if t:
        changes.append(t)
    ps, pm = _units(pa)
    cs, cm = _units(ca)
    if ps != cs:
        if ps is None or cs is None:
            detail = "system prompt " + ("added" if ps is None else "removed")
            changes.append({"cause": "system_changed", "section": "system", "index": 0, "char": 0,
                            "label": "system prompt", "detail": detail})
        else:
            pos, x, y, vol = _snippet(ps, cs)
            changes.append({"cause": "system_changed", "section": "system", "index": 0, "char": pos,
                            "label": "system prompt",
                            "detail": f"system prompt changed at char {pos:,}: '{x}' -> '{y}'"
                                      + (_VOLATILE_HINT if vol else "")})
    for a, b in zip(pm, cm):
        if a.text == b.text and a.role == b.role:
            continue
        pos, x, y, vol = _snippet(a.text, b.text)
        where = f"at char {pos:,}: '{x}' -> '{y}'" + (_VOLATILE_HINT if vol else "")
        if a.role != b.role:
            where = f"role {clean(a.role)} -> {clean(b.role)}"
        if a.section == "system" or b.section == "system":
            changes.append({"cause": "system_changed", "section": "system", "index": a.index, "char": pos,
                            "label": f"message {a.index}",
                            "detail": f"system prompt changed (message {a.index}) {where}"})
        else:
            changes.append({"cause": "history_edited", "section": "messages", "index": a.index, "char": pos,
                            "label": f"message {a.index}",
                            "detail": f"a message earlier in history was edited (message {a.index}, "
                                      f"{clean(a.role)}) {where}"})
        break
    else:
        if len(cm) < len(pm):
            changes.append({"cause": "history_truncated", "section": "messages", "index": len(cm),
                            "label": f"message {len(cm)}",
                            "detail": f"conversation history was shortened ({len(pm)} -> {len(cm)} messages; "
                                      f"compaction or a dropped turn)"})
    return changes, None


def _ttl_s(prev: Exchange, cur: Exchange) -> float:
    body = prev.request_body if isinstance(prev.request_body, str) else ""
    if _anthropic_usage(cur):
        return ANTHROPIC_TTL_1H_S if _TTL_1H_RE.search(body) else ANTHROPIC_TTL_S
    if str(cur.style) == "openai" or str(cur.provider) == "openai":
        return OPENAI_TTL_24H_S if _RETENTION_24H_RE.search(body) else OPENAI_TTL_S
    return OTHER_TTL_S


def _fmt_gap(s: float) -> str:
    s = max(0.0, float(s))
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def _miss_cost(cur: Exchange, lost: int, overrides: dict | None) -> float:
    """Extra dollars paid because ``lost`` tokens were written / read at full price instead of a cache read."""
    rates = _safe_price(cur.model, overrides)
    if not rates or lost <= 0:
        return 0.0
    _, _, cw = _cache_counts(cur)
    if _anthropic_usage(cur):
        w = min(lost, cw)
        usd = w * (rates["cache_write"] - rates["cache_read"]) + (lost - w) * (rates["input"] - rates["cache_read"])
    else:
        usd = lost * (rates["input"] - rates["cache_read"])
    return round(max(usd, 0.0) / 1_000_000 * _multiplier(cur), 10)


def explain_cache(prev: Exchange | None, cur: Exchange | None, overrides: dict | None = None) -> dict[str, Any] | None:
    """Explain how ``cur`` used the prompt cache compared with ``prev`` (the previous request of the same
    session/model). Returns None when ``cur`` has no reported usage or ``prev`` is missing / later.

    ``verdict`` is "hit", "partial", "miss" or "no_cache" (caching not in use). For a (partial) miss, ``cause``
    is one of "model_changed", "ttl_expired", "tools_changed", "tool_order_changed", "system_changed",
    "history_edited", "history_truncated" or "unknown"; ``detail`` names it precisely (e.g. ``system prompt
    changed at char 1,204: '…10:31…' -> '…10:32…'``) and ``lost_usd`` prices the tokens that were re-written or
    re-read at full price instead of a cache read. ``changes`` lists every changed prefix section."""
    try:
        return _explain(prev, cur, overrides)
    except Exception:  # pragma: no cover - defensive
        return None


def _explain(prev: Exchange | None, cur: Exchange | None, overrides: dict | None) -> dict[str, Any] | None:
    if prev is None or cur is None:
        return None
    cur_prompt = _reported_prompt(cur)
    if not cur_prompt:
        return None
    gap = float(cur.ts or 0) - float(prev.ts or 0)
    if gap < 0:
        return None
    anth = _anthropic_usage(cur)
    _, cr, cw = _cache_counts(cur)
    _, pcr, pcw = _cache_counts(prev)
    prev_prompt = _reported_prompt(prev) or 0
    if anth:
        expected = pcr + pcw                        # everything prev had in the cache
    else:
        expected = prev_prompt if prev_prompt >= MIN_AUTO_CACHE_TOKENS else 0
    shared = _shared_tokens(prev, cur)
    expected = min(expected, cur_prompt, shared if shared is not None else cur_prompt)
    lost = max(0, expected - cr)
    ttl = _ttl_s(prev, cur)
    res: dict[str, Any] = {
        "id": cur.id, "prev_id": prev.id, "ts": cur.ts, "gap_s": round(gap, 3), "model": cur.model,
        "prev_model": prev.model, "prompt_tokens": cur_prompt, "cache_read_tokens": cr, "cache_write_tokens": cw,
        "expected_cached_tokens": expected, "lost_tokens": 0, "lost_usd": 0.0, "ttl_s": ttl,
        "verdict": "hit", "cause": None, "detail": "", "where": None, "changes": [], "notes": [],
    }
    if expected <= 0:
        res["verdict"] = "no_cache"
        if anth:
            res["detail"] = ("caching not in use: the previous request neither read nor wrote the cache "
                             "(no cache_control breakpoint, or the prefix is below the minimum cacheable size)")
        else:
            res["detail"] = (f"caching not in use: the previous prompt ({fmt_tokens(prev_prompt)} tokens) is below "
                             f"the {MIN_AUTO_CACHE_TOKENS:,}-token automatic caching minimum")
        return res
    if lost < MIN_LOST_TOKENS or cr >= HIT_RATIO * expected:
        res["detail"] = f"cache hit: read {fmt_tokens(cr)} of ~{fmt_tokens(expected)} cacheable tokens"
        return res
    res["verdict"] = "partial" if cr >= MIN_LOST_TOKENS else "miss"
    res["lost_tokens"] = lost
    res["lost_usd"] = _miss_cost(cur, lost, overrides)
    # the (costlier) structural diff only runs for misses
    changes, why_not = _prefix_changes(prev, cur)
    if why_not:
        res["notes"].append(why_not)
    model_changed = normalize_model(prev.model or "") != normalize_model(cur.model or "")
    if model_changed:
        changes.insert(0, {"cause": "model_changed", "section": "model", "index": 0, "label": "model",
                           "detail": f"model changed: {clean(prev.model)} -> {clean(cur.model)} "
                                     f"(each model has its own cache)"})
    res["changes"] = changes
    prefix = [c for c in changes if c["cause"] != "model_changed"]
    if model_changed:
        first = changes[0]
    elif gap > ttl:
        first = {"cause": "ttl_expired", "section": "ttl", "index": 0, "label": "ttl",
                 "detail": f"cache expired: {_fmt_gap(gap)} since the previous request (cache TTL ~{_fmt_gap(ttl)})"}
        if prefix:
            first["detail"] += f"; the prefix also changed: {prefix[0]['detail']}"
    elif prefix:
        first = prefix[0]
    else:
        reason = ("prefix unchanged and within the cache TTL; possible causes: concurrent requests (cache not "
                  "written yet), a moved or missing cache breakpoint, or provider-side eviction")
        if not anth:
            reason = ("prefix unchanged and within the cache TTL; automatic prompt caching is best-effort "
                      "(requests routed to a machine without the prefix, or evicted early)")
        if why_not:
            reason = why_not
        first = {"cause": "unknown", "section": None, "index": None, "label": None, "detail": reason}
    res["cause"] = first["cause"]
    res["detail"] = first["detail"]
    res["where"] = {k: first.get(k) for k in ("section", "index", "label", "char") if first.get(k) is not None} \
        or None
    return res


_SENTENCE_END = re.compile(r"[.!?;:]\s+")
_MAX_CHUNK = 2048
_CHUNK_MEMO: dict[tuple[str, str, int, int], list[tuple[int, int]]] = {}
_CHUNK_MEMO_MAX = 20_000


def _chunks(text: str) -> list[str]:
    """Lines, with long lines cut after sentence punctuation (content-defined, so an edit only changes nearby
    chunks) and anything still longer than ``_MAX_CHUNK`` cut into fixed slices."""
    out: list[str] = []
    for line in text.splitlines(True):
        if len(line) <= 400:
            out.append(line)
            continue
        start = 0
        for m in _SENTENCE_END.finditer(line):
            out.append(line[start:m.start() + 1])
            start = m.end()
        out.append(line[start:])
    final: list[str] = []
    for piece in out:
        if len(piece) <= _MAX_CHUNK:
            final.append(piece)
        else:
            final.extend(piece[k:k + _MAX_CHUNK] for k in range(0, len(piece), _MAX_CHUNK))
    return final or [text]


def _chunk_hashes(kind: str, role: str, text: str) -> list[tuple[int, int]]:
    """(hash, chars) per chunk; memoized because a conversation's history repeats in every later request."""
    key = (kind, role, hash(text), len(text))
    hit = _CHUNK_MEMO.get(key)
    if hit is None:
        hit = [(hash((kind, role, c)), len(c)) for c in _chunks(text)]
        if len(_CHUNK_MEMO) >= _CHUNK_MEMO_MAX:
            _CHUNK_MEMO.clear()
        _CHUNK_MEMO[key] = hit
    return hit


@dataclass
class _FP:
    """Prompt fingerprint: units in cache order as (hash, estimated tokens, kind) - one per tool ("t"), per
    system-prompt chunk ("s") and per message chunk ("m") - plus derived lookups."""
    units: list[tuple[int, float, str]]
    hashes: set[int]
    head: float                       # tokens of the tools + top-level system units
    body: list[tuple[int, float]]     # the system + message units
    body_total: float


def _fingerprint(ex: Exchange) -> _FP:
    """Fingerprint of a request's prompt, cached on the (memoized) analysis."""
    an = _analyze(ex)
    if an.fp is None:
        an.fp = _make_fingerprint(an)
    return an.fp


def _make_fingerprint(an: _Analysis) -> _FP:
    units: list[tuple[int, float, str]] = []
    if an.parsed:
        units = [(hash(("t", t.text)), t.raw, "t") for t in an.tools]
        system, msgs = _units(an)
        if system:
            units.extend((h, n / 4.0, "s") for h, n in _chunk_hashes("s", "", system))
        for i, u in enumerate(msgs):
            weight = an.msg_tokens[i] if i < len(an.msg_tokens) else _text_tokens(u.text)
            size = len(u.text) or 1
            kind = "s" if u.section == "system" else "m"
            units.extend((h, weight * n / size, kind) for h, n in _chunk_hashes(kind, u.role, u.text))
    body = [(h, w) for h, w, k in units if k != "t"]
    return _FP(units=units, hashes={h for h, _, _ in units}, head=sum(w for _, w, k in units if k in ("t", "s")),
               body=body, body_total=sum(w for _, w in body))


def _shared_tokens(prev: Exchange, cur: Exchange) -> int | None:
    """Tokens of ``cur``'s prompt whose content also appeared in ``prev`` (scaled to cur's reported size): what
    could have come from the cache had nothing changed. None when a body is unreadable."""
    pf, cf = _fingerprint(prev), _fingerprint(cur)
    if not pf.units or not cf.units:
        return None
    est_total = sum(_analyze(cur).raw.values())
    reported = _reported_prompt(cur) or 0
    scale = reported / est_total if est_total > 0 and reported else 1.0
    return int(round(sum(w for h, w, _ in cf.units if h in pf.hashes) * scale))


def _relatedness(cand: _FP, cur: _FP) -> tuple[bool, float, float]:
    """(same thread?, tokens of the common leading units, share of cur's system+message tokens seen in cand).

    Same thread: most of cur's system prompt and messages already appeared in ``cand`` (a later turn, or the same
    system prompt with a new question), or cur repeats cand's whole tools + system prefix. Sub-agents and side
    calls that share only some tool definitions are different threads."""
    lead = 0.0
    for x, y in zip(cand.units, cur.units):
        if x[0] != y[0]:
            break
        lead += y[1]
    have = cand.hashes
    overlap = (sum(w for h, w in cur.body if h in have) / cur.body_total) if cur.body_total > 0 else 0.0
    related = overlap >= 0.5 or (cand.head > 0 and lead >= cand.head - 1e-9)
    return related, lead, overlap


def cache_report(exchanges: Iterable[Exchange], overrides: dict | None = None) -> dict[str, Any]:
    """Walk billable requests in time order, pair each with its predecessor in the same session and explain
    cache misses.

    Sessions are the ``session`` tag; untagged requests group by provider + model, and a gap over
    ``SESSION_GAP_S`` starts a new session. Within a session the predecessor is the recent request of the same
    thread sharing the longest prompt prefix, so interleaved sub-agent / title calls do not look like misses; a
    request unrelated to all of them (see ``_relatedness``) starts a new thread and is not compared.

    Returns ``{"pairs", "hits", "partial", "no_cache", "misses": [explain_cache dicts + "session"], "total_lost_usd",
    "total_lost_tokens", "by_cause": {cause: {"count", "lost_usd"}}}`` with misses sorted by cost."""
    exs = sorted((e for e in exchanges if _billable(e) and _reported_prompt(e)),
                 key=lambda e: (float(e.ts or 0)))
    groups: dict[tuple[str, ...], list[tuple[Exchange, _FP]]] = {}
    out: dict[str, Any] = {"pairs": 0, "hits": 0, "partial": 0, "no_cache": 0, "misses": [], "total_lost_usd": 0.0,
                           "total_lost_tokens": 0, "by_cause": {}}
    for ex in exs:
        try:
            sess = str((ex.tags or {}).get("session") or "")
            key = ("session", sess) if sess else ("auto", str(ex.provider), str(ex.model))
            hist = groups.setdefault(key, [])
            if not sess and hist and float(ex.ts or 0) - float(hist[-1][0].ts or 0) > SESSION_GAP_S:
                hist.clear()
            fp = _fingerprint(ex)
            best: tuple[Any, ...] | None = None
            for n, (cand, cfp) in enumerate(hist):
                related, lead, overlap = _relatedness(cfp, fp)
                if not related:
                    continue
                score = (lead, overlap, cand.model == ex.model, n)
                if best is None or score > best[0]:
                    best = (score, cand)
            hist.append((ex, fp))
            del hist[:-PARENT_CANDIDATES]
            if best is None:
                continue
            r = explain_cache(best[1], ex, overrides)
        except Exception:  # pragma: no cover - defensive
            continue
        if r is None:
            continue
        out["pairs"] += 1
        if r["verdict"] == "hit":
            out["hits"] += 1
            continue
        if r["verdict"] == "no_cache":
            out["no_cache"] += 1
            continue
        if r["verdict"] == "partial":
            out["partial"] += 1
        r["session"] = sess or None
        out["misses"].append(r)
        out["total_lost_usd"] += r["lost_usd"]
        out["total_lost_tokens"] += r["lost_tokens"]
        bc = out["by_cause"].setdefault(r["cause"], {"count": 0, "lost_usd": 0.0})
        bc["count"] += 1
        bc["lost_usd"] = round(bc["lost_usd"] + r["lost_usd"], 10)
    out["misses"].sort(key=lambda r: (-r["lost_usd"], -r["lost_tokens"], r["ts"]))
    out["total_misses"] = len(out["misses"])
    out["total_lost_usd"] = round(out["total_lost_usd"], 10)
    return out


# --------------------------------------------------------------------------------------------
# 4. renderers


def _bar(share: float, width: int = 24) -> str:
    share = min(max(share, 0.0), 1.0)
    n = int(round(share * width))
    if share > 0 and n == 0:
        n = 1
    return "█" * n + "░" * (width - n)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%" if 0 < x < 0.1 else f"{100 * x:.0f}%"


def _verdict_line(r: dict[str, Any] | None, color: bool) -> str:
    p = lambda s, *st: paint(s, *st, color=color)  # noqa: E731
    if r is None:
        return p("cache: no comparison (no reported usage)", "dim")
    v = r["verdict"]
    if v == "hit":
        return p("cache: HIT", "bold", "green") + " " + r["detail"].removeprefix("cache hit: ")
    if v == "no_cache":
        return p("cache: not in use", "bold", "gray") + " " + p(r["detail"].removeprefix("caching not in use: "), "dim")
    label = "MISS" if v == "miss" else "PARTIAL MISS"
    return (p(f"cache: {label}", "bold", "red") + " " + r["detail"] + " "
            + p(f"(~{fmt_tokens(r['lost_tokens'])} tokens, ~{fmt_usd(r['lost_usd'])} lost)", "yellow"))


def render_xray_text(ex: Exchange, color: bool = False, prev: Exchange | None = None, overrides: dict | None = None,
                     top: int = 5) -> str:
    """Compact terminal view of one request: a bar per part, the top tool definitions, MCP servers and, when
    ``prev`` is given, the cache verdict against it."""
    p = lambda s, *st: paint(s, *st, color=color)  # noqa: E731
    bd = breakdown(ex, overrides)
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ex.ts or 0)))
    head = (p("x-ray", "bold", "bright_magenta") + " " + p(str(ex.id)[:8], "dim") + "  "
            + p(clean(f"{ex.provider}/{ex.model or '-'}"), "bold") + "  " + p(when, "gray"))
    total = bd["input_tokens"]
    est = "" if bd["scaled"] else "~"
    detail = ""
    if bd["scaled"] and (bd["cache_read_tokens"] or bd["cache_write_tokens"]):
        detail = (f" ({fmt_tokens(bd['uncached_input_tokens'])} uncached + {fmt_tokens(bd['cache_read_tokens'])}"
                  f" cache read + {fmt_tokens(bd['cache_write_tokens'])} cache write)")
    cost = p(fmt_usd(bd["input_cost_usd"]), "yellow") + ("" if bd["scaled"] else p("  (estimated)", "dim"))
    lines = [head, f"input {p(est + fmt_tokens(total), 'bold')} tokens{detail}  {cost}", ""]
    for part, d in bd["parts"].items():
        lines.append(f"  {PART_LABELS[part]:<10} {p(_bar(d['share']), _PART_HUE[part])} "
                     f"{est + fmt_tokens(d['tokens']):>7} {_pct(d['share']):>6}  {fmt_usd(d['usd']):>9}")
    for note in bd["notes"]:
        lines.append(p(f"  note: {note}", "dim"))
    items = _tool_items(ex, bd)
    if items:
        tool_tok = sum(i["tokens"] for i in items)
        lines += ["", p(f"tool definitions: {len(items)} tools, {est}{fmt_tokens(tool_tok)} tokens", "bold")]
        w = min(44, max(len(clean(i["name"])) for i in items[:top]))
        for it in items[:top]:
            name = clean(it["name"])
            name = name if len(name) <= w else name[:w - 1] + "…"
            lines.append(f"  {name:<{w}} {est + fmt_tokens(it['tokens']):>7} {_pct(it['share']):>6}")
        servers = _group_servers(items)
        if servers:
            lines.append("  MCP: " + " · ".join(
                f"{clean(s['server'])} {s['tools']} tool{'s' if s['tools'] != 1 else ''} {est}{fmt_tokens(s['tokens'])}"
                for s in servers[:6]))
    if prev is not None:
        lines += ["", _verdict_line(explain_cache(prev, ex, overrides), color)]
    return "\n".join(lines)


def summary_json(exchanges: Iterable[Exchange], overrides: dict | None = None, top: int = 10) -> dict[str, Any]:
    """Period summary as a JSON-safe dict: request size, share per part, tool bloat and cache misses."""
    exs = list(exchanges)
    acc = _Acc(overrides)
    for ex in exs:
        try:
            acc.add(ex)
        except Exception:  # pragma: no cover - defensive
            continue
    n = acc.analyzed
    psum = sum(d["tokens"] for d in acc.parts.values())
    parts = {}
    for part, d in acc.parts.items():
        if part == "other" and not d["tokens"]:
            continue
        parts[part] = {"tokens": d["tokens"], "avg_tokens": int(round(d["tokens"] / n)) if n else 0,
                       "share": (d["tokens"] / psum) if psum else 0.0, "usd": round(d["usd"], 10)}
    cache = cache_report(exs, overrides)
    cache["misses"] = cache["misses"][:max(top, 0) * 5]
    return {
        "requests": acc.requests, "analyzed": n, "unparsed": acc.unparsed, "truncated": acc.truncated,
        "scaled": acc.scaled, "total_input_tokens": acc.input_tokens,
        "avg_input_tokens": int(round(acc.input_tokens / n)) if n else 0, "input_usd": round(acc.input_usd, 10),
        "parts": parts, "tool_bloat": acc.tool_summary(top), "cache": cache,
    }


def render_summary_text(exchanges: Iterable[Exchange], color: bool = False, overrides: dict | None = None,
                        top: int = 5) -> str:
    """Period summary for the terminal: average request size, share per part, tool bloat, top cache misses."""
    p = lambda s, *st: paint(s, *st, color=color)  # noqa: E731
    data = summary_json(exchanges, overrides, top=max(top, 10))
    n = data["analyzed"]
    if not n:
        return p("x-ray: no billable requests with a readable body in this period.", "dim")
    lines = [p("x-ray", "bold", "bright_magenta") + f"  {n} request{'s' if n != 1 else ''} · avg input "
             + p(fmt_tokens(data["avg_input_tokens"]), "bold") + " tokens · input cost "
             + p(fmt_usd(data["input_usd"]), "yellow")]
    if data["unparsed"]:
        lines.append(p(f"  ({data['unparsed']} request(s) with an unreadable/truncated body skipped)", "dim"))
    lines += ["", p("where the input goes", "bold")]
    for part, d in data["parts"].items():
        lines.append(f"  {PART_LABELS[part]:<10} {p(_bar(d['share']), _PART_HUE[part])} {_pct(d['share']):>6}  "
                     f"avg {fmt_tokens(d['avg_tokens']):>6}  {fmt_usd(d['usd']):>9}")
    tb = data["tool_bloat"]
    if tb["total_tool_tokens"]:
        lines += ["", p("tool definitions", "bold")]
        lines += [f"  {h}" for h in tb["headlines"]]
        if tb["top_tools"]:
            largest = sorted(tb["top_tools"], key=lambda t: -t["avg_tokens"])[:top]
            lines.append("  largest: " + ", ".join(f"{clean(t['name'])} ~{fmt_tokens(t['avg_tokens'])}"
                                                   for t in largest))
    c = data["cache"]
    lines += ["", p("prompt cache", "bold")]
    if not c["pairs"]:
        lines.append(p("  no consecutive requests with reported usage to compare", "dim"))
    else:
        nm = c["total_misses"]
        misses = f"{nm} miss{'es' if nm != 1 else ''}"
        lines.append(f"  {c['pairs']} follow-up request{'s' if c['pairs'] != 1 else ''}: {c['hits']} hit"
                     f"{'s' if c['hits'] != 1 else ''}, " + p(misses, "red" if nm else "green")
                     + (f", {c['no_cache']} without caching" if c["no_cache"] else "")
                     + (p(f" · ~{fmt_usd(c['total_lost_usd'])} lost", "yellow") if nm else ""))
        for cause, d in sorted(c["by_cause"].items(), key=lambda kv: -kv[1]["lost_usd"]):
            lines.append(f"    {cause:<19} {d['count']:>4}x  {fmt_usd(d['lost_usd']):>9}")
        for r in c["misses"][:top]:
            when = time.strftime("%H:%M:%S", time.localtime(float(r["ts"] or 0)))
            cost = fmt_usd(r["lost_usd"])
            lines.append(f"  {p(f'{cost:>9}', 'yellow')}  {p(when, 'gray')}  {clean(r['model'] or '-')}  {r['detail']}")
    return "\n".join(lines)
