"""Routing + wire-format knowledge for Anthropic / OpenAI / OpenAI-compatible APIs.

Every function here is defensive: malformed or unexpected input never raises, it yields a
safe default (``None``, ``""``, ``Usage()``...).

Path normalization (openai style only): if the upstream path does not start with a version
segment (``/v1``, ``/v2``, ...), ``/v1`` is prepended -- e.g. ``/chat/completions`` becomes
``/v1/chat/completions`` (users often set ``OPENAI_BASE_URL`` without ``/v1``). The exception is an
explicit-prefix route whose upstream base already ends in a version segment (e.g. a custom upstream
``http://host/v1``); there the path is passed through unchanged. Anthropic-style paths are never
rewritten.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .config import Config
from .models import Route, Usage

__all__ = [
    "HOP_BY_HOP",
    "error_body",
    "estimate_tokens",
    "estimate_usage",
    "extract_text",
    "extract_usage",
    "forward_headers",
    "is_stream_request",
    "parse_json",
    "parse_sse",
    "request_model",
    "resolve_route",
    "response_model",
    "tags_from_headers",
]

_VERSION_SEG = re.compile(r"^/v\d+(?:[a-z0-9]*)?(?:[/?#]|$)")
_BASE_ENDS_WITH_VERSION = re.compile(r"/v\d+[a-z0-9]*$")

_DEFAULT_BASES = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
}


# --------------------------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------------------------
def _header_names(headers: Mapping[str, str] | None) -> set[str]:
    try:
        return {str(k).lower() for k in (headers or {}).keys()}
    except Exception:
        return set()


def _upstream(cfg: Config, name: str) -> tuple[str, str]:
    """(base, style) for a named upstream, with sensible fallbacks."""
    spec = {}
    try:
        spec = (cfg.upstreams or {}).get(name) or {}
    except Exception:
        spec = {}
    base = str(spec.get("base") or _DEFAULT_BASES.get(name, "")).rstrip("/")
    style = str(spec.get("style") or ("anthropic" if name == "anthropic" else "openai"))
    return base, style


def _normalize_openai_path(path: str, base: str) -> str:
    if _VERSION_SEG.match(path) or _BASE_ENDS_WITH_VERSION.search(base):
        return path
    return "/v1" + path


def resolve_route(path: str, headers: Mapping[str, str], cfg: Config) -> Route:
    """Decide which upstream a proxied request goes to (see module docstring for path rules)."""
    path = path or "/"
    if not path.startswith("/"):
        path = "/" + path
    pure, sep, query = path.partition("?")

    # 1. explicit prefix: /<name>/rest
    parts = pure.split("/", 2)  # ["", name, rest]
    name = parts[1] if len(parts) > 1 else ""
    upstreams = getattr(cfg, "upstreams", None) or {}
    if name and name in upstreams:
        base, style = _upstream(cfg, name)
        rest = "/" + (parts[2] if len(parts) > 2 else "")
        up_path = rest + (sep + query if sep else "")
        if style == "openai":
            up_path = _normalize_openai_path(up_path, base)
        return Route(provider=name, upstream_base=base, upstream_path=up_path, style=style)

    # 2./3. anthropic by path or headers
    hdrs = _header_names(headers)
    if (
        pure.startswith("/v1/messages")
        or pure.startswith("/v1/complete")
        or "anthropic-version" in hdrs
        or "x-api-key" in hdrs
    ):
        base, style = _upstream(cfg, "anthropic")
        return Route(provider="anthropic", upstream_base=base, upstream_path=path, style=style)

    # 4. openai
    base, style = _upstream(cfg, "openai")
    up_path = _normalize_openai_path(path, base) if style == "openai" else path
    return Route(provider="openai", upstream_base=base, upstream_path=up_path, style=style)


# --------------------------------------------------------------------------------------------
# request helpers
# --------------------------------------------------------------------------------------------
def parse_json(body: bytes) -> dict | None:
    """Parse a JSON object body; None if empty, invalid, or not an object."""
    if not body:
        return None
    try:
        if isinstance(body, (bytes, bytearray, memoryview)):
            text = bytes(body).decode("utf-8-sig")
        else:
            text = str(body)
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def request_model(req: dict | None) -> str | None:
    """The requested model name, if any."""
    if not isinstance(req, dict):
        return None
    m = req.get("model")
    return m if isinstance(m, str) and m else None


def is_stream_request(req: dict | None) -> bool:
    """True when the request asks for a streamed (SSE) response."""
    return isinstance(req, dict) and req.get("stream") is True


# --------------------------------------------------------------------------------------------
# response parsing
# --------------------------------------------------------------------------------------------
def _int(v: Any) -> int:
    if isinstance(v, bool) or v is None:
        return 0
    if isinstance(v, int):
        return max(0, v)
    if isinstance(v, float):
        return max(0, int(v))
    if isinstance(v, str):
        try:
            return max(0, int(float(v)))
        except ValueError:
            return 0
    return 0


def _dict(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def _usage_from_dict(u: Any) -> Usage:
    """Map any provider's usage object onto Usage (anthropic, openai chat, responses, embeddings)."""
    u = _dict(u)
    inp = u.get("input_tokens")
    if inp is None:
        inp = u.get("prompt_tokens")
    out = u.get("output_tokens")
    if out is None:
        out = u.get("completion_tokens")
    cache_read = u.get("cache_read_input_tokens")
    if cache_read is None:
        cache_read = _dict(u.get("prompt_tokens_details")).get("cached_tokens")
    if cache_read is None:
        cache_read = _dict(u.get("input_tokens_details")).get("cached_tokens")
    return Usage(
        input_tokens=_int(inp),
        output_tokens=_int(out),
        cache_read_tokens=_int(cache_read),
        cache_write_tokens=_int(u.get("cache_creation_input_tokens")),
    )


def extract_usage(style: str, resp: dict) -> Usage:
    """Token usage from a non-streaming response JSON. Missing fields -> 0."""
    if not isinstance(resp, dict):
        return Usage()
    usage = resp.get("usage")
    if usage is None and isinstance(resp.get("response"), dict):  # wrapped responses-API object
        usage = resp["response"].get("usage")
    return _usage_from_dict(usage)


def _anthropic_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            out.append(block["text"])
    return "".join(out)


def _openai_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "output_text") and isinstance(part.get("text"), str):
                out.append(part["text"])
            elif isinstance(part, str):
                out.append(part)
        return "".join(out)
    return ""


def _openai_chat_text(resp: dict) -> str:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    msg = first.get("message")
    if isinstance(msg, dict):
        return _openai_message_content(msg.get("content"))
    if isinstance(first.get("text"), str):  # legacy /v1/completions
        return first["text"]
    return ""


def _responses_text(resp: dict) -> str:
    top = resp.get("output_text")
    if isinstance(top, str) and top:
        return top
    output = resp.get("output")
    if not isinstance(output, list):
        return ""
    out = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                out.append(part["text"])
    return "".join(out)


def extract_text(style: str, resp: dict) -> str:
    """Assistant text of a non-streaming response ("" for embeddings, errors, tool-only replies)."""
    if not isinstance(resp, dict):
        return ""
    try:
        if "choices" in resp:
            return _openai_chat_text(resp)
        if "output" in resp or "output_text" in resp:
            return _responses_text(resp)
        if "content" in resp:
            return _anthropic_text(resp.get("content"))
        if isinstance(resp.get("response"), dict):
            return _responses_text(resp["response"])
    except Exception:
        return ""
    return ""


def response_model(style: str, resp: dict) -> str | None:
    """Model name reported by the upstream response."""
    if not isinstance(resp, dict):
        return None
    m = resp.get("model")
    if not (isinstance(m, str) and m) and isinstance(resp.get("response"), dict):
        m = resp["response"].get("model")
    return m if isinstance(m, str) and m else None


# --------------------------------------------------------------------------------------------
# SSE
# --------------------------------------------------------------------------------------------
def _sse_events(raw: str) -> list[tuple[str, str]]:
    """Split an SSE transcript into (event_name, data) pairs. Multi-line data joined with \\n."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    events: list[tuple[str, str]] = []
    name = ""
    data: list[str] = []
    for line in text.split("\n") + [""]:
        if line == "":
            if data:
                events.append((name, "\n".join(data)))
            name, data = "", []
            continue
        if line.startswith(":"):
            continue
        field, colon, value = line.partition(":")
        if not colon:
            field, value = line, ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "event":
            name = value.strip()
        elif field == "data":
            data.append(value)
    return events


def _merge_usage(dst: Usage, src: dict, override_output: bool = True) -> None:
    """Merge a (possibly partial) usage dict into dst; only fields present and > 0 overwrite."""
    u = _usage_from_dict(src)
    if u.input_tokens:
        dst.input_tokens = u.input_tokens
    if u.output_tokens and override_output:
        dst.output_tokens = u.output_tokens
    if u.cache_read_tokens:
        dst.cache_read_tokens = u.cache_read_tokens
    if u.cache_write_tokens:
        dst.cache_write_tokens = u.cache_write_tokens


def parse_sse(style: str, raw: str) -> tuple[Usage, str, str | None]:
    """Parse a full raw SSE transcript into (usage, assembled_text, model).

    Handles anthropic messages streams, openai chat-completion chunks (incl. final usage chunk),
    legacy completions chunks, and the openai Responses API event stream. Bad lines are skipped.
    """
    usage = Usage()
    text_parts: list[str] = []
    model: str | None = None
    completed_text = ""
    if not isinstance(raw, str):
        try:
            raw = bytes(raw).decode("utf-8", "replace")
        except Exception:
            return usage, "", None

    for name, data in _sse_events(raw):
        payload = data.strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        etype = obj.get("type") if isinstance(obj.get("type"), str) else name

        try:
            # ---- anthropic ----
            if etype == "message_start":
                msg = _dict(obj.get("message"))
                if not model and isinstance(msg.get("model"), str) and msg["model"]:
                    model = msg["model"]
                _merge_usage(usage, _dict(msg.get("usage")))
                continue
            if etype == "message_delta":
                _merge_usage(usage, _dict(obj.get("usage")))
                continue
            if etype == "content_block_delta":
                delta = _dict(obj.get("delta"))
                if delta.get("type", "text_delta") == "text_delta" and isinstance(delta.get("text"), str):
                    text_parts.append(delta["text"])
                continue
            if etype in ("content_block_start", "content_block_stop", "message_stop", "ping", "error"):
                continue

            # ---- openai responses API ----
            if etype == "response.output_text.delta":
                if isinstance(obj.get("delta"), str):
                    text_parts.append(obj["delta"])
                continue
            if etype in ("response.created", "response.in_progress", "response.completed",
                         "response.incomplete", "response.failed"):
                resp = _dict(obj.get("response"))
                if isinstance(resp.get("model"), str) and resp["model"]:
                    model = model or resp["model"]
                if isinstance(resp.get("usage"), dict):
                    _merge_usage(usage, resp["usage"])
                if etype != "response.created" and etype != "response.in_progress":
                    completed_text = _responses_text(resp) or completed_text
                continue
            if isinstance(etype, str) and etype.startswith("response."):
                continue

            # ---- openai chat / legacy completions chunks ----
            if not model and isinstance(obj.get("model"), str) and obj["model"]:
                model = obj["model"]
            choices = obj.get("choices")
            if isinstance(choices, list):
                for ch in choices:
                    if not isinstance(ch, dict) or _int(ch.get("index", 0)) != 0:
                        continue
                    delta = ch.get("delta")
                    if isinstance(delta, dict):
                        piece = _openai_message_content(delta.get("content"))
                        if piece:
                            text_parts.append(piece)
                    elif isinstance(ch.get("text"), str):
                        text_parts.append(ch["text"])
            if isinstance(obj.get("usage"), dict):
                _merge_usage(usage, obj["usage"])
        except Exception:
            continue

    text = "".join(text_parts) or completed_text
    return usage, text, model


# --------------------------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Crude chars/4 token estimate: max(1, len//4) for non-empty text, else 0."""
    if not text or not isinstance(text, str):
        return 0
    return max(1, len(text) // 4)


_B64_RE = re.compile(r"[A-Za-z0-9+/=_-]+")


def _strings(obj: Any, out: list[str], depth: int = 0) -> None:
    if depth > 64:
        return
    if isinstance(obj, str):
        # skip inline binary payloads (base64 images/audio, data: URLs) — they aren't text tokens
        if obj.startswith("data:") or (len(obj) > 512 and _B64_RE.fullmatch(obj[:512])):
            return
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _strings(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _strings(v, out, depth + 1)


def estimate_usage(req: dict | None, output_text: str) -> Usage:
    """Estimated usage: input from every string value in the request (keys excluded), output from text."""
    strings: list[str] = []
    if isinstance(req, (dict, list)):
        _strings(req, strings)
    return Usage(
        input_tokens=estimate_tokens("\n".join(strings)),
        output_tokens=estimate_tokens(output_text if isinstance(output_text, str) else ""),
        estimated=True,
    )


# --------------------------------------------------------------------------------------------
# errors + headers
# --------------------------------------------------------------------------------------------
def error_body(style: str, status: int, message: str, err_type: str = "llmtap_error") -> bytes:
    """Provider-shaped JSON error body so SDKs surface llmtap errors nicely."""
    if style == "anthropic":
        obj: dict[str, Any] = {"type": "error", "error": {"type": err_type, "message": message}}
    else:
        obj = {"error": {"message": message, "type": err_type, "code": err_type}}
    return json.dumps(obj).encode("utf-8")


HOP_BY_HOP: frozenset[str] = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding",
})


def _items(headers: Mapping[str, str] | None) -> list[tuple[str, str]]:
    try:
        return [(str(k), str(v)) for k, v in (headers or {}).items()]
    except Exception:
        return []


def forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Headers to send upstream: drop hop-by-hop (+ those named in Connection) and x-llmtap-*."""
    items = _items(headers)
    extra: set[str] = set()
    for k, v in items:
        if k.lower() == "connection":
            extra |= {t.strip().lower() for t in v.split(",") if t.strip()}
    out: dict[str, str] = {}
    for k, v in items:
        lk = k.lower()
        if lk in HOP_BY_HOP or lk in extra or lk.startswith("x-llmtap-"):
            continue
        out[k] = v
    out["accept-encoding"] = "identity"
    return out


def tags_from_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """``X-LLMTap-Tag-Foo: bar`` -> {"foo": "bar"}; ``X-LLMTap-Session: s`` -> {"session": "s"}."""
    tags: dict[str, str] = {}
    for k, v in _items(headers):
        lk = k.lower()
        if lk == "x-llmtap-session":
            tags["session"] = v.strip()
        elif lk.startswith("x-llmtap-tag-"):
            name = lk[len("x-llmtap-tag-"):].strip()
            if name:
                tags[name] = v.strip()
    return tags
