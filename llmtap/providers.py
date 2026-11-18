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

import base64
import binascii
import json
import os
import re
import struct
import zlib
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote

from .config import Config
from .models import Route, Usage

__all__ = [
    "HOP_BY_HOP",
    "body_error",
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
    "sse_error",
    "tags_from_headers",
]

_SIGNED_HEADERS_RE = re.compile(
    r"(?:^|,\s*)SignedHeaders=([^,\s]+)",
    re.IGNORECASE,
)

# Amazon Bedrock host and credential regexes, for routing and model extraction.
_AWS_REGION = r"[a-z]{2}(?:-[a-z]+){1,2}-\d{1,2}"       # us-east-1, us-gov-west-1, ap-southeast-2, cn-north-1
_BEDROCK_HOST_RE = re.compile(
    r"^bedrock-runtime(?:-fips)?"
    rf"\.({_AWS_REGION})"
    r"\.(amazonaws\.com(?:\.cn)?|api\.aws)$",
    re.IGNORECASE,
)

_BEDROCK_CREDENTIAL_RE = re.compile(
    r"Credential=[^/\s,]+"
    r"/\d{8}"
    rf"/({_AWS_REGION})"
    r"/bedrock"
    r"/aws4_request",
    re.IGNORECASE,
)

# The Bedrock Runtime generation calls: InvokeModel, InvokeModelWithResponseStream, Converse, ConverseStream.
_BEDROCK_RUNTIME_PATH = re.compile(
    r"^/model/([^/?#]+)/(invoke|invoke-with-response-stream|converse|converse-stream)$")
_BEDROCK_PATH_PREFIXES = ("/model/", "/async-invoke", "/guardrail/")
BEDROCK_DEFAULT_BASE = "https://bedrock-runtime.{region}.amazonaws.com"

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

def _header_value(headers: Mapping[str, str] | None, name: str) -> str:
    """Return a header value without assuming a particular name casing."""
    wanted = name.lower()
    try:
        for key, value in (headers or {}).items():
            if str(key).lower() == wanted:
                return str(value).strip()
    except Exception:
        pass
    return ""

def _aws_signed_headers(headers: Mapping[str, str]) -> set[str]:
    authorization = _header_value(headers, "authorization")
    match = _SIGNED_HEADERS_RE.search(authorization)

    if match is None:
        return set()

    return {
        name.strip().lower()
        for name in match.group(1).split(";")
        if name.strip()
    }

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

def _resolve_bedrock_route(
    path: str,
    headers: Mapping[str, str],
    cfg: Config,
) -> Route | None:
    """Resolve a native AWS Bedrock Runtime request.

    Bedrock requests use SigV4, so the original path and query string must
    be forwarded unchanged.
    """
    host = _header_value(headers, "host")
    authorization = _header_value(headers, "authorization")

    host_match = _BEDROCK_HOST_RE.fullmatch(host)
    credential_match = _BEDROCK_CREDENTIAL_RE.search(authorization)

    # A Bedrock host is definitive (the client addressed AWS). When the request is sent to llmtap itself, a
    # Bedrock SigV4 credential scope on a Bedrock path identifies it, and so does a Bedrock API key (a Bearer
    # token that would otherwise go to OpenAI) on a Bedrock Runtime path. Any other credential on a
    # /model/... path keeps its usual route, so no other provider's key is ever sent to AWS.
    pure_path = path.partition("?")[0]
    signed_runtime_request = credential_match is not None and pure_path.startswith(_BEDROCK_PATH_PREFIXES)
    api_key_request = _is_bedrock_api_key(authorization) and bool(_BEDROCK_RUNTIME_PATH.match(pure_path))
    if host_match is None and not signed_runtime_request and not api_key_request:
        return None

    region = host_match.group(1).lower() if host_match is not None else bedrock_region(headers)
    base = bedrock_base(cfg, region)
    # Keep the exact AWS endpoint variant (FIPS, dual-stack, China) because Host participates in SigV4.
    if host_match is not None and _bedrock_spec(cfg).get("base") in (None, "", BEDROCK_DEFAULT_BASE):
        base = f"https://{host.lower()}"

    return Route(
        provider="bedrock",
        upstream_base=base,
        upstream_path=path,  # Preserve the signed path and query exactly.
        style=str(_bedrock_spec(cfg).get("style") or "bedrock"),
    )


def _is_bedrock_api_key(authorization: str) -> bool:
    """``Authorization: Bearer`` with a Bedrock API key: long-term keys start with ``ABSK``, short-term ones
    with ``bedrock-api-key-``."""
    scheme, _, token = (authorization or "").partition(" ")
    token = token.strip()
    return scheme.lower() == "bearer" and (token.startswith("ABSK") or token.startswith("bedrock-api-key-"))


def _bedrock_spec(cfg: Config | None) -> dict:
    try:
        return dict((getattr(cfg, "upstreams", None) or {}).get("bedrock") or {})
    except Exception:
        return {}


def bedrock_base(cfg: Config | None, region: str) -> str:
    """The Bedrock upstream base for ``region``. Without a region the ``{region}`` placeholder stays in the
    result, and the proxy answers with an error that says how to set one."""
    return _with_region(str(_bedrock_spec(cfg).get("base") or BEDROCK_DEFAULT_BASE).rstrip("/"), region)


def _with_region(template: str, region: str) -> str:
    if not region or "{region}" not in template:
        return template
    if template == BEDROCK_DEFAULT_BASE and region.startswith("cn-"):
        template += ".cn"                                  # China regions live under amazonaws.com.cn
    try:
        return template.format(region=region)
    except (KeyError, IndexError, ValueError):
        return template


def bedrock_region(headers: Mapping[str, str] | None) -> str:
    """The AWS region of a Bedrock request sent to llmtap itself: the SigV4 credential scope, else the region
    inside a short-term Bedrock API key, else ``AWS_REGION`` / ``AWS_DEFAULT_REGION``; "" when unknown."""
    authorization = _header_value(headers, "authorization")
    m = _BEDROCK_CREDENTIAL_RE.search(authorization)
    if m:
        return m.group(1).lower()
    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    if scheme.lower() == "bearer" and token.startswith("bedrock-api-key-"):
        # Short-term keys are a base64-encoded presigned URL that carries X-Amz-Credential=<id>/<date>/<region>/...
        try:
            raw = token[len("bedrock-api-key-"):]
            url = unquote(base64.b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8", "replace"))
            m = re.search(rf"X-Amz-Credential=[^/&\s]+/\d{{8}}/({_AWS_REGION})/bedrock/", url, re.IGNORECASE)
            if m:
                return m.group(1).lower()
        except (binascii.Error, ValueError):
            pass
    for var in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        v = (os.environ.get(var) or "").strip().lower()
        if re.fullmatch(_AWS_REGION, v):
            return v
    return ""


def is_sigv4(headers: Mapping[str, str] | None) -> bool:
    """True when the request is signed with AWS Signature Version 4 (the body and signed headers are frozen)."""
    return _header_value(headers, "authorization").upper().startswith("AWS4-HMAC-SHA256")


def bedrock_route_problem(route: Route, client_path: str, headers: Mapping[str, str] | None) -> str | None:
    """Why a Bedrock request can't be forwarded as routed (an error message), or None when it can."""
    if route.provider != "bedrock" and route.style != "bedrock":
        return None
    if "{region}" in route.upstream_base:
        return ("llmtap can't tell which AWS region this Bedrock call is for. Start llmtap with AWS_REGION set "
                "(e.g. AWS_REGION=us-east-1), or set [upstreams.bedrock] base in llmtap.toml.")
    pure = (client_path or "").partition("?")[0]
    if is_sigv4(headers) and pure != route.upstream_path.partition("?")[0]:
        return ("This request is signed with AWS SigV4, which covers the URL path, so llmtap can't remove the "
                "/bedrock/ prefix without breaking the signature. Point the AWS SDK at llmtap's root URL "
                "(endpoint_url=http://127.0.0.1:4141) instead.")
    return None

def upstream_path_for(provider: str, path: str, cfg: Config | None = None) -> str:
    """The upstream path the proxy sent for a request it recorded as (``provider``, client ``path``): strips a
    ``/<provider>/`` routing prefix and applies OpenAI's ``/v1`` normalization, like :func:`resolve_route`.
    Used to key rows that were recorded before the request key was stored."""
    cfg = cfg or Config()
    path = path or "/"
    if not path.startswith("/"):
        path = "/" + path
    upstreams = getattr(cfg, "upstreams", None) or {}
    if provider in upstreams and (path == f"/{provider}" or path.startswith(f"/{provider}/")
                                  or path.startswith(f"/{provider}?")):
        pure, sep, query = path.partition("?")
        rest = pure[len(provider) + 1:] or "/"
        base, style = _upstream(cfg, provider)
        up = rest + (sep + query if sep else "")
        return _normalize_openai_path(up, base) if style == "openai" else up
    if provider == "openai":
        base, style = _upstream(cfg, "openai")
        return _normalize_openai_path(path, base) if style == "openai" else path
    return path


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
    # 1b. native AWS Bedrock Runtime request (unless an explicit prefix names an upstream)
    if not (name and name in upstreams):
        bedrock = _resolve_bedrock_route(path, headers, cfg)
        if bedrock is not None:
            return bedrock
    if name and name in upstreams:
        base, style = _upstream(cfg, name)
        if "{region}" in base:              # e.g. the built-in bedrock upstream
            base = _with_region(base, bedrock_region(headers))
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
        or "x-api-key" in hdrs
        # any anthropic-* header (anthropic-version, anthropic-beta, anthropic-dangerous-direct-browser-access):
        # Claude Code with an OAuth bearer token sends no x-api-key / anthropic-version on e.g. /api/oauth/*
        or any(h.startswith("anthropic-") for h in hdrs)
    ):
        base, style = _upstream(cfg, "anthropic")
        return Route(provider="anthropic", upstream_base=base, upstream_path=path, style=style)

    # Gemini: any Google Generative Language endpoint, a Google API key header, or a `?key=` query
    # parameter. Without the broader checks, e.g. `:embedContent?key=AIza...` fell through to the OpenAI
    # route and the Google key was forwarded to api.openai.com.
    qkeys = {p.split("=", 1)[0].lower() for p in query.split("&") if p} if sep else set()
    if (
        ":generateContent" in pure
        or ":streamGenerateContent" in pure
        or re.match(r"^/v1(alpha|beta)\d*/", pure)
        or re.search(r"/models/[^/]+:[A-Za-z]+$", pure)
        or "x-goog-api-key" in hdrs
        or "key" in qkeys
    ):
        base, style = _upstream(cfg,"gemini")
        return Route(provider="gemini", upstream_base=base, upstream_path=path,style=style)

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


_PATH_MODEL_RE = re.compile(r"/models/([^/:?#]+)(?::[A-Za-z]+)?")


def model_from_path(path: str | None) -> str | None:
    """Model name embedded in a URL path, e.g. /v1beta/models/gemini-2.5-flash:generateContent -> gemini-2.5-flash.
    Gemini carries the model in the URL rather than the request body."""
    if not isinstance(path, str):
        return None
    pure = path.split("?", 1)[0]
    b = _BEDROCK_RUNTIME_PATH.match(pure)
    if b:   # Bedrock: /model/<id or URL-encoded ARN>/converse
        return unquote(b.group(1)) or None
    m = _PATH_MODEL_RE.search(pure)
    return m.group(1) if m else None


def is_bedrock_call(path: str | None) -> bool:
    """True for the four Bedrock Runtime generation calls (invoke / converse, streaming or not)."""
    return isinstance(path, str) and bool(_BEDROCK_RUNTIME_PATH.match(path.split("?", 1)[0]))



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

    def first(*names: str) -> Any:
        for n in names:
            if u.get(n) is not None:
                return u[n]
        return None

    # snake_case: Anthropic / OpenAI; camelCase: Bedrock Converse (and Nova's InvokeModel *Count variants)
    inp = first("input_tokens", "prompt_tokens", "inputTokens")
    out = first("output_tokens", "completion_tokens", "outputTokens")
    cache_read = first("cache_read_input_tokens", "cacheReadInputTokens", "cacheReadInputTokenCount")
    if cache_read is None:
        cache_read = _dict(u.get("prompt_tokens_details")).get("cached_tokens")
    if cache_read is None:
        cache_read = _dict(u.get("input_tokens_details")).get("cached_tokens")
    speed = u.get("speed")
    return Usage(
        input_tokens=_int(inp),
        output_tokens=_int(out),
        cache_read_tokens=_int(cache_read),
        cache_write_tokens=_int(first("cache_creation_input_tokens", "cacheWriteInputTokens",
                                      "cacheWriteInputTokenCount")),
        cache_write_1h_tokens=_int(_dict(u.get("cache_creation")).get("ephemeral_1h_input_tokens")),
        speed=speed if isinstance(speed, str) else "",
        web_search_requests=_int(_dict(u.get("server_tool_use")).get("web_search_requests")),
    )


def _gemini_usage(meta: dict) -> Usage:
    """Gemini usageMetadata -> Usage. promptTokenCount already includes cached tokens; thinking tokens
    (thoughtsTokenCount) are billed as output but reported separately from candidatesTokenCount."""
    return Usage(
        input_tokens=_int(meta.get("promptTokenCount")) + _int(meta.get("toolUsePromptTokenCount")),
        output_tokens=_int(meta.get("candidatesTokenCount")) + _int(meta.get("thoughtsTokenCount")),
        cache_read_tokens=_int(meta.get("cachedContentTokenCount")),
    )



def extract_usage(style: str, resp: dict) -> Usage:
    """Token usage from a non-streaming response JSON. Missing fields -> 0."""
    if not isinstance(resp, dict):
        return Usage()
    if style == "bedrock":
        b = _bedrock_usage(resp)
        if b.total():
            return b
    if "usageMetadata" in resp or style=="gemini":
        meta = resp.get("usageMetadata")
        if isinstance(meta,dict):
            return _gemini_usage(meta)
    usage = resp.get("usage")
    if usage is None and isinstance(resp.get("response"), dict):  # wrapped responses-API object
        usage = resp["response"].get("usage")
    u = _usage_from_dict(usage)
    tier = resp.get("service_tier") or _dict(resp.get("response")).get("service_tier")
    if isinstance(tier, str):
        u.service_tier = tier
    return u


def _bedrock_usage(resp: dict) -> Usage:
    """Usage from a Bedrock response body: Converse / Claude / Nova ``usage``, else the per-model InvokeModel
    fields (Llama, Titan), else ``amazon-bedrock-invocationMetrics`` (the last chunk of a stream)."""
    if isinstance(resp.get("usage"), dict):
        u = _usage_from_dict(resp["usage"])
        if u.total():
            return u
    if "prompt_token_count" in resp or "generation_token_count" in resp:            # Meta Llama
        return Usage(input_tokens=_int(resp.get("prompt_token_count")),
                     output_tokens=_int(resp.get("generation_token_count")))
    if "inputTextTokenCount" in resp:                                                # Amazon Titan Text
        results = resp.get("results") if isinstance(resp.get("results"), list) else []
        return Usage(input_tokens=_int(resp.get("inputTextTokenCount")),
                     output_tokens=sum(_int(_dict(r).get("tokenCount")) for r in results))
    m = _dict(resp.get("amazon-bedrock-invocationMetrics"))
    return Usage(input_tokens=_int(m.get("inputTokenCount")), output_tokens=_int(m.get("outputTokenCount")),
                 cache_read_tokens=_int(m.get("cacheReadInputTokenCount")),
                 cache_write_tokens=_int(m.get("cacheWriteInputTokenCount")))


def bedrock_header_usage(headers: Mapping[str, str] | None) -> Usage:
    """Usage from the ``X-Amzn-Bedrock-*-Token-Count`` headers InvokeModel returns for every model."""
    def h(name: str) -> int:
        return _int(_header_value(headers, f"x-amzn-bedrock-{name}"))
    return Usage(input_tokens=h("input-token-count"), output_tokens=h("output-token-count"),
                 cache_read_tokens=h("cache-read-input-token-count"),
                 cache_write_tokens=h("cache-write-input-token-count"))


def _bedrock_text(resp: dict) -> str:
    """Assistant text of a Bedrock response: Converse / Nova ``output.message``, or a model-native body."""
    msg = _dict(_dict(resp.get("output")).get("message"))
    if isinstance(msg.get("content"), list):
        return "".join(b["text"] for b in msg["content"] if isinstance(b, dict) and isinstance(b.get("text"), str))
    if isinstance(resp.get("generation"), str):                                      # Meta Llama
        return resp["generation"]
    for key, field in (("results", "outputText"), ("outputs", "text"), ("generations", "text")):
        items = resp.get(key)                                                        # Titan / Mistral / Cohere
        if isinstance(items, list) and items and isinstance(_dict(items[0]).get(field), str):
            return "".join(_dict(i).get(field) or "" for i in items[:1])
    if isinstance(resp.get("completion"), str):                                      # Claude text completions
        return resp["completion"]
    return ""


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

def _gemini_text(resp:dict)-> str:
    candidates = resp.get("candidates")
    if not isinstance(candidates,list) or not candidates:
        return""

    first = candidates[0]
    if not isinstance(first, dict):
        return ""
    content = first.get("content") or {}
    parts = content.get("parts") or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))

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
        if style == "bedrock":
            t = _bedrock_text(resp)
            if t:
                return t
        if "choices" in resp:
            return _openai_chat_text(resp)
        if "output" in resp or "output_text" in resp:
            return _responses_text(resp)
        if "content" in resp:
            return _anthropic_text(resp.get("content"))
        if style == "gemini" or "candidates" in resp:
            return _gemini_text(resp)
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
    if u.cache_write_1h_tokens:
        dst.cache_write_1h_tokens = u.cache_write_1h_tokens
    if u.speed:
        dst.speed = u.speed
    if u.web_search_requests:
        dst.web_search_requests = u.web_search_requests


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
                if isinstance(resp.get("service_tier"), str) and resp["service_tier"]:
                    usage.service_tier = resp["service_tier"]
                if etype != "response.created" and etype != "response.in_progress":
                    completed_text = _responses_text(resp) or completed_text
                continue
            if isinstance(etype, str) and etype.startswith("response."):
                continue
            
            #Google Gemini API Responses
            if style == "gemini" or "candidates" in obj or "usageMetadata" in obj:
                # 1. Extract text delta from candidate parts
                for cand in obj.get("candidates") or []:
                    for part in (cand.get("content") or {}).get("parts") or []:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            text_parts.append(part["text"])
                    
                
                # 2. Extract usage if present in this chunk
                if isinstance(obj.get("usageMetadata"), dict):
                    g = _gemini_usage(obj["usageMetadata"])
                    # usageMetadata is cumulative per chunk: keep the latest non-empty totals
                    if g.input_tokens:
                        usage.input_tokens = g.input_tokens
                    if g.output_tokens:
                        usage.output_tokens = g.output_tokens
                    if g.cache_read_tokens:
                        usage.cache_read_tokens = g.cache_read_tokens
                if not model and isinstance(obj.get("modelVersion"), str):
                    model = obj["modelVersion"]
                continue
            # ---- openai chat / legacy completions chunks ----
            if not model and isinstance(obj.get("model"), str) and obj["model"]:
                model = obj["model"]
            if isinstance(obj.get("service_tier"), str) and obj["service_tier"]:
                usage.service_tier = obj["service_tier"]
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


def _error_message(err: Any) -> str:
    if isinstance(err, dict):
        m = err.get("message")
        t = err.get("type") or err.get("code")
        if isinstance(m, str) and m:
            return m[:500]
        if isinstance(t, str) and t:
            return t[:500]
        return "upstream error"
    if isinstance(err, str) and err:
        return err[:500]
    return ""


def body_error(resp: Any) -> str | None:
    """Message of a top-level ``error`` in a (successful-status) JSON response, e.g. gateways that answer
    HTTP 200 with ``{"error": {...}}``; None otherwise."""
    if isinstance(resp, dict) and resp.get("error"):
        return _error_message(resp["error"]) or "upstream error"
    return None


def sse_error(raw: str) -> str | None:
    """Message of the first in-band error in an SSE transcript (anthropic ``event: error``, openai
    ``data: {"error": ...}``, responses ``response.failed``); None when the stream carried no error."""
    if not isinstance(raw, str) or "error" not in raw and "response.failed" not in raw:
        return None
    for name, data in _sse_events(raw):
        payload = data.strip()
        if not payload.startswith("{"):
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        etype = obj.get("type") if isinstance(obj.get("type"), str) else name
        if etype == "error" or (obj.get("error") and "choices" not in obj and etype != "message_start"):
            return _error_message(obj.get("error")) or _error_message(obj.get("message")) or "upstream stream error"
        if etype == "response.failed":
            return _error_message(_dict(obj.get("response")).get("error")) or "response failed"
    return None


# --------------------------------------------------------------------------------------------
# AWS EventStream (Bedrock's streaming responses)
# --------------------------------------------------------------------------------------------
EVENTSTREAM_CTYPE = "application/vnd.amazon.eventstream"
_ES_MAX_MESSAGE = 16 * 1024 * 1024
_ES_FIXED = {2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16}    # value widths of the fixed-size header types


def _es_headers(buf: bytes) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    i = 0
    try:
        while i < len(buf):
            nlen = buf[i]
            name = buf[i + 1:i + 1 + nlen].decode("utf-8")
            i += 1 + nlen
            t = buf[i]
            i += 1
            val: Any
            if t in (0, 1):                       # bool true / false
                val = t == 0
            elif t in (6, 7):                     # bytes / string, 2-byte length
                (vlen,) = struct.unpack_from(">H", buf, i)
                raw = buf[i + 2:i + 2 + vlen]
                i += 2 + vlen
                val = raw.decode("utf-8", "replace") if t == 7 else raw
            elif t in _ES_FIXED:
                raw = buf[i:i + _ES_FIXED[t]]
                i += _ES_FIXED[t]
                val = raw.hex() if t == 9 else int.from_bytes(raw, "big", signed=True)
            else:
                return None
            if i > len(buf):
                return None
            out[name] = val
    except (IndexError, struct.error, UnicodeDecodeError):
        return None
    return out


def decode_eventstream(raw: bytes) -> tuple[list[tuple[dict[str, Any], bytes]], int]:
    """Split AWS EventStream bytes into (headers, payload) messages, checking both CRC32s. Returns the messages
    and the number of bytes used: decoding stops at a partial trailing message or the first corrupt one."""
    frames: list[tuple[dict[str, Any], bytes]] = []
    i, n = 0, len(raw)
    while n - i >= 16:
        total, hlen, prelude_crc = struct.unpack_from(">III", raw, i)
        if zlib.crc32(raw[i:i + 8]) != prelude_crc or total < 16 + hlen or total > _ES_MAX_MESSAGE:
            break
        if i + total > n:
            break
        (msg_crc,) = struct.unpack_from(">I", raw, i + total - 4)
        if zlib.crc32(raw[i:i + total - 4]) != msg_crc:
            break
        headers = _es_headers(raw[i + 12:i + 12 + hlen])
        if headers is None:
            break
        frames.append((headers, raw[i + 12 + hlen:i + total - 4]))
        i += total
    return frames, i


def encode_eventstream(headers: Mapping[str, Any], payload: bytes) -> bytes:
    """One EventStream message (used to replay recorded Bedrock streams). Header values may be str, bool or
    int; raises ValueError for a header name over 255 bytes or a value over 65535 bytes."""
    hb = bytearray()
    for k, v in headers.items():
        kb = str(k).encode("utf-8")
        if not 0 < len(kb) < 256:
            raise ValueError(f"EventStream header name too long: {len(kb)} bytes")
        hb += bytes([len(kb)]) + kb
        if isinstance(v, bool):
            hb += b"\x00" if v else b"\x01"
        elif isinstance(v, int) and -2 ** 63 <= v < 2 ** 63:
            hb += b"\x05" + struct.pack(">q", v)
        else:
            vb = str(v).encode("utf-8")
            if len(vb) > 0xFFFF:
                raise ValueError(f"EventStream header value too long: {len(vb)} bytes")
            hb += b"\x07" + struct.pack(">H", len(vb)) + vb
    prelude = struct.pack(">II", 16 + len(hb) + len(payload), len(hb))
    head = prelude + struct.pack(">I", zlib.crc32(prelude)) + bytes(hb) + payload
    return head + struct.pack(">I", zlib.crc32(head))


def eventstream_to_text(raw: bytes) -> str:
    """Readable JSON-lines form of a Bedrock event stream, one ``{"headers": ..., "payload": ...}`` per message.
    The base64 ``bytes`` of an InvokeModelWithResponseStream ``chunk`` is decoded into ``"chunk"``.
    :func:`text_to_eventstream` turns it back into EventStream bytes for replay."""
    frames, used = decode_eventstream(raw)
    lines = []
    for headers, payload in frames:
        hdrs = {k: v for k, v in headers.items() if isinstance(v, (str, int))}      # bool is an int too
        try:
            obj: dict[str, Any] = {"headers": hdrs}
            p = json.loads(payload.decode("utf-8")) if payload else None
            if isinstance(p, dict) and isinstance(p.get("bytes"), str):
                try:
                    obj["chunk"] = json.loads(base64.b64decode(p["bytes"], validate=True).decode("utf-8"))
                    p = {k: v for k, v in p.items() if k != "bytes"}
                except (binascii.Error, ValueError, RecursionError):
                    pass
            obj["payload"] = p
            lines.append(json.dumps(obj, ensure_ascii=False))
        except (ValueError, RecursionError):    # not JSON, or nested too deeply to parse: keep the raw bytes
            lines.append(json.dumps({"headers": hdrs, "payload_b64": base64.b64encode(payload).decode("ascii")}))
    if used < len(raw):
        lines.append(json.dumps({"llmtap": f"undecodable or incomplete EventStream data: {len(raw) - used} bytes"}))
    return "\n".join(lines)


def text_to_eventstream(text: str) -> list[bytes]:
    """EventStream messages rebuilt from :func:`eventstream_to_text` output (one bytes object per message)."""
    out: list[bytes] = []
    for line in (text or "").splitlines():
        try:   # a line a person edited badly (cassettes are plain files) is skipped, never a crash
            obj = json.loads(line)
            if not isinstance(obj, dict) or not isinstance(obj.get("headers"), dict):
                continue
            if isinstance(obj.get("payload_b64"), str):
                payload = base64.b64decode(obj["payload_b64"], validate=True)
            else:
                p = obj.get("payload")
                if "chunk" in obj:
                    inner = json.dumps(obj["chunk"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    p = {**_dict(p), "bytes": base64.b64encode(inner).decode("ascii")}
                payload = (b"" if p is None
                           else json.dumps(p, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            out.append(encode_eventstream(obj["headers"], payload))
        except (ValueError, TypeError, RecursionError, binascii.Error):
            continue
    return out


def parse_bedrock_stream(text: str) -> tuple[Usage, str, str | None, str | None]:
    """(usage, text, model, error) from :func:`eventstream_to_text` output. Handles ConverseStream events and
    InvokeModelWithResponseStream chunks: Claude's Messages events, Nova, Llama, Titan and Mistral chunks, and
    the ``amazon-bedrock-invocationMetrics`` of the last chunk."""
    usage, metrics = Usage(), Usage()
    parts: list[str] = []
    claude: list[dict] = []
    err: str | None = None
    for line in (text or "").splitlines():
        try:
            obj = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("llmtap"), str) and err is None:
            err = "incomplete or corrupt AWS EventStream response"     # see eventstream_to_text
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("headers"), dict):
            continue
        h, p = obj["headers"], _dict(obj.get("payload"))
        if h.get(":message-type", "event") != "event":          # "exception" / "error" messages
            if err is None:
                kind = h.get(":exception-type") or h.get(":error-code") or "error"
                msg = p.get("message") or p.get("Message") or h.get(":error-message") or "stream error"
                err = f"{kind}: {msg}"
            continue
        etype = h.get(":event-type")
        if etype == "contentBlockDelta":                         # ConverseStream
            delta = _dict(p.get("delta"))
            if isinstance(delta.get("text"), str):
                parts.append(delta["text"])
        elif etype == "metadata":
            _merge_usage(usage, _dict(p.get("usage")))
        elif etype == "chunk":                                   # InvokeModelWithResponseStream
            c = _dict(obj.get("chunk"))
            m = _dict(c.get("amazon-bedrock-invocationMetrics"))
            if m:
                metrics = _bedrock_usage({"amazon-bedrock-invocationMetrics": m})
            if isinstance(c.get("type"), str):                   # Claude: Anthropic Messages stream events
                claude.append(c)
                continue
            delta = _dict(_dict(c.get("contentBlockDelta")).get("delta"))   # Nova
            if isinstance(delta.get("text"), str):
                parts.append(delta["text"])
            if isinstance(_dict(c.get("metadata")).get("usage"), dict):
                _merge_usage(usage, c["metadata"]["usage"])
            for field in ("generation", "outputText", "completion"):       # Llama / Titan / Claude 2
                if isinstance(c.get(field), str):
                    parts.append(c[field])
            outputs = c.get("outputs")                                       # Mistral
            if isinstance(outputs, list) and outputs and isinstance(_dict(outputs[0]).get("text"), str):
                parts.append(outputs[0]["text"])
    model = None
    if claude:
        sse = "".join(f"data: {json.dumps(e)}\n\n" for e in claude)
        u, t, model = parse_sse("anthropic", sse)
        parts.append(t)
        if u.total():
            usage = u
        err = err or sse_error(sse)
    if not usage.total():
        usage = metrics
    return usage, "".join(parts), model, err


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
    elif style == "gemini":
        obj = {"error": {"code": status, "message": message, "status": err_type.upper()}}
    elif style == "bedrock":                       # AWS JSON protocol: botocore reads __type and message
        obj = {"__type": err_type, "message": message}
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


def forward_headers(
    headers: Mapping[str, str],
    preserve_signed: bool = False,
) -> dict[str, str]:
    """Headers to send upstream: drop hop-by-hop (+ those named in Connection) and x-llmtap-*."""
    items = _items(headers)
    extra: set[str] = set()
    for key, value in items:
        if key.lower() == "connection":
            extra |= {
                token.strip().lower()
                for token in value.split(",")
                if token.strip()
            }

    signed = _aws_signed_headers(headers) if preserve_signed else set()
    signed_transport_headers = {
        "host",
        "content-length",
        "accept-encoding",
    }

    out: dict[str, str] = {}

    for key, value in items:
        lower_key = key.lower()

        # `expect` is answered by llmtap itself (100-continue); the body is
        # re-sent with a Content-Length.
        if lower_key in extra or lower_key == "expect":
            continue
        # x-llmtap-* tags are llmtap's own, except when a SigV4 signature covers them: dropping a signed
        # header would make AWS reject the request.
        if lower_key.startswith("x-llmtap-") and lower_key not in signed:
            continue

        if lower_key in HOP_BY_HOP:
            keep_signed = (
                preserve_signed
                and lower_key in signed
                and lower_key in signed_transport_headers
            )
            if not keep_signed:
                continue

        out[key] = value

    if not any(key.lower() == "accept-encoding" for key in out):
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
