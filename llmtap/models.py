"""Shared data model. Every module speaks in these types."""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0     # anthropic cache_read_input_tokens / openai cached_tokens
    cache_write_tokens: int = 0    # anthropic cache_creation_input_tokens
    estimated: bool = False        # True when tokens were guessed (chars/4), not reported

    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Route:
    provider: str          # "anthropic" | "openai" | custom upstream name (e.g. "ollama")
    upstream_base: str     # e.g. "https://api.anthropic.com" (no trailing slash)
    upstream_path: str     # path + query to send upstream, e.g. "/v1/messages?beta=true"
    style: str             # wire format: "anthropic" | "openai"


@dataclass
class Exchange:
    """One request/response through the proxy."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)    # request start, epoch seconds
    provider: str = "unknown"
    style: str = "openai"
    method: str = "POST"
    path: str = ""                 # path as received by the proxy
    model: str | None = None
    request_body: str = ""         # as sent upstream (post-redaction), utf-8 text
    status: int = 0
    response_body: str = ""        # raw JSON text, or raw SSE text when streamed
    streamed: bool = False
    latency_ms: float = 0.0
    ttft_ms: float | None = None   # time to first byte of body (streaming)
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    saved_usd: float = 0.0         # cost avoided when served from cache/cassette
    source: str = "upstream"       # "upstream" | "cache" | "cassette" | "blocked" | "budget" | "loop" | "error"
    tags: dict[str, str] = field(default_factory=dict)       # from X-LLMTap-* headers
    guard_findings: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    output_text: str = ""          # assistant text extracted from response (for UI/search)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Exchange":
        d = dict(d)
        u = d.pop("usage", None) or {}
        known = {f for f in cls.__dataclass_fields__}
        ex = cls(**{k: v for k, v in d.items() if k in known})
        ex.usage = Usage(**{k: v for k, v in u.items() if k in Usage.__dataclass_fields__})
        return ex


@dataclass
class CachedResponse:
    """A response that can be replayed verbatim (from cache or cassette)."""
    status: int
    headers: dict[str, str]
    body: str                      # raw JSON text or raw SSE text
    streamed: bool
    cost_usd: float = 0.0          # original cost, used to compute saved_usd
