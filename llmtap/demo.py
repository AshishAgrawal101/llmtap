"""Realistic synthetic traffic for `llmtap demo` (and the e2e tests).

Pure urllib client that talks to a running llmtap proxy: a mix of Anthropic / OpenAI calls,
streaming and non-streaming, several models, repeated prompts (cache hits), a prompt with a
leaked (fake) AWS key for the guard, and a burst of identical calls for the loop breaker.
Errors (429 / 409 / 400 / 5xx) are expected and swallowed.
"""
from __future__ import annotations

import json
import random
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

FAKE_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"

_PY_SNIPPET = '''def merge_intervals(intervals):
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged'''

_INCIDENT = (
    "INC-4821 — 2026-10-28 14:02 UTC. Checkout API p99 latency rose from 180ms to 4.2s after deploy "
    "v2.31.0. Root cause: a new ORM eager-load on Order.line_items produced N+1 queries against the "
    "replica, saturating its connection pool (max 200). Mitigation: rollback at 14:37, pool size raised "
    "to 400. Impact: 35 minutes, ~3.1% of checkouts failed, est. revenue impact $48k. Follow-ups: add "
    "query-count assertions to CI, alert on replica pool saturation > 80%."
)

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"],
        },
    },
}

_SEARCH_TOOL_ANTHROPIC = {
    "name": "search_docs",
    "description": "Full-text search over the internal engineering wiki",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                     "required": ["query"]},
}


@dataclass
class Call:
    """One scripted request."""
    kind: str                      # "anthropic" | "chat" | "responses" | "embeddings"
    model: str
    feature: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    system: str | None = None
    stream: bool = False
    max_tokens: int = 1024
    extra: dict[str, Any] = field(default_factory=dict)
    input: Any = None              # responses / embeddings input
    session: str = ""

    def path(self) -> str:
        return {"anthropic": "/v1/messages", "chat": "/v1/chat/completions",
                "responses": "/v1/responses", "embeddings": "/v1/embeddings"}[self.kind]

    def body(self) -> dict[str, Any]:
        if self.kind == "anthropic":
            b: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "messages": self.messages}
            if self.system:
                b["system"] = self.system
        elif self.kind == "chat":
            msgs = ([{"role": "system", "content": self.system}] if self.system else []) + self.messages
            b = {"model": self.model, "messages": msgs}
            if self.stream:
                b["stream_options"] = {"include_usage": True}
        elif self.kind == "responses":
            b = {"model": self.model, "input": self.input}
            if self.system:
                b["instructions"] = self.system
        else:
            return {"model": self.model, "input": self.input}
        if self.stream:
            b["stream"] = True
        b.update(self.extra)
        return b

    def headers(self) -> dict[str, str]:
        h = {"content-type": "application/json", "user-agent": "llmtap-demo/0.1",
             "x-llmtap-tag-feature": self.feature}
        if self.session:
            h["x-llmtap-session"] = self.session
        if self.kind == "anthropic":
            h["x-api-key"] = "llmtap-demo-key"
            h["anthropic-version"] = "2023-06-01"
        else:
            h["authorization"] = "Bearer llmtap-demo-key"
        return h


def _u(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _a(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _script() -> tuple[list[Call], list[Call]]:
    """Return (ordered first pass, pool for repeats)."""
    code_review = Call("anthropic", "claude-sonnet-4-5", "code-review", stream=True,
                       system="You are a meticulous senior Python reviewer. Point out bugs first, then style.",
                       messages=[_u("Review this Python function for bugs and edge cases:\n\n" + _PY_SNIPPET)],
                       session="agent-run-17")
    incident = Call("chat", "gpt-4o", "summarize",
                    system="You write crisp executive summaries for incident reports.",
                    messages=[_u("Summarize this incident report in 3 bullet points for the VP of Eng:\n\n" + _INCIDENT)])
    sql = Call("anthropic", "claude-haiku-4-5", "sql-assistant",
               system="You are a PostgreSQL expert. Return only SQL in a code block.",
               messages=[_u("Write a SQL query that returns the top 10 customers by lifetime revenue in 2026, "
                            "joining orders(id, customer_id, total_cents, created_at) with customers(id, name, "
                            "email), excluding refunded orders.")])
    translate = Call("chat", "gpt-4o-mini", "i18n", stream=True,
                     system="Translate faithfully. Keep placeholders like {name} untouched.",
                     messages=[_u("Translate to Japanese: \"Hi {name}, your order has shipped and should arrive "
                                  "by Friday. Thanks for shopping with us!\"")])
    architecture = Call("anthropic", "claude-opus-4-1", "architecture", stream=True,
                        system="You are a staff engineer. Be concrete about trade-offs.",
                        messages=[_u("We're outgrowing a single Postgres instance at ~12k writes/sec. Options?"),
                                  _a("Main options: vertical scaling, read replicas + CQRS, partitioning, or "
                                     "sharding with Citus. What dominates your write load?"),
                                  _u("Mostly append-only event rows, 90% of reads hit the last 7 days. "
                                     "Should we partition by time or shard by tenant?")],
                        session="design-review")
    weather_tool = Call("chat", "gpt-5", "agent-tools",
                        system="You are a travel assistant. Use tools when you need live data.",
                        messages=[_u("Should I pack an umbrella for Lisbon tomorrow?")],
                        extra={"tools": [_WEATHER_TOOL], "tool_choice": "auto"}, session="agent-run-17")
    leaked = Call("anthropic", "claude-sonnet-4-5", "devops",
                  messages=[_u("Why does this deploy script fail with AccessDenied?\n\n"
                               f"export AWS_ACCESS_KEY_ID={FAKE_AWS_KEY}\n"
                               "aws s3 sync ./build s3://prod-web-assets --delete")])
    embed = Call("embeddings", "text-embedding-3-small", "rag-index",
                 input=["How do I rotate API keys?", "Rotating credentials without downtime",
                        "Quarterly OKR planning template"])
    unit_tests = Call("chat", "gpt-4o-mini", "test-gen",
                      messages=[_u("Write pytest unit tests for merge_intervals covering empty input, "
                                   "touching intervals and unsorted input.\n\n" + _PY_SNIPPET)])
    commit_msg = Call("anthropic", "claude-haiku-4-5", "git-helper", stream=True,
                      system="Write conventional-commit messages. Max 72 chars in the subject.",
                      messages=[_u("Write a commit message for: switched session store from Redis to Postgres "
                                   "advisory locks, removed redis dependency, added migration 0042.")])
    wiki_search = Call("anthropic", "claude-sonnet-4-5", "support-bot",
                       system="You answer engineering questions using the internal wiki. Cite pages.",
                       messages=[_u("What's our on-call escalation policy for SEV-2 incidents?")],
                       extra={"tools": [_SEARCH_TOOL_ANTHROPIC]}, session="support-4412")
    responses_api = Call("responses", "gpt-5", "release-notes", stream=True,
                         system="You write friendly, scannable release notes.",
                         input="Draft release notes for v2.32: dark mode, 40% faster search, "
                               "fixed CSV export timezone bug, deprecated the v1 webhooks API.")
    regex = Call("chat", "gpt-4o", "dev-tools",
                 messages=[_u("Give me a regex that matches ISO-8601 dates with optional time and timezone, "
                              "and explain each group.")])
    email = Call("anthropic", "claude-sonnet-4-5", "email-drafts",
                 messages=[_u("Draft a polite email to a customer explaining that their feature request "
                              "(SAML SSO for the Starter plan) is on our Q1 roadmap.")])
    classify = Call("chat", "gpt-4o-mini", "triage", stream=True,
                    system="Classify support tickets into: billing, bug, feature_request, account, other. "
                           "Reply with JSON {\"label\":..., \"confidence\":...}.",
                    messages=[_u("Ticket #88123: 'I was charged twice for October and the invoice page "
                                 "shows a 500 error.'")])
    docstring = Call("anthropic", "claude-haiku-4-5", "docs",
                     messages=[_u("Write a Google-style docstring for merge_intervals:\n\n" + _PY_SNIPPET)])
    explain_k8s = Call("chat", "gpt-5", "dev-tools",
                       messages=[_u("Explain why my Kubernetes pod is in CrashLoopBackOff with exit code 137 "
                                    "and how to confirm it's an OOM kill.")])
    story_points = Call("anthropic", "claude-opus-4-1", "planning",
                        system="You are a pragmatic engineering manager.",
                        messages=[_u("Break the epic 'Usage-based billing' into 6-8 tickets with rough "
                                     "story-point estimates and dependencies.")])
    outage = Call("chat", "gpt-4o", "summarize",
                  messages=[_u("Summarize the status page update (simulated upstream outage MOCK_ERROR_500)")])
    retry_loop = Call("chat", "gpt-4o-mini", "agent-loop",
                      system="You are an autonomous coding agent.",
                      messages=[_u("The test tests/test_billing.py::test_proration still fails. "
                                   "Fix it and try again.")], session="agent-run-17")

    first = [code_review, incident, sql, translate, architecture, weather_tool, leaked, embed,
             code_review, incident,                       # repeats -> cache hits
             retry_loop, retry_loop, retry_loop, retry_loop, retry_loop, retry_loop,  # runaway agent burst
             unit_tests, commit_msg, wiki_search, responses_api, regex, email, classify, docstring,
             explain_k8s, story_points, outage, sql, translate]
    pool = [code_review, incident, sql, translate, architecture, weather_tool, embed, unit_tests, commit_msg,
            wiki_search, responses_api, regex, email, classify, docstring, explain_k8s, story_points]
    return first, pool


def send(base_url: str, call: Call, timeout: float = 30.0) -> tuple[int, bytes]:
    """Send one call through the proxy. Never raises; returns (status, body) — status 0 on connection error."""
    data = json.dumps(call.body()).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + call.path(), data=data, method="POST",
                                 headers=call.headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except Exception:
            body = b""
        return e.code, body
    except Exception:
        return 0, b""


def generate_traffic(base_url: str, count: int = 40, delay_s: float = 0.3,
                     stop: threading.Event | None = None) -> None:
    """Send `count` realistic requests to the llmtap proxy at `base_url`, pausing ~delay_s between them."""
    first, pool = _script()
    rng = random.Random(4141)
    for i in range(count):
        if stop is not None and stop.is_set():
            return
        call = first[i] if i < len(first) else rng.choice(pool)
        send(base_url, call)
        if delay_s > 0 and i < count - 1:
            pause = delay_s * rng.uniform(0.4, 1.6)
            if call.feature == "agent-loop":
                pause = delay_s * 0.15  # the runaway agent hammers quickly
            if stop is not None:
                if stop.wait(pause):
                    return
            else:
                threading.Event().wait(pause)
