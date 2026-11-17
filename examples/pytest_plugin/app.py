"""A tiny LLM feature, written with only the standard library so the example runs anywhere.

With the official SDK it is the same thing; the plugin points ANTHROPIC_BASE_URL at the proxy either way:

    client = anthropic.Anthropic()            # picks up ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY
    msg = client.messages.create(model=MODEL, max_tokens=100, messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text
"""
from __future__ import annotations

import json
import os
import urllib.request

MODEL = "claude-haiku-4-5"


def _ask(prompt: str) -> str:
    req = urllib.request.Request(
        os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com") + "/v1/messages",
        data=json.dumps({"model": MODEL, "max_tokens": 100,
                         "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""), "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
    )
    # ProxyHandler({}): talk to the (local) base URL directly, ignoring any HTTP_PROXY in the environment
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=60) as resp:
        return json.loads(resp.read())["content"][0]["text"]


def summarize(text: str) -> str:
    return _ask(f"Summarize in one sentence: {text}")


def classify(text: str) -> str:
    return _ask(f"Answer with one word, positive or negative. Sentiment of: {text}").strip().lower()
