"""llmtap + the OpenAI Python SDK. Note there is no llmtap-specific code here.

    pip install openai
    export OPENAI_API_KEY=sk-...

    llmtap wrap -- python examples/openai_quickstart.py
    llmtap wrap --guard redact -- python examples/openai_quickstart.py   # watch the fake key get redacted

`llmtap wrap` sets OPENAI_BASE_URL (http://127.0.0.1:<port>/v1), which the SDK reads automatically.
"""
from __future__ import annotations

import os

from openai import OpenAI

MODEL = os.environ.get("MODEL", "gpt-4o-mini")

client = OpenAI(default_headers={"X-LLMTap-Session": "quickstart", "X-LLMTap-Tag-Feature": "demo"})


def main() -> None:
    print("base URL:", os.environ.get("OPENAI_BASE_URL", "(not set: run me under `llmtap wrap`)"))

    # 1. Plain call: shows up in the dashboard with tokens, latency and cost.
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Give me one tip for writing good unit tests."}],
    )
    print(r.choices[0].message.content)

    # 2. Streaming. include_usage gives llmtap exact token counts (otherwise they're estimated).
    stream = client.chat.completions.create(
        model=MODEL,
        stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "user", "content": "Count from 1 to 5, one per line."}],
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    # 3. A prompt that "accidentally" contains a credential. This is AWS's documented example key, not a real one.
    #    --guard warn (default) flags it, --guard redact replaces it, --guard block refuses the request (HTTP 400).
    leaked = "Why does this config fail?\n[default]\naws_access_key_id = AKIAIOSFODNN7EXAMPLE\nregion = us-east-1"
    try:
        r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": leaked}])
        print(r.choices[0].message.content)
    except Exception as e:  # BadRequestError under --guard block
        print("request refused by llmtap guard:", e)


if __name__ == "__main__":
    main()
