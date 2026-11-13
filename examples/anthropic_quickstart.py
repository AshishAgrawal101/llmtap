"""llmtap + the Anthropic Python SDK. Note there is no llmtap-specific code here.

    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

    llmtap wrap -- python examples/anthropic_quickstart.py            # see every call + cost
    llmtap wrap --cache -- python examples/anthropic_quickstart.py    # run twice: second run costs $0

`llmtap wrap` sets ANTHROPIC_BASE_URL, which the SDK reads automatically.
"""
from __future__ import annotations

import os

import anthropic

MODEL = os.environ.get("MODEL", "claude-haiku-4-5")

client = anthropic.Anthropic(
    # Optional: tag traffic so you can slice it in the dashboard / `llmtap stats --by tag:feature`.
    # llmtap strips X-LLMTap-* headers before forwarding them to Anthropic.
    default_headers={"X-LLMTap-Session": "quickstart", "X-LLMTap-Tag-Feature": "demo"},
)


def ask(question: str) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(block.text for block in msg.content if block.type == "text")


def stream(question: str) -> None:
    with client.messages.stream(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": question}],
    ) as s:
        for text in s.text_stream:
            print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    print("base URL:", os.environ.get("ANTHROPIC_BASE_URL", "(not set: run me under `llmtap wrap`)"))
    print(ask("In one sentence: what is a reverse proxy?"))
    # Identical request: with --cache this one is served locally and costs nothing.
    print(ask("In one sentence: what is a reverse proxy?"))
    # Streaming is relayed chunk by chunk; llmtap records time-to-first-token too.
    stream("Write a haiku about watching network traffic.")
