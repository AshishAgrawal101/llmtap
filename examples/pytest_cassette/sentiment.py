"""The application code under test: a tiny LLM-powered feature."""
from __future__ import annotations

import anthropic

MODEL = "claude-haiku-4-5"


def classify_sentiment(client: anthropic.Anthropic, text: str) -> str:
    """Return "positive", "negative" or "neutral"."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=5,
        system="Classify the sentiment of the user's text. Answer with exactly one word: positive, negative or neutral.",
        messages=[{"role": "user", "content": text}],
    )
    answer = "".join(b.text for b in msg.content if b.type == "text").strip().lower()
    return answer.rstrip(".")


def summarize(client: anthropic.Anthropic, text: str, max_words: int = 20) -> str:
    """One-sentence summary, streamed (cassettes record and replay streams too)."""
    with client.messages.stream(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": f"Summarize in at most {max_words} words:\n\n{text}"}],
    ) as stream:
        return "".join(stream.text_stream).strip()
