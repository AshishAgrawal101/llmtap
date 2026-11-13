"""LLM tests that are deterministic, offline and free in CI thanks to llmtap replay."""
from __future__ import annotations

import pytest

from sentiment import classify_sentiment, summarize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I absolutely love this keyboard, best purchase all year!", "positive"),
        ("The update deleted my files and support never answered.", "negative"),
        ("The package arrived on Tuesday.", "neutral"),
    ],
)
def test_classify_sentiment(claude, text: str, expected: str) -> None:
    assert classify_sentiment(claude, text) == expected


def test_summary_is_short_and_on_topic(claude) -> None:
    text = (
        "llmtap is a local proxy for LLM APIs. It records every request, tracks cost, caches responses, "
        "replays recorded traffic in CI, redacts secrets from prompts and stops runaway agent loops."
    )
    summary = summarize(claude, text)
    assert 0 < len(summary.split()) <= 25
    assert "proxy" in summary.lower() or "llm" in summary.lower()
