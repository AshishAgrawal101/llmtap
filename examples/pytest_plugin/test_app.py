"""Run `pytest` here: the first run records (needs a key), every later run replays for free."""
import pytest

from app import classify, summarize


def test_summarize(llmtap):
    out = summarize("llmtap records LLM calls once and replays them forever.")
    assert out                      # assert on shape, not exact wording, so re-recording stays painless
    assert llmtap.calls == 1


@pytest.mark.parametrize("text", ["I love this", "This is awful"])
def test_classify(llmtap, text):    # each parameter gets its own cassette
    assert isinstance(classify(text), str)


@pytest.mark.llmtap(cassette="shared_greeting", guard="redact")   # marker alone is enough: no fixture argument
def test_marker_only():
    assert summarize("hello")
