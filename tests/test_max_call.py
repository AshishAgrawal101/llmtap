"""--max-call-usd: a single call likely to cost more than the limit is refused before it's sent."""
from __future__ import annotations

import base64
from typing import Any

from test_vault import _call

PRICES = {"big-model": {"input": 1000.0, "output": 0.0}}        # $0.001 per input token


def _msg(content: Any, **extra: Any) -> dict:
    return {"model": "big-model", "messages": [{"role": "user", "content": content}], **extra}


def test_giant_call_refused_small_allowed(tap: Any, mock_upstream: Any) -> None:
    t = tap(max_call_usd=5.0, pricing_overrides=PRICES)
    auth = {"Authorization": "Bearer sk-x"}
    assert _call(t, "/v1/chat/completions", _msg("hello"), auth)[0] == 200
    status, body = _call(t, "/v1/chat/completions", _msg("hello world " * 3_400), auth)   # ~10k tokens = ~$10
    err = body["error"]
    assert status == 403 and err["type"] == "llmtap_call_too_expensive"
    assert "$5.00" in err["message"] and "big-model" in err["message"]
    assert len(mock_upstream.requests) == 1


def test_images_count_as_media_not_length(tap: Any) -> None:
    t = tap(max_call_usd=5.0, pricing_overrides=PRICES)
    img = base64.b64encode(bytes(range(256)) * 4000).decode()                      # ~1.4 MB of base64
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
               {"type": "text", "text": "what is this?"}]
    assert _call(t, "/v1/chat/completions", _msg(content), {"Authorization": "Bearer sk-x"})[0] == 200


def test_off_by_default_and_unpriced_models_pass(tap: Any) -> None:
    t = tap(pricing_overrides=PRICES)
    assert _call(t, "/v1/chat/completions", _msg("hello world " * 3_400), {"Authorization": "Bearer sk-x"})[0] == 200
    t2 = tap(max_call_usd=0.01)
    body = {"model": "some-unknown-model", "messages": [{"role": "user", "content": "hello world " * 3_400}]}
    assert _call(t2, "/v1/chat/completions", body, {"Authorization": "Bearer sk-x"})[0] == 200


def test_long_word_is_text_not_media(tap: Any) -> None:
    t = tap(max_call_usd=5.0, pricing_overrides=PRICES)
    status, body = _call(t, "/v1/chat/completions", _msg("x" * 40_000), {"Authorization": "Bearer sk-x"})
    assert status == 403 and body["error"]["type"] == "llmtap_call_too_expensive"
