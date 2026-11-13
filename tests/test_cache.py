"""Tests for llmtap.cache."""
from __future__ import annotations

import json
import threading

import pytest

from llmtap import cache as cache_mod
from llmtap.cache import ResponseCache, cache_key
from llmtap.models import CachedResponse


def _body(obj: object) -> bytes:
    return json.dumps(obj).encode()


def _resp(body: str = '{"ok":true}', **kw: object) -> CachedResponse:
    return CachedResponse(status=200, headers={"content-type": "application/json"}, body=body, streamed=False, **kw)


# ---- cache_key ------------------------------------------------------------------------------------

def test_key_is_sha256_hex() -> None:
    k = cache_key("openai", "/v1/chat/completions", b"{}")
    assert len(k) == 64 and all(c in "0123456789abcdef" for c in k)


def test_key_ignores_key_order_and_whitespace() -> None:
    a = b'{"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}'
    b = b'{"temperature":0,"messages":[{"content":"hi","role":"user"}],"model":"gpt-4o"}'
    assert cache_key("openai", "/v1/chat/completions", a) == cache_key("openai", "/v1/chat/completions", b)


def test_key_ignores_volatile_top_level_fields() -> None:
    base = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}]}
    noisy = dict(base, user="u-123", metadata={"user_id": "x"}, request_id="r-9")
    assert cache_key("anthropic", "/v1/messages", _body(base)) == cache_key("anthropic", "/v1/messages", _body(noisy))


def test_volatile_keys_only_stripped_at_top_level() -> None:
    a = {"model": "m", "messages": [{"role": "user", "content": "hi", "metadata": 1}]}
    b = {"model": "m", "messages": [{"role": "user", "content": "hi", "metadata": 2}]}
    assert cache_key("openai", "/v1/x", _body(a)) != cache_key("openai", "/v1/x", _body(b))


def test_key_differs_on_body_provider_path() -> None:
    b1 = _body({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    b2 = _body({"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]})
    b3 = _body({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})
    keys = {
        cache_key("openai", "/v1/chat/completions", b1),
        cache_key("openai", "/v1/chat/completions", b2),
        cache_key("openai", "/v1/chat/completions", b3),
        cache_key("openrouter", "/v1/chat/completions", b1),
        cache_key("openai", "/v1/responses", b1),
    }
    assert len(keys) == 5


def test_key_ignores_query_string() -> None:
    b = b'{"a":1}'
    assert cache_key("anthropic", "/v1/messages?beta=true", b) == cache_key("anthropic", "/v1/messages", b)


def test_key_non_json_body_hashes_raw_bytes() -> None:
    assert cache_key("openai", "/v1/x", b"not json") == cache_key("openai", "/v1/x", b"not json")
    assert cache_key("openai", "/v1/x", b"not json") != cache_key("openai", "/v1/x", b"not json!")
    assert cache_key("openai", "/v1/x", b"\xff\xfe") != cache_key("openai", "/v1/x", b"")


def test_key_empty_body() -> None:
    assert cache_key("openai", "/v1/models", b"") == cache_key("openai", "/v1/models", b"")


def test_key_unicode_stable() -> None:
    a = '{"q":"héllo ✓"}'.encode()
    b = json.dumps({"q": "héllo ✓"}).encode()  # ascii-escaped form
    assert cache_key("openai", "/v1/x", a) == cache_key("openai", "/v1/x", b)


# ---- ResponseCache --------------------------------------------------------------------------------

def test_put_get_roundtrip_memory() -> None:
    c = ResponseCache(":memory:")
    assert c.get("k") is None
    r = CachedResponse(status=200, headers={"x-a": "1"}, body="event: x\ndata: {}\n\n", streamed=True, cost_usd=0.0123)
    c.put("k", r)
    got = c.get("k")
    assert got == r
    assert c.stats() == {"entries": 1, "hits": 1}


def test_put_replaces() -> None:
    c = ResponseCache(":memory:")
    c.put("k", _resp("1"))
    c.put("k", _resp("2"))
    assert c.get("k").body == "2"
    assert c.stats()["entries"] == 1


def test_hits_counted_only_on_hit() -> None:
    c = ResponseCache(":memory:")
    c.get("missing")
    c.put("k", _resp())
    c.get("k")
    c.get("k")
    assert c.stats() == {"entries": 1, "hits": 2}


def test_clear() -> None:
    c = ResponseCache(":memory:")
    c.put("a", _resp())
    c.put("b", _resp())
    c.get("a")
    c.clear()
    assert c.stats() == {"entries": 0, "hits": 0}
    assert c.get("a") is None


def test_ttl_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1_000_000.0]
    monkeypatch.setattr(cache_mod.time, "time", lambda: now[0])
    c = ResponseCache(":memory:", ttl_s=60)
    c.put("k", _resp())
    now[0] += 59
    assert c.get("k") is not None
    now[0] += 2
    assert c.get("k") is None
    assert c.stats()["entries"] == 0  # expired row purged


def test_ttl_zero_means_forever(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1_000_000.0]
    monkeypatch.setattr(cache_mod.time, "time", lambda: now[0])
    c = ResponseCache(":memory:", ttl_s=0)
    c.put("k", _resp())
    now[0] += 10 * 365 * 86400
    assert c.get("k") is not None


def test_file_persistence_and_shared_db(tmp_path) -> None:
    db = tmp_path / "sub" / "llmtap.db"
    c = ResponseCache(str(db))
    c.put("k", _resp("persisted", cost_usd=0.5))
    c.close()
    # an unrelated table in the same file is fine
    import sqlite3

    con = sqlite3.connect(db)
    con.execute("CREATE TABLE IF NOT EXISTS exchanges (id TEXT)")
    con.commit()
    con.close()
    c2 = ResponseCache(str(db))
    got = c2.get("k")
    assert got is not None and got.body == "persisted" and got.cost_usd == 0.5
    assert c2.stats()["hits"] == 1  # in-process counter starts fresh


def test_thread_safety() -> None:
    c = ResponseCache(":memory:")
    errors: list[BaseException] = []

    def work(n: int) -> None:
        try:
            for i in range(50):
                c.put(f"{n}-{i}", _resp(str(i)))
                assert c.get(f"{n}-{i}").body == str(i)
        except BaseException as e:  # pragma: no cover - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert c.stats() == {"entries": 400, "hits": 400}
