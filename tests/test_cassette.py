"""Tests for llmtap.cassette."""
from __future__ import annotations

import json
import threading

import pytest

from llmtap.cassette import Cassette, CassetteMiss, redact_request_for_cassette
from llmtap.models import CachedResponse


def _resp(body: str = '{"ok":true}', **kw) -> CachedResponse:
    kw.setdefault("headers", {"content-type": "application/json"})
    return CachedResponse(status=200, body=body, streamed=kw.pop("streamed", False), **kw)


def _lines(path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_cassette_miss_is_exception() -> None:
    assert issubclass(CassetteMiss, Exception)


def test_invalid_mode(tmp_path) -> None:
    with pytest.raises(ValueError):
        Cassette(str(tmp_path / "c.jsonl"), mode="bogus")


def test_roundtrip(tmp_path) -> None:
    p = tmp_path / "c.jsonl"
    c = Cassette(str(p), mode="record")
    r = CachedResponse(status=201, headers={"content-type": "text/event-stream"}, body="data: {}\n\n", streamed=True, cost_usd=0.01)
    c.record("k1", {"model": "gpt-4o", "messages": []}, r)
    assert len(c) == 1
    assert c.lookup("k1") is None  # record mode never replays
    c2 = Cassette(str(p), mode="replay")
    assert len(c2) == 1
    assert c2.lookup("k1") == r
    assert c2.lookup("nope") is None


def test_line_format_is_compact_sorted_and_human_diffable(tmp_path) -> None:
    p = tmp_path / "c.jsonl"
    c = Cassette(str(p), mode="record")
    c.record("k", {"b": 1, "a": "x"}, _resp())
    c.record("k2", "raw text body", _resp())
    raw = p.read_text().splitlines()
    assert len(raw) == 2
    d = json.loads(raw[0])
    assert set(d) == {"key", "request", "response", "recorded_at"}
    assert set(d["response"]) == {"status", "headers", "body", "streamed", "cost_usd"}
    assert raw[0] == json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert d["request"] == {"a": "x", "b": 1}
    assert json.loads(raw[1])["request"] == "raw text body"
    assert "T" in d["recorded_at"]


def test_ordered_replay_then_repeat_last(tmp_path) -> None:
    p = tmp_path / "c.jsonl"
    c = Cassette(str(p), mode="record")
    for i in range(3):
        c.record("same", {}, _resp(f"r{i}"))
    c.record("other", {}, _resp("o"))
    r = Cassette(str(p), mode="replay")
    assert [r.lookup("same").body for _ in range(5)] == ["r0", "r1", "r2", "r2", "r2"]
    assert r.lookup("other").body == "o"
    assert r.lookup("other").body == "o"


def test_record_mode_truncates_and_creates_dirs(tmp_path) -> None:
    p = tmp_path / "deep" / "dir" / "c.jsonl"
    c = Cassette(str(p), mode="record")
    c.record("old", {}, _resp())
    assert p.exists()
    c2 = Cassette(str(p), mode="record")
    assert len(_lines(p)) == 1  # truncation is lazy: constructing alone never wipes a cassette
    assert len(c2) == 0
    c2.record("new", {}, _resp())
    assert [d["key"] for d in _lines(p)] == ["new"]


def test_replay_never_writes(tmp_path) -> None:
    p = tmp_path / "c.jsonl"
    Cassette(str(p), mode="record").record("k", {}, _resp())
    before = p.read_text()
    r = Cassette(str(p), mode="replay")
    assert not r.should_record()
    r.record("k2", {}, _resp())
    assert p.read_text() == before
    missing = tmp_path / "missing.jsonl"
    r2 = Cassette(str(missing), mode="replay")
    assert len(r2) == 0 and r2.lookup("k") is None
    assert not missing.exists()


def test_auto_mode_appends_and_replays(tmp_path) -> None:
    p = tmp_path / "c.jsonl"
    Cassette(str(p), mode="record").record("k1", {}, _resp("one"))
    a = Cassette(str(p), mode="auto")
    assert a.should_record()
    assert a.lookup("k1").body == "one"
    assert a.lookup("k2") is None
    a.record("k2", {}, _resp("two"))
    assert a.lookup("k2").body == "two"  # available right away
    assert [d["key"] for d in _lines(p)] == ["k1", "k2"]  # appended, not truncated
    assert len(Cassette(str(p), mode="replay")) == 2


def test_should_record_modes(tmp_path) -> None:
    assert Cassette(str(tmp_path / "a.jsonl"), mode="record").should_record()
    assert Cassette(str(tmp_path / "b.jsonl"), mode="auto").should_record()
    assert not Cassette(str(tmp_path / "c.jsonl"), mode="replay").should_record()


def test_sensitive_headers_stripped(tmp_path) -> None:
    p = tmp_path / "c.jsonl"
    c = Cassette(str(p), mode="record")
    headers = {
        "content-type": "application/json",
        "Set-Cookie": "session=abc",
        "x-api-key": "sk-ant-secret",
        "Authorization": "Bearer x",
        "openai-organization": "org",
        "x-ratelimit-remaining-tokens": "100",
        "request-id": "req_123",
        "x-request-id": "req_456",
    }
    c.record("k", {}, _resp(headers=headers))
    stored = _lines(p)[0]["response"]["headers"]
    assert stored == {
        "content-type": "application/json",
        "openai-organization": "org",
        "request-id": "req_123",
        "x-request-id": "req_456",
    }
    assert Cassette(str(p), mode="replay").lookup("k").headers == stored


def test_redact_request_helper() -> None:
    ant = "sk-ant-api03-" + "A" * 40
    oai = "sk-proj-" + "b" * 40
    req = {"messages": [{"role": "user", "content": f"my keys are {ant} and {oai}"}], "n": 1, "x": None}
    out = redact_request_for_cassette(req)
    s = json.dumps(out)
    assert ant not in s and oai not in s
    assert "[REDACTED" in s
    assert out["n"] == 1 and out["x"] is None
    assert req["messages"][0]["content"].startswith("my keys are sk-ant")  # original untouched
    assert redact_request_for_cassette(f"key={ant}") != f"key={ant}"
    # no false positives on normal words / short tokens
    assert redact_request_for_cassette("task-oriented risk-aware desk-bound") == "task-oriented risk-aware desk-bound"
    assert redact_request_for_cassette("sk-short") == "sk-short"


def test_record_applies_redaction(tmp_path) -> None:
    p = tmp_path / "c.jsonl"
    key = "sk-" + "Z9" * 20
    Cassette(str(p), mode="record").record("k", {"prompt": f"use {key}"}, _resp())
    assert key not in p.read_text()


def test_corrupted_line_skipped_with_warning(tmp_path) -> None:
    p = tmp_path / "c.jsonl"
    c = Cassette(str(p), mode="record")
    c.record("a", {}, _resp("A"))
    with p.open("a") as f:
        f.write("{this is not json\n")
        f.write('{"key": "nokeys"}\n')
        f.write("\n")
    with pytest.warns(UserWarning):
        Cassette(str(p), mode="auto").record("b", {}, _resp("B"))
    with pytest.warns(UserWarning, match="corrupted"):
        r = Cassette(str(p), mode="replay")
    assert len(r) == 2
    assert r.lookup("a").body == "A" and r.lookup("b").body == "B"


def test_concurrent_record(tmp_path) -> None:
    p = tmp_path / "c.jsonl"
    c = Cassette(str(p), mode="record")
    big = "x" * 20000  # large lines make interleaving likely if unlocked

    def work(n: int) -> None:
        for i in range(25):
            c.record(f"k{n}", {"i": i}, _resp(json.dumps({"n": n, "i": i, "pad": big})))

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(c) == 200
    lines = _lines(p)  # every line parses
    assert len(lines) == 200
    r = Cassette(str(p), mode="replay")
    for n in range(8):
        got = [json.loads(r.lookup(f"k{n}").body)["i"] for _ in range(25)]
        assert got == list(range(25))  # per-thread ordering preserved


def test_concurrent_lookup_cursor(tmp_path) -> None:
    p = tmp_path / "c.jsonl"
    c = Cassette(str(p), mode="record")
    for i in range(100):
        c.record("k", {}, _resp(str(i)))
    r = Cassette(str(p), mode="replay")
    seen: list[str] = []
    lock = threading.Lock()

    def work() -> None:
        for _ in range(25):
            b = r.lookup("k").body
            with lock:
                seen.append(b)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(seen, key=int) == [str(i) for i in range(100)]  # each recording served exactly once


def test_auto_append_after_truncated_last_line(tmp_path) -> None:
    p = tmp_path / "c.jsonl"
    Cassette(str(p), mode="record").record("a", {}, _resp("A"))
    with p.open("a") as f:
        f.write('{"key": "half')  # crash mid-write, no newline
    with pytest.warns(UserWarning):
        Cassette(str(p), mode="auto").record("b", {}, _resp("B"))
    with pytest.warns(UserWarning):
        r = Cassette(str(p), mode="replay")
    assert r.lookup("a").body == "A" and r.lookup("b").body == "B"
