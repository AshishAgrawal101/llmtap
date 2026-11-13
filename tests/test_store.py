"""Tests for llmtap.store.Store."""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime

import pytest

from llmtap.models import Exchange, Usage
from llmtap.store import Store


def mk(**kw) -> Exchange:
    base = dict(provider="anthropic", style="anthropic", path="/v1/messages", model="claude-sonnet-4-5",
                request_body='{"messages":[{"role":"user","content":"hi"}]}', status=200,
                response_body='{"ok":true}', usage=Usage(10, 5), cost_usd=0.01, latency_ms=100.0,
                output_text="hello there")
    base.update(kw)
    return Exchange(**base)


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def test_roundtrip_exact(store):
    ex = mk(
        ts=1700000000.123456, method="POST", streamed=True, ttft_ms=12.5,
        usage=Usage(input_tokens=11, output_tokens=22, cache_read_tokens=3, cache_write_tokens=4, estimated=True),
        saved_usd=0.5, source="cache", tags={"session": "s1", "feature": "ünïcode"},
        guard_findings=[{"kind": "aws_access_key", "path": "messages[0].content", "preview": "AKIA…MPLE",
                         "start": 1, "end": 21}],
        error="boom", output_text="text ✓",
    )
    store.record(ex)
    got = store.get(ex.id)
    assert got == ex
    assert got.to_dict() == ex.to_dict()


def test_roundtrip_defaults_and_nones(store):
    ex = Exchange()
    store.record(ex)
    got = store.get(ex.id)
    assert got == ex
    assert got.model is None and got.ttft_ms is None and got.error is None
    assert got.usage.estimated is False and got.streamed is False


def test_record_replaces_by_id(store):
    ex = mk(status=0)
    store.record(ex)
    ex.status = 200
    ex.output_text = "updated"
    store.record(ex)
    assert store.count() == 1
    assert store.get(ex.id).output_text == "updated"


def test_get_prefix(store):
    a = mk(id="abcdef0001")
    b = mk(id="abcdef0002")
    c = mk(id="zzzz1234")
    for e in (a, b, c):
        store.record(e)
    assert store.get("abcdef0001") == a
    assert store.get("zzzz").id == c.id
    assert store.get("abcdef000") is None  # ambiguous
    assert store.get("abcdef0002")
    assert store.get("zzz") is None  # too short for prefix lookup
    assert store.get("nope") is None
    assert store.get("") is None
    # LIKE wildcards in the prefix are literal
    assert store.get("abc%") is None
    assert store.get("____") is None


def test_list_order_and_paging(store):
    now = time.time()
    exs = [mk(ts=now + i) for i in range(10)]
    for e in reversed(exs):
        store.record(e)
    got = store.list()
    assert [e.id for e in got] == [e.id for e in reversed(exs)]
    page = store.list(limit=3, offset=2)
    assert [e.id for e in page] == [exs[7].id, exs[6].id, exs[5].id]
    assert store.list(limit=0) == []
    assert len(store.list(limit=100, offset=8)) == 2


def test_list_filters(store):
    now = time.time()
    store.record(mk(ts=now - 100, provider="openai", style="openai", model="gpt-4o", path="/v1/chat/completions",
                    request_body='{"q":"translate to French"}', output_text="Bonjour"))
    store.record(mk(ts=now - 50, model="claude-haiku-4-5", source="cache", output_text="cached answer"))
    store.record(mk(ts=now, model="claude-sonnet-4-5", source="upstream", request_body="SQL query please"))
    assert {e.model for e in store.list(provider="openai")} == {"gpt-4o"}
    assert len(store.list(provider="anthropic")) == 2
    assert [e.model for e in store.list(model="claude-haiku-4-5")] == ["claude-haiku-4-5"]
    assert [e.source for e in store.list(source="cache")] == ["cache"]
    assert len(store.list(since=now - 60)) == 2
    # q searches request_body / output_text / model / path, case-insensitively
    assert [e.model for e in store.list(q="french")] == ["gpt-4o"]
    assert [e.model for e in store.list(q="BONJOUR")] == ["gpt-4o"]
    assert [e.model for e in store.list(q="haiku")] == ["claude-haiku-4-5"]
    assert [e.model for e in store.list(q="chat/completions")] == ["gpt-4o"]
    assert [e.model for e in store.list(q="sql")] == ["claude-sonnet-4-5"]
    assert store.list(q="100%") == []
    assert len(store.list(q="claude", provider="anthropic", since=now - 10)) == 1


def test_list_q_literal_wildcards(store):
    store.record(mk(request_body="discount 100% off", output_text=""))
    store.record(mk(request_body="discount 1000 off", output_text=""))
    assert len(store.list(q="100%")) == 1
    assert len(store.list(q="a_b")) == 0


def test_stats_totals(store):
    now = time.time()
    store.record(mk(ts=now, usage=Usage(100, 50), cost_usd=1.0, latency_ms=100))
    store.record(mk(ts=now, usage=Usage(10, 5), cost_usd=0.0, saved_usd=0.25, source="cache", latency_ms=1))
    store.record(mk(ts=now, cost_usd=0.0, source="cassette", latency_ms=1))
    store.record(mk(ts=now, cost_usd=0.0, status=400, source="blocked", latency_ms=0,
                    guard_findings=[{"kind": "a"}, {"kind": "b"}]))
    store.record(mk(ts=now, cost_usd=0.0, status=429, source="budget", latency_ms=0))
    store.record(mk(ts=now, cost_usd=0.0, status=429, source="loop", latency_ms=0))
    store.record(mk(ts=now, cost_usd=0.0, status=502, source="error", latency_ms=0))
    store.record(mk(ts=now, cost_usd=0.5, status=500, latency_ms=98, guard_findings=[{"kind": "c"}]))
    store.record(mk(ts=now, cost_usd=0.0, status=0, source="error", latency_ms=0))
    s = store.stats()
    assert s["requests"] == 9
    assert s["cost_usd"] == pytest.approx(1.5 + 0.01 * 0)  # cost_usd overridden everywhere
    assert s["saved_usd"] == pytest.approx(0.25)
    assert s["cache_hits"] == 2
    assert s["blocked"] == 3
    # real failures only (502 error, 500 upstream, status-0 error); deliberate blocks are not errors
    assert s["errors"] == 3
    assert s["guard_findings"] == 3
    assert s["input_tokens"] == 100 + 10 + 7 * 10
    assert s["output_tokens"] == 50 + 5 + 7 * 5
    assert s["avg_latency_ms"] == pytest.approx((100 + 98) / 2)  # upstream calls only
    assert s["groups"] == []


def test_stats_empty(store):
    s = store.stats()
    assert s == {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "saved_usd": 0.0,
                 "avg_latency_ms": 0.0, "errors": 0, "cache_hits": 0, "blocked": 0, "guard_findings": 0,
                 "groups": []}
    assert store.stats(group_by="model")["groups"] == []
    assert store.stats(group_by="tag:x")["groups"] == []


def test_stats_since(store):
    now = time.time()
    store.record(mk(ts=now - 1000, cost_usd=5.0))
    store.record(mk(ts=now, cost_usd=1.0))
    assert store.stats(since=now - 10)["requests"] == 1
    assert store.stats(since=now - 10)["cost_usd"] == pytest.approx(1.0)
    assert store.stats()["cost_usd"] == pytest.approx(6.0)


def test_stats_group_by_model_provider_source(store):
    store.record(mk(model="a", cost_usd=1.0, usage=Usage(1, 2)))
    store.record(mk(model="a", cost_usd=1.0, usage=Usage(1, 2)))
    store.record(mk(model="b", cost_usd=5.0, provider="openai", source="cache"))
    store.record(mk(model=None, cost_usd=0.0, provider="openai", source="error"))
    g = store.stats(group_by="model")["groups"]
    assert [x["key"] for x in g] == ["b", "a", None]
    a = g[1]
    assert a == {**a, "requests": 2, "cost_usd": pytest.approx(2.0), "input_tokens": 2, "output_tokens": 4}
    g = store.stats(group_by="provider")["groups"]
    assert [(x["key"], x["requests"]) for x in g] == [("openai", 2), ("anthropic", 2)]
    g = store.stats(group_by="source")["groups"]
    assert [x["key"] for x in g] == ["cache", "upstream", "error"]
    for x in g:
        assert set(x) >= {"key", "requests", "cost_usd", "input_tokens", "output_tokens"}


def test_stats_group_by_day_hour_localtime(store):
    t1 = datetime(2026, 3, 1, 23, 30).timestamp()  # local time
    t2 = datetime(2026, 3, 2, 0, 15).timestamp()
    t3 = datetime(2026, 3, 2, 0, 45).timestamp()
    store.record(mk(ts=t3, cost_usd=0.1))
    store.record(mk(ts=t1, cost_usd=9.0))
    store.record(mk(ts=t2, cost_usd=0.2))
    days = store.stats(group_by="day")["groups"]
    assert [(d["key"], d["requests"]) for d in days] == [("2026-03-01", 1), ("2026-03-02", 2)]
    assert days[1]["cost_usd"] == pytest.approx(0.3)
    hours = store.stats(group_by="hour")["groups"]
    assert [(h["key"], h["requests"]) for h in hours] == [("2026-03-01 23:00", 1), ("2026-03-02 00:00", 2)]


def test_stats_group_by_tag(store):
    store.record(mk(tags={"feature": "search"}, cost_usd=1.0))
    store.record(mk(tags={"feature": "search", "session": "x"}, cost_usd=1.0))
    store.record(mk(tags={"feature": "chat"}, cost_usd=3.0))
    store.record(mk(tags={}, cost_usd=0.5))
    store.record(mk(tags={"we\"ird.key": "v"}, cost_usd=0.1))
    g = store.stats(group_by="tag:feature")["groups"]
    assert [(x["key"], x["requests"]) for x in g] == [("chat", 1), ("search", 2), (None, 2)]
    g = store.stats(group_by="tag:session")["groups"]
    assert ("x", 1) in [(x["key"], x["requests"]) for x in g]
    g = store.stats(group_by='tag:we"ird.key')["groups"]
    assert ("v", 1) in [(x["key"], x["requests"]) for x in g]


def test_stats_bad_group_by(store):
    with pytest.raises(ValueError):
        store.stats(group_by="bogus")
    with pytest.raises(ValueError):
        store.stats(group_by="tag:")


def test_spent_since(store):
    now = time.time()
    store.record(mk(ts=now - 100, cost_usd=10.0))
    store.record(mk(ts=now, cost_usd=1.0))
    store.record(mk(ts=now, cost_usd=2.0, source="cache"))
    store.record(mk(ts=now, cost_usd=4.0, source="cassette"))
    assert store.spent_since(now - 10) == pytest.approx(1.0)
    assert store.spent_since(0) == pytest.approx(11.0)
    assert store.spent_since(now + 100) == 0.0


def test_timeseries(store):
    base = 1_700_000_000 - (1_700_000_000 % 3600)
    store.record(mk(ts=base + 10, cost_usd=1.0, usage=Usage(10, 5)))
    store.record(mk(ts=base + 20, cost_usd=2.0, usage=Usage(1, 1)))
    store.record(mk(ts=base + 3600 * 2 + 5, cost_usd=0.5, usage=Usage(0, 3)))
    store.record(mk(ts=base - 7200, cost_usd=100.0))
    ts = store.timeseries(since=base)
    assert ts == [
        {"t": base, "requests": 2, "cost_usd": pytest.approx(3.0), "tokens": 17},
        {"t": base + 7200, "requests": 1, "cost_usd": pytest.approx(0.5), "tokens": 3},
    ]
    ts60 = store.timeseries(since=base, bucket_s=60)
    assert [x["t"] for x in ts60] == [base, base + 7200]
    assert store.timeseries(since=base + 10**6) == []


def test_count_and_clear(store):
    for _ in range(5):
        store.record(mk())
    assert store.count() == 5
    store.clear()
    assert store.count() == 0
    assert store.list() == []


def test_close_idempotent_and_raises_after(tmp_path):
    s = Store(str(tmp_path / "a.db"))
    s.close()
    s.close()
    with pytest.raises(sqlite3.ProgrammingError):
        s.count()


def test_file_db_persists_wal_and_creates_dirs(tmp_path):
    p = tmp_path / "nested" / "dir" / "llmtap.db"
    s = Store(str(p))
    ex = mk()
    s.record(ex)
    mode = s._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    s.close()
    assert p.exists()
    s2 = Store(str(p))
    assert s2.get(ex.id) == ex
    assert s2.schema_version() >= 1
    s2.close()


def test_migration_adds_missing_columns(tmp_path):
    p = tmp_path / "old.db"
    c = sqlite3.connect(str(p))
    c.execute("CREATE TABLE exchanges (id TEXT PRIMARY KEY, ts REAL NOT NULL DEFAULT 0, provider TEXT, model TEXT)")
    c.execute("INSERT INTO exchanges (id, ts, provider, model) VALUES ('oldrow01', 1.0, 'openai', 'gpt-4o')")
    c.commit()
    c.close()
    s = Store(str(p))
    old = s.get("oldrow01")
    assert old is not None and old.model == "gpt-4o" and old.tags == {} and old.guard_findings == []
    assert old.output_text == "" and old.source == "upstream"
    ex = mk()
    s.record(ex)
    assert s.get(ex.id) == ex
    assert s.stats()["requests"] == 2
    s.close()


def test_index_on_ts(store):
    idx = [r[1] for r in store._conn.execute("PRAGMA index_list(exchanges)")]
    cols = {store._conn.execute(f"PRAGMA index_info({i})").fetchone()[2] for i in idx}
    assert "ts" in cols


def test_concurrent_records_and_reads(tmp_path):
    s = Store(str(tmp_path / "c.db"))
    n_threads, per = 16, 50
    errors: list[BaseException] = []
    start = threading.Barrier(n_threads + 2)

    def writer(i: int) -> None:
        try:
            start.wait()
            for j in range(per):
                s.record(mk(id=f"t{i:02d}-{j:03d}", cost_usd=0.01, tags={"thread": str(i)}))
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    def reader() -> None:
        try:
            start.wait()
            for _ in range(50):
                s.list(limit=10)
                s.stats(group_by="model")
                s.spent_since(0)
                s.count()
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    assert s.count() == n_threads * per
    st = s.stats(group_by="tag:thread")
    assert st["cost_usd"] == pytest.approx(n_threads * per * 0.01)
    assert len(st["groups"]) == n_threads
    assert all(g["requests"] == per for g in st["groups"])
    s.close()


def test_concurrent_memory_store():
    s = Store(":memory:")
    threads = [threading.Thread(target=lambda: [s.record(mk()) for _ in range(100)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert s.count() == 800
    s.close()
