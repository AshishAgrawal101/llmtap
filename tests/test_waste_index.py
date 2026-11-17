"""Index-backed waste finder and keyset paging (big-database performance work)."""
from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import pytest

from llmtap import report
from llmtap.cache import cache_key
from llmtap.models import Exchange, Usage
from llmtap.store import Store

from test_report import BASE, DAY, ex


class _Plain:
    """A store without the indexed helpers: forces report's reference (row-by-row) path."""
    def __init__(self, s: Store) -> None:
        self._s = s

    def list(self, **kw):
        return self._s.list(**kw)

    def get(self, i):
        return self._s.get(i)


def _keyed(e: Exchange) -> Exchange:
    e.request_key = cache_key(e.provider, e.path, e.request_body.encode())
    return e


def _mixed_store(seed: int = 7) -> Store:
    rnd = random.Random(seed)
    s = Store(":memory:")
    prompts = [f"p{i}" for i in range(12)]
    for i in range(400):
        e = ex(rnd.choice(prompts), ts=BASE - rnd.random() * 5 * DAY, cost=round(rnd.random(), 4),
               model=rnd.choice(["gpt-4o", "gpt-5", None]),
               source=rnd.choice(["upstream"] * 6 + ["cache", "loop", "blocked"]),
               status=rnd.choice([200] * 8 + [500, 429]))
        if rnd.random() < 0.6:          # rows recorded after the upgrade carry their key
            _keyed(e)
        s.record(e)
    return s


@pytest.mark.parametrize("since", [None, BASE - 2 * DAY, BASE + DAY])
def test_indexed_waste_matches_reference_on_mixed_rows(since):
    s = _mixed_store()
    fast = report.find_waste(s, since)
    slow = report.find_waste(_Plain(s), since)
    assert fast["duplicate_requests"] == slow["duplicate_requests"]
    assert fast["duplicate_groups"] == slow["duplicate_groups"]
    assert fast["savings_usd"] == pytest.approx(slow["savings_usd"])
    norm = lambda w: [{**g, "savings_usd": round(g["savings_usd"], 9), "total_cost_usd": round(g["total_cost_usd"], 9)}
                      for g in w["groups"]]
    assert norm(fast) == norm(slow)
    s.close()


def test_legacy_row_duplicating_a_single_keyed_row_counts():
    s = Store(":memory:")
    s.record(ex("same", ts=BASE, cost=0.5))                      # old row, no key
    s.record(_keyed(ex("same", ts=BASE + 10, cost=0.25)))        # new row, keyed, only one of its kind
    w = report.find_waste(s)
    assert (w["duplicate_requests"], w["savings_usd"]) == (1, pytest.approx(0.25))
    assert w["groups"][0]["preview"] == "same"


def test_iter_light_pages_by_keyset_and_skips_heavy_columns():
    s = Store(":memory:")
    for i in range(25):
        s.record(Exchange(ts=BASE + (i // 5), response_body="R" * 50, output_text="O", request_body=f"r{i}"))
    got = list(s.iter_light(page=4))                              # equal timestamps across page breaks
    assert len(got) == 25 and len({e.id for e in got}) == 25
    assert [e.ts for e in got] == sorted((e.ts for e in got), reverse=True)
    assert all(e.response_body == "" and e.output_text == "" and e.request_body for e in got)
    assert [e.id for e in got] == [e.id for e in s.list(limit=-1)]
    assert len(list(s.iter_light(since=BASE + 3))) == 10


def test_request_key_round_trips_and_old_databases_upgrade(tmp_path: Path):
    db = tmp_path / "old.db"
    c = sqlite3.connect(db)                                       # a schema-1 table, without request_key
    c.execute("CREATE TABLE exchanges (id TEXT PRIMARY KEY, ts REAL NOT NULL, provider TEXT, request_body TEXT, "
              "source TEXT NOT NULL DEFAULT 'upstream', status INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0)")
    c.execute("INSERT INTO exchanges (id, ts, provider, request_body, status, cost_usd) VALUES ('a', ?, 'openai', '{}', 200, 1)",
              (BASE,))
    c.commit()
    c.close()
    s = Store(str(db))
    assert s.schema_version() == 2
    assert s.get("a").request_key is None
    e = _keyed(ex("x"))
    s.record(e)
    assert s.get(e.id).request_key == e.request_key
    idx = {r[1] for r in s._c().execute("PRAGMA index_list(exchanges)")}
    assert "idx_exchanges_waste" in idx
    s.close()


def test_iter_light_ignores_rows_recorded_mid_scan():
    s = Store(":memory:")
    for i in range(10):
        s.record(Exchange(ts=BASE + i, request_body=f"r{i}"))
    seen = []
    for e in s.iter_light(page=3):
        seen.append(e.id)
        s.record(Exchange(ts=BASE + 1000 + len(seen), request_body="new"))   # newer than the cursor
    assert len(seen) == len(set(seen)) == 10


def test_legacy_rows_on_prefix_routes_group_with_new_rows():
    s = Store(":memory:")
    old = ex("same", ts=BASE, cost=0.5, provider="groq", path="/groq/v1/chat/completions")    # pre-upgrade row
    new = ex("same", ts=BASE + 5, cost=0.5, provider="groq", path="/groq/v1/chat/completions")
    new.request_key = cache_key("groq", "/v1/chat/completions", new.request_body.encode())    # as the proxy keys it
    s.record(old)
    s.record(new)
    old2 = ex("bare", ts=BASE, cost=0.2, path="/chat/completions")                             # openai without /v1
    new2 = ex("bare", ts=BASE + 5, cost=0.2, path="/chat/completions")
    new2.request_key = cache_key("openai", "/v1/chat/completions", new2.request_body.encode())
    s.record(old2)
    s.record(new2)
    assert report.find_waste(s)["duplicate_requests"] == 2
    assert report.build_report(s, since=None)["waste"]["duplicate_requests"] == 2


def test_iter_legacy_waste_pages_without_losing_rows():
    s = Store(":memory:")
    for i in range(11):
        s.record(ex(f"p{i}", ts=BASE + (i // 4)))
    rows = list(s.iter_legacy_waste(page=3))
    assert len(rows) == 11 and len({r[0] for r in rows}) == 11


def test_migration_tolerates_a_column_another_process_just_added(tmp_path: Path):
    db = tmp_path / "race.db"
    Store(str(db)).close()

    class Stale:
        """A connection whose first table_info misses request_key, as if read just before another process
        added it."""
        def __init__(self, c):
            self.c, self.stale = c, True

        def execute(self, sql, *a):
            cur = self.c.execute(sql, *a)
            if self.stale and sql.startswith("PRAGMA table_info"):
                self.stale = False
                return [r for r in cur.fetchall() if r[1] != "request_key"]
            return cur

    c = sqlite3.connect(db)
    Store._migrate(Stale(c))          # used to raise "duplicate column name: request_key"
    c.close()


def test_read_only_old_database_still_opens(tmp_path: Path):
    import os
    import stat
    db = tmp_path / "ro.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE exchanges (id TEXT PRIMARY KEY, ts REAL NOT NULL, provider TEXT NOT NULL DEFAULT 'openai', "
              "path TEXT NOT NULL DEFAULT '', request_body TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT 'upstream', "
              "status INTEGER NOT NULL DEFAULT 200, cost_usd REAL NOT NULL DEFAULT 0)")
    for i, t in enumerate((BASE, BASE + 1)):
        c.execute("INSERT INTO exchanges (id, ts, request_body, cost_usd) VALUES (?, ?, '{\"m\":1}', 0.5)", (f"id{i}", t))
    c.commit()
    c.close()
    os.chmod(db, stat.S_IREAD)
    try:
        s = Store(str(db))
        assert len(s.list(limit=10)) == 2 and s.get("id0").request_key is None
        assert report.find_waste(s)["duplicate_requests"] == 1
        assert s.stats()["requests"] == 2
        s.close()
    finally:
        os.chmod(db, stat.S_IREAD | stat.S_IWRITE)


def test_iter_legacy_waste_window_and_all_time_agree():
    s = Store(":memory:")
    for i in range(13):
        s.record(ex(f"p{i}", ts=BASE + i))
    every = {r[0] for r in s.iter_legacy_waste(page=4)}
    window = {r[0] for r in s.iter_legacy_waste(since=BASE + 6, page=4)}
    assert len(every) == 13 and len(window) == 7 and window <= every
