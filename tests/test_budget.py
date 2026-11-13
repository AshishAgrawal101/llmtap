"""Tests for llmtap.budget."""
from __future__ import annotations

import threading
from datetime import datetime

import pytest

from llmtap.budget import Budget, Verdict, local_midnight
from llmtap.config import Config


class FakeClock:
    def __init__(self, t: float = 1_700_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class FakeSpent:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.calls: list[float] = []

    def __call__(self, since: float) -> float:
        self.calls.append(since)
        return self.value


def make(**cfg_kw):
    cfg = Config(**cfg_kw)
    clock = FakeClock()
    spent = FakeSpent()
    return Budget(cfg, spent, clock=clock), cfg, clock, spent


def test_verdict_defaults():
    v = Verdict(True)
    assert v.allowed and v.reason == "" and v.kind == ""


def test_everything_off_allows_and_never_queries_store():
    b, _, _, spent = make()
    for _ in range(100):
        assert b.check("k").allowed
    assert spent.calls == []
    assert b.tracked_keys() == 0


# ---------------------------------------------------------------- daily budget
def test_daily_budget_blocks_when_reached():
    b, _, _, spent = make(budget_daily_usd=5.0)
    spent.value = 4.99
    assert b.check("k").allowed
    b2, _, _, spent2 = make(budget_daily_usd=5.0)
    spent2.value = 5.03
    v = b2.check("k")
    assert not v.allowed and v.kind == "budget"
    assert "$5.00" in v.reason and "$5.03" in v.reason
    assert "--budget-daily" in v.reason and "midnight" in v.reason
    assert v.reason.startswith("llmtap:")


def test_daily_budget_exactly_equal_blocks():
    b, _, _, spent = make(budget_daily_usd=5.0)
    spent.value = 5.0
    assert not b.check("k").allowed


def test_daily_uses_local_midnight():
    b, _, clock, spent = make(budget_daily_usd=1.0)
    b.check("k")
    expected = datetime.fromtimestamp(clock.t).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    assert spent.calls == [expected]
    assert spent.calls[0] <= clock.t < spent.calls[0] + 86400 + 3600


def test_local_midnight_helper():
    now = datetime(2026, 3, 10, 15, 30, 12).timestamp()
    assert local_midnight(now) == datetime(2026, 3, 10).timestamp()
    assert local_midnight(datetime(2026, 3, 10).timestamp()) == datetime(2026, 3, 10).timestamp()


def test_spent_since_cached_for_about_a_second():
    b, _, clock, spent = make(budget_daily_usd=10.0)
    for _ in range(50):
        b.check("k")
    assert len(spent.calls) == 1
    clock.advance(0.5)
    b.check("k")
    assert len(spent.calls) == 1
    clock.advance(0.6)
    b.check("k")
    assert len(spent.calls) == 2


def test_cache_refreshes_and_picks_up_new_spend():
    b, _, clock, spent = make(budget_daily_usd=1.0)
    assert b.check("k").allowed
    spent.value = 2.0
    assert b.check("k").allowed  # still cached
    clock.advance(1.5)
    assert not b.check("k").allowed


def test_add_spend_bumps_cached_daily_figure():
    b, _, _, spent = make(budget_daily_usd=1.0)
    spent.value = 0.5
    assert b.check("k").allowed
    b.add_spend(0.6)  # within the cache TTL, still counts immediately
    v = b.check("k")
    assert not v.allowed and v.kind == "budget"


def test_cache_invalidated_at_midnight():
    b, _, clock, spent = make(budget_daily_usd=1.0)
    midnight_tomorrow = local_midnight(clock.t) + 86400 + 7200
    clock.t = local_midnight(midnight_tomorrow) - 0.2
    spent.value = 2.0
    assert not b.check("k").allowed
    first_since = spent.calls[-1]
    clock.advance(0.4)  # crosses midnight, within TTL
    spent.value = 0.0
    assert b.check("k").allowed
    assert spent.calls[-1] > first_since


def test_spent_since_error_fails_open():
    def boom(_since: float) -> float:
        raise RuntimeError("db locked")
    b = Budget(Config(budget_daily_usd=1.0), boom, clock=FakeClock())
    assert b.check("k").allowed


def test_daily_spent_accessor():
    b, _, _, spent = make(budget_daily_usd=1.0)
    spent.value = 0.25
    assert b.daily_spent() == pytest.approx(0.25)


# -------------------------------------------------------------- session budget
def test_session_budget():
    b, _, _, spent = make(budget_session_usd=1.0)
    assert b.session_spent() == 0.0
    b.add_spend(0.4)
    b.add_spend(0.5)
    assert b.session_spent() == pytest.approx(0.9)
    assert b.check("k").allowed
    b.add_spend(0.1)
    v = b.check("k")
    assert not v.allowed and v.kind == "budget"
    assert "session" in v.reason and "--budget-session" in v.reason
    assert "$1.00" in v.reason
    assert spent.calls == []  # daily disabled -> store never queried


def test_session_spend_ignores_zero_and_negative():
    b, _, _, _ = make()
    b.add_spend(0.0)
    b.add_spend(-3.0)
    assert b.session_spent() == 0.0


def test_daily_checked_before_session():
    b, _, _, spent = make(budget_daily_usd=1.0, budget_session_usd=1.0)
    spent.value = 5.0
    b.add_spend(5.0)
    assert "daily" in b.check("k").reason


def test_small_amount_formatting():
    b, _, _, _ = make(budget_session_usd=0.005)
    b.add_spend(0.0062)
    v = b.check("k")
    assert "$0.0050" in v.reason and "$0.0062" in v.reason


# ---------------------------------------------------------------- loop detector
def test_loop_threshold_allows_n_blocks_n_plus_one():
    b, _, _, _ = make(loop_threshold=4, loop_window_s=60)
    for _ in range(4):
        assert b.check("same").allowed
    v = b.check("same")
    assert not v.allowed and v.kind == "loop"
    assert "5 times in 60s" in v.reason
    assert "--loop-threshold 0" in v.reason
    assert "runaway loop" in v.reason


def test_loop_threshold_one():
    b, _, _, _ = make(loop_threshold=1)
    assert b.check("k").allowed
    assert not b.check("k").allowed


def test_loop_keys_are_independent():
    b, _, _, _ = make(loop_threshold=2)
    assert b.check("a").allowed
    assert b.check("a").allowed
    assert b.check("b").allowed
    assert not b.check("a").allowed
    assert b.check("b").allowed
    assert not b.check("b").allowed


def test_loop_window_expiry():
    b, _, clock, _ = make(loop_threshold=3, loop_window_s=10)
    for _ in range(3):
        assert b.check("k").allowed
        clock.advance(1)
    assert not b.check("k").allowed  # 4 within 10s
    clock.advance(11)  # everything ages out
    assert b.check("k").allowed


def test_sliding_window_spaced_requests_never_block():
    b, _, clock, _ = make(loop_threshold=2, loop_window_s=10)
    for _ in range(20):
        assert b.check("k").allowed
        clock.advance(6)  # at most 2 within any 10s window


def test_blocked_attempts_still_recorded():
    b, _, clock, _ = make(loop_threshold=2, loop_window_s=10)
    b.check("k"); b.check("k")
    for _ in range(5):
        clock.advance(4)
        assert not b.check("k").allowed  # hammering keeps it blocked
    clock.advance(11)
    assert b.check("k").allowed


def test_loop_disabled_with_zero():
    b, _, _, _ = make(loop_threshold=0)
    for _ in range(1000):
        assert b.check("k").allowed
    assert b.tracked_keys() == 0


def test_lru_bounds_tracked_keys():
    b = Budget(Config(loop_threshold=2), FakeSpent(), clock=FakeClock(), max_keys=100)
    for i in range(1000):
        b.check(f"key{i}")
    assert b.tracked_keys() == 100
    # most recently used survive; oldest evicted (forgotten -> allowed again)
    b.check("key999")
    assert not b.check("key999").allowed
    assert b.check("key0").allowed


def test_lru_touch_keeps_hot_key():
    b = Budget(Config(loop_threshold=3), FakeSpent(), clock=FakeClock(), max_keys=3)
    b.check("hot")
    b.check("a"); b.check("b")
    b.check("hot")  # refresh
    b.check("c")    # evicts "a", not "hot"
    b.check("hot")
    assert not b.check("hot").allowed


def test_budget_denial_takes_priority_over_loop():
    b, _, _, _ = make(loop_threshold=1, budget_session_usd=0.01)
    b.add_spend(1.0)
    b.check("k")
    assert b.check("k").kind == "budget"


def test_config_changes_take_effect_live():
    b, cfg, _, _ = make(loop_threshold=0)
    assert b.check("k").allowed
    cfg.loop_threshold = 1
    assert b.check("k").allowed
    assert not b.check("k").allowed
    cfg.loop_threshold = 0
    assert b.check("k").allowed


def test_reset():
    b, _, _, _ = make(loop_threshold=1, budget_session_usd=1.0)
    b.add_spend(2.0)
    b.check("k")
    b.reset()
    assert b.session_spent() == 0.0
    assert b.tracked_keys() == 0
    assert b.check("k").allowed


# ------------------------------------------------------------------ thread safety
def test_thread_safety_session_accumulator_and_loop_counts():
    b = Budget(Config(loop_threshold=100, loop_window_s=60), FakeSpent())
    results: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        local = []
        for _ in range(50):
            b.add_spend(0.01)
            local.append(b.check("shared").allowed)
        with lock:
            results.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert b.session_spent() == pytest.approx(4.0)
    assert sum(results) == 100  # exactly threshold allowed, rest blocked
    assert len(results) == 400


def test_real_clock_default():
    b = Budget(Config(loop_threshold=2), FakeSpent())
    assert b.check("k").allowed and b.check("k").allowed
    assert not b.check("k").allowed
