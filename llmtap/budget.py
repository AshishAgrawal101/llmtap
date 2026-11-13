"""Spend caps (daily / per-session) and a runaway-agent loop breaker.

Loop threshold semantics: ``loop_threshold = N`` allows N identical requests (same key) within
``loop_window_s`` seconds and blocks the (N+1)th and every further one until older attempts age out
of the window. Every call to :meth:`Budget.check` records an attempt (blocked ones included), so a
stuck agent that keeps hammering stays blocked until it pauses for a full window.
``loop_threshold = 0`` disables the detector.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .config import Config

SPENT_CACHE_TTL_S = 1.0
DEFAULT_MAX_KEYS = 10_000
MAX_HITS_PER_KEY = 10_000   # bounds per-key memory under extreme hammering


@dataclass
class Verdict:
    """Outcome of a budget check. ``kind`` is "budget", "loop" or "" when allowed."""
    allowed: bool
    reason: str = ""
    kind: str = ""


def _usd(x: float) -> str:
    """Format dollars: 2 decimals normally, 4 for small non-zero amounts."""
    if x != 0 and abs(x) < 1:
        return f"${x:.4f}" if abs(x) < 0.01 else f"${x:.2f}"
    return f"${x:,.2f}"


def _fmt_window(seconds: float) -> str:
    s = int(seconds) if float(seconds).is_integer() else seconds
    return f"{s}s"


def local_midnight(now: float) -> float:
    """Epoch timestamp of the most recent local midnight at or before ``now``."""
    dt = datetime.fromtimestamp(now)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


class Budget:
    """Thread-safe spend caps + identical-request loop detector."""

    def __init__(
        self,
        cfg: Config,
        spent_since: Callable[[float], float],
        clock: Callable[[], float] = time.time,
        max_keys: int = DEFAULT_MAX_KEYS,
        spent_cache_ttl_s: float = SPENT_CACHE_TTL_S,
    ) -> None:
        self.cfg = cfg
        self._spent_since = spent_since
        self._clock = clock
        self._max_keys = max(1, int(max_keys))
        self._ttl = spent_cache_ttl_s
        self._lock = threading.Lock()
        self._session = 0.0
        self._seen: OrderedDict[str, deque[float]] = OrderedDict()
        # cached daily spend: (midnight it was computed for, fetched_at, value)
        self._daily_cache: tuple[float, float, float] | None = None

    # ------------------------------------------------------------------ spend
    def add_spend(self, usd: float) -> None:
        """Add upstream cost to the session accumulator (and the cached daily figure)."""
        if not usd or usd < 0:
            return
        with self._lock:
            self._session += usd
            if self._daily_cache is not None:
                m, at, v = self._daily_cache
                self._daily_cache = (m, at, v + usd)

    def session_spent(self) -> float:
        with self._lock:
            return self._session

    def daily_spent(self) -> float:
        """Spend since local midnight (cached for ~1s)."""
        now = self._clock()
        midnight = local_midnight(now)
        with self._lock:
            c = self._daily_cache
            if c is not None and c[0] == midnight and 0 <= now - c[1] < self._ttl:
                return c[2]
        try:
            value = float(self._spent_since(midnight) or 0.0)
        except Exception:  # fail open on storage errors: never block traffic because the DB hiccuped
            with self._lock:
                c = self._daily_cache
                return c[2] if c is not None and c[0] == midnight else 0.0
        with self._lock:
            self._daily_cache = (midnight, now, value)
        return value

    # ------------------------------------------------------------------ check
    def check(self, key: str) -> Verdict:
        """Decide whether a request (identified by its cache ``key``) may go upstream."""
        cfg = self.cfg
        count = self._record_attempt(key)

        daily_cap = float(cfg.budget_daily_usd or 0)
        if daily_cap > 0:
            spent = self.daily_spent()
            if spent >= daily_cap:
                return Verdict(False, (
                    f"llmtap: daily budget of {_usd(daily_cap)} reached ({_usd(spent)} spent today). "
                    "Raise it with --budget-daily or wait until midnight."), "budget")

        session_cap = float(cfg.budget_session_usd or 0)
        if session_cap > 0:
            spent = self.session_spent()
            if spent >= session_cap:
                return Verdict(False, (
                    f"llmtap: session budget of {_usd(session_cap)} reached ({_usd(spent)} spent this session). "
                    "Raise it with --budget-session or restart llmtap to start a new session."), "budget")

        threshold = int(cfg.loop_threshold or 0)
        if threshold > 0 and count > threshold:
            return Verdict(False, (
                f"llmtap: runaway loop detected — this exact request was sent {count} times in "
                f"{_fmt_window(cfg.loop_window_s)}. Your agent may be stuck. "
                "(Disable with --loop-threshold 0)"), "loop")
        return Verdict(True)

    def _record_attempt(self, key: str) -> int:
        """Record an attempt for ``key``; return attempts within the window including this one."""
        if int(self.cfg.loop_threshold or 0) <= 0:
            return 0
        now = self._clock()
        cutoff = now - float(self.cfg.loop_window_s or 0)
        with self._lock:
            dq = self._seen.get(key)
            if dq is None:
                dq = deque(maxlen=MAX_HITS_PER_KEY)
                self._seen[key] = dq
            else:
                self._seen.move_to_end(key)
            while dq and dq[0] <= cutoff:
                dq.popleft()
            dq.append(now)
            while len(self._seen) > self._max_keys:
                self._seen.popitem(last=False)
            return len(dq)

    def tracked_keys(self) -> int:
        """Number of request keys currently tracked by the loop detector."""
        with self._lock:
            return len(self._seen)

    def reset(self) -> None:
        """Forget session spend, loop history and cached daily spend."""
        with self._lock:
            self._session = 0.0
            self._seen.clear()
            self._daily_cache = None
