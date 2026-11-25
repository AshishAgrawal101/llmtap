"""Budget warnings before a cap is hit (50 / 80 / 95% by default), and optional desktop notifications.

A hard cap that only acts once the money is spent tells you too late. :class:`Alerts` checks every priced call
against the daily and per-run budgets and against the caps of the stored key it used, and warns once per
threshold per window: in the terminal, on the dashboard (an event on the bus) and, with ``notify``, as a
desktop notification (no dependencies: ``osascript`` on macOS, ``notify-send`` on Linux, PowerShell on Windows).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from .budget import local_midnight

__all__ = ["Alerts", "parse_thresholds", "notify"]


def parse_thresholds(text: str) -> list[float]:
    """``"50,80,95"`` -> [50.0, 80.0, 95.0] (sorted, 0 < p < 100). Invalid parts raise ValueError."""
    out: list[float] = []
    for part in (text or "").replace(";", ",").split(","):
        part = part.strip().rstrip("%")
        if not part:
            continue
        v = float(part)
        if not 0 < v <= 100:
            raise ValueError(f"budget_warn: percentages must be between 0 and 100, got {part}")
        out.append(v)
    return sorted(set(out))


def notify(title: str, message: str) -> None:
    """Best-effort desktop notification in a background thread. Never raises, never blocks the caller."""
    # printable ASCII only: these strings can contain file names an agent chose
    clean = lambda s: "".join(c if 32 <= ord(c) < 127 else "?" for c in s)  # noqa: E731
    title, message = clean(title)[:120], clean(message)[:400]

    def run() -> None:
        try:
            if sys.platform == "darwin" and shutil.which("osascript"):
                esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')  # noqa: E731
                cmd = ["osascript", "-e", f'display notification "{esc(message)}" with title "{esc(title)}"']
            elif sys.platform.startswith("win"):
                ps = shutil.which("powershell") or shutil.which("pwsh")
                if not ps:
                    return
                # the text goes in through environment variables, never into the script itself
                script = (
                    "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; "
                    "$n = New-Object System.Windows.Forms.NotifyIcon; $n.Icon = [System.Drawing.SystemIcons]::Warning; "
                    "$n.BalloonTipTitle = $env:LLMTAP_NOTIFY_TITLE; $n.BalloonTipText = $env:LLMTAP_NOTIFY_TEXT; "
                    "$n.Visible = $true; $n.ShowBalloonTip(8000); Start-Sleep -Seconds 9; $n.Dispose()")
                cmd = [ps, "-NoProfile", "-NonInteractive", "-Command", script]
            elif shutil.which("notify-send"):
                cmd = ["notify-send", "--app-name=llmtap", title, message]
            else:
                return
            env = dict(os.environ, LLMTAP_NOTIFY_TITLE=title, LLMTAP_NOTIFY_TEXT=message)
            subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=20, check=False, env=env)
        except Exception:
            pass

    threading.Thread(target=run, name="llmtap-notify", daemon=True).start()


class Alerts:
    """Tracks which thresholds were already announced, per cap and window."""

    def __init__(self, thresholds: list[float], desktop: bool = False,
                 emit: Callable[[str, str, dict[str, Any]], None] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.thresholds = thresholds
        self.desktop = desktop
        self.emit = emit
        self.clock = clock
        self._lock = threading.Lock()
        self._done: dict[tuple[str, float], float] = {}      # (cap id, window start) -> highest % announced

    def _crossed(self, cap_id: str, window: float, pct: float) -> float | None:
        """The highest threshold ``pct`` reached that wasn't announced yet for this cap and window."""
        hit = [t for t in self.thresholds if pct >= t]
        if not hit:
            return None
        top = hit[-1]
        with self._lock:
            if self._done.get((cap_id, window), 0.0) >= top:
                return None
            self._done[(cap_id, window)] = top
            if len(self._done) > 1000:                       # old windows
                for k in list(self._done)[:500]:
                    self._done.pop(k, None)
        return top

    def check(self, caps: list[tuple[str, str, float, float, float]]) -> list[str]:
        """``caps``: (cap id, label, spent, limit, window start). Returns the warnings announced now."""
        said = []
        for cap_id, label, spent, limit, window in caps:
            if not limit or limit <= 0:
                continue
            pct = spent / limit * 100.0
            top = self._crossed(cap_id, window, pct)
            if top is None or pct >= 100.0:
                continue                                      # at 100% the cap itself refuses, loudly
            msg = f"{label}: ${spent:.2f} of ${limit:.2f} spent ({pct:.0f}%, warning at {top:g}%)"
            said.append(msg)
            self.announce("llmtap budget warning", msg, {"cap": cap_id, "percent": round(pct, 1),
                                                         "spent": spent, "limit": limit})
        return said

    def announce(self, title: str, message: str, data: dict[str, Any] | None = None) -> None:
        if self.emit is not None:
            try:
                self.emit(title, message, data or {})
            except Exception:
                pass
        if self.desktop:
            notify(title, message)

    @staticmethod
    def caps_for(tap: Any, ex: Any) -> list[tuple[str, str, float, float, float]]:
        """The caps a finished priced call counts against, with what is spent in each window now."""
        cfg = tap.cfg
        now = time.time()
        caps: list[tuple[str, str, float, float, float]] = []
        midnight = local_midnight(now)
        if cfg.budget_daily_usd:
            caps.append(("daily", "daily budget", tap.budget.daily_spent(), float(cfg.budget_daily_usd), midnight))
        if cfg.budget_session_usd:
            caps.append(("session", "this run's budget", tap.budget.session_spent(),
                         float(cfg.budget_session_usd), tap.started_at))
        info = ex.__dict__.get("_vault")
        if info is not None:
            from .vault import TAG
            try:
                k = tap.vault.get(info.name) or info
            except Exception:
                k = info
            if k.daily_usd:
                caps.append((f"key:{k.name}:day", f"stored key {k.name!r} daily cap",
                             tap.store.spent_by_tag(TAG, k.name, midnight), float(k.daily_usd), midnight))
            if k.total_usd:
                caps.append((f"key:{k.name}:total", f"stored key {k.name!r} total cap",
                             tap.store.spent_by_tag(TAG, k.name, k.created), float(k.total_usd), k.created))
        return caps
