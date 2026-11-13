"""Pretty terminal output: ANSI colors, banner, per-request log lines, summary box, tables.

Colors are auto-disabled when ``NO_COLOR`` is set or the stream is not a TTY; ``FORCE_COLOR``
forces them on. Every layout helper measures *visible* width (ANSI codes stripped, wide glyphs
counted as two cells) so boxes and tables stay aligned with or without color.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
import unicodedata
from typing import IO, Any, Iterable

from llmtap import __version__

# --------------------------------------------------------------------------------------------
# color support

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

CODES: dict[str, str] = {
    "reset": "0", "bold": "1", "dim": "2", "italic": "3", "underline": "4", "inverse": "7",
    "black": "30", "red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35",
    "cyan": "36", "white": "37", "gray": "90", "grey": "90",
    "bright_red": "91", "bright_green": "92", "bright_yellow": "93", "bright_blue": "94",
    "bright_magenta": "95", "bright_cyan": "96", "bright_white": "97",
    "bg_red": "41", "bg_green": "42", "bg_yellow": "43", "bg_blue": "44", "bg_magenta": "45",
    "bg_cyan": "46", "bg_white": "47", "bg_gray": "100",
}


def color_enabled(stream: IO[Any] | None = None) -> bool:
    """Decide whether to emit ANSI codes for ``stream`` (default: stderr)."""
    if os.environ.get("NO_COLOR"):
        return False
    force = os.environ.get("FORCE_COLOR")
    if force is not None and force.strip().lower() not in ("", "0", "false", "no", "off"):
        return True
    stream = sys.stderr if stream is None else stream
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def paint(text: str, *styles: str, color: bool | None = None) -> str:
    """Wrap ``text`` in the named ANSI styles (``paint("x", "bold", "red")``)."""
    if color is None:
        color = color_enabled()
    if not color or not styles:
        return text
    codes = ";".join(CODES[s] for s in styles if s in CODES)
    if not codes:
        return text
    return f"\x1b[{codes}m{text}\x1b[0m"


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences."""
    return _ANSI_RE.sub("", text)


def _char_width(ch: str) -> int:
    if unicodedata.combining(ch) or ch in "​‍︎️":
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def vlen(text: str) -> int:
    """Visible width in terminal cells (ANSI stripped, wide glyphs = 2)."""
    return sum(_char_width(c) for c in strip_ansi(text))


def pad(text: str, width: int, align: str = "<") -> str:
    """Pad ``text`` to ``width`` visible cells. align: '<' left, '>' right, '^' center."""
    gap = max(0, width - vlen(text))
    if align == ">":
        return " " * gap + text
    if align == "^":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


def truncate(text: str, width: int, ellipsis: str = "…") -> str:
    """Truncate plain ``text`` to at most ``width`` visible cells (ANSI is stripped if cut)."""
    if width <= 0:
        return ""
    if vlen(text) <= width:
        return text
    plain = strip_ansi(text)
    out, used = [], 0
    limit = width - vlen(ellipsis)
    for ch in plain:
        w = _char_width(ch)
        if used + w > limit:
            break
        out.append(ch)
        used += w
    return "".join(out) + ellipsis


def term_width(default: int = 100) -> int:
    """Current terminal width (columns)."""
    try:
        return max(40, shutil.get_terminal_size((default, 24)).columns)
    except Exception:
        return default


# --------------------------------------------------------------------------------------------
# human formatting

def fmt_tokens(n: int | float | None) -> str:
    """1234 -> '1.2k', 340 -> '340', 2_500_000 -> '2.5M'."""
    if n is None:
        return "0"
    n = int(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n < 1000:
        return f"{sign}{n}"
    for div, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if n >= div:
            v = n / div
            s = f"{v:.1f}" if v < 100 else f"{v:.0f}"
            if s.endswith(".0"):
                s = s[:-2]
            return f"{sign}{s}{suffix}"
    return f"{sign}{n}"  # pragma: no cover


def fmt_ms(ms: float | None) -> str:
    """210 -> '210ms', 1240 -> '1.24s', 75000 -> '1m15s'."""
    if ms is None:
        return "-"
    ms = float(ms)
    if ms < 1:
        return f"{ms:.1f}ms" if ms > 0 else "0ms"
    if ms < 1000:
        return f"{ms:.0f}ms"
    s = ms / 1000
    if s < 60:
        return f"{s:.2f}s" if s < 10 else f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def fmt_usd(x: float | None) -> str:
    """Format dollars via pricing.format_usd (falls back to a local implementation)."""
    x = float(x or 0.0)
    try:
        from llmtap.pricing import format_usd
        return format_usd(x)
    except Exception:
        if x == 0:
            return "$0.00"
        if abs(x) < 1:
            return f"${x:.4f}"
        return f"${x:,.2f}"


def fmt_count(n: int | float | None) -> str:
    """Thousands separators: 12345 -> '12,345'."""
    return f"{int(n or 0):,}"


# --------------------------------------------------------------------------------------------
# boxes and tables

def box(lines: Iterable[str], title: str = "", width: int | None = None, color: bool | None = None,
        border: str = "cyan") -> str:
    """Draw a rounded unicode box around ``lines``. Width = widest line, capped at the terminal."""
    if color is None:
        color = color_enabled()
    lines = list(lines)
    cap = (width or term_width()) - 4
    inner = max([vlen(l) for l in lines] + [vlen(title) + 2 if title else 0, 20])
    inner = min(inner, max(20, cap))
    b = lambda s: paint(s, border, color=color)  # noqa: E731
    if title:
        t = f" {title} "
        top = b("╭─") + t + b("─" * max(0, inner - vlen(t)) + "─╮")
    else:
        top = b("╭" + "─" * (inner + 2) + "╮")
    out = [top]
    for l in lines:
        if vlen(l) > inner:
            l = truncate(l, inner)
        out.append(b("│") + " " + pad(l, inner) + " " + b("│"))
    out.append(b("╰" + "─" * (inner + 2) + "╯"))
    return "\n".join(out)


def table(rows: list[list[str]], headers: list[str], align: list[str] | None = None,
          color: bool | None = None, max_width: int | None = None) -> str:
    """Render a unicode box-drawing table. Column widths use visible length (ANSI ignored).

    ``align`` is a per-column list of '<' / '>' / '^'; by default numeric-looking columns are
    right-aligned. Wide tables shrink their widest columns to fit the terminal.
    """
    if color is None:
        color = color_enabled()
    ncols = max([len(headers)] + [len(r) for r in rows]) if (headers or rows) else 0
    hdrs = [str(h) for h in headers] + [""] * (ncols - len(headers))
    body = [[("" if c is None else str(c)) for c in r] + [""] * (ncols - len(r)) for r in rows]
    if align is None:
        align = []
        for i in range(ncols):
            vals = [strip_ansi(r[i]).strip() for r in body if strip_ansi(r[i]).strip()]
            numeric = bool(vals) and all(_looks_numeric(v) for v in vals)
            align.append(">" if numeric else "<")
    align = list(align) + ["<"] * (ncols - len(align))
    widths = [max([vlen(hdrs[i])] + [vlen(r[i]) for r in body]) for i in range(ncols)]
    limit = (max_width or term_width()) - (3 * ncols + 1)
    while sum(widths) > limit and max(widths, default=0) > 6:
        j = widths.index(max(widths))
        widths[j] -= 1
    b = lambda s: paint(s, "gray", color=color)  # noqa: E731

    def line(l: str, m: str, r: str) -> str:
        return b(l + m.join("─" * (w + 2) for w in widths) + r)

    def row(cells: list[str], hdr: bool = False) -> str:
        parts = []
        for i, c in enumerate(cells):
            if vlen(c) > widths[i]:
                c = truncate(c, widths[i])
            c = pad(c, widths[i], "<" if hdr else align[i])
            parts.append(paint(c, "bold", color=color) if hdr else c)
        return b("│") + b("│").join(f" {p} " for p in parts) + b("│")

    out = [line("┌", "┬", "┐"), row(hdrs, hdr=True), line("├", "┼", "┤")]
    out += [row(r) for r in body]
    out.append(line("└", "┴", "┘"))
    return "\n".join(out)


_NUM_RE = re.compile(r"^[$\-+]?[\d,.]+\s*(%|[kMB]|ms|s|tok)?$")


def _looks_numeric(s: str) -> bool:
    return bool(_NUM_RE.match(s))


# --------------------------------------------------------------------------------------------
# banner

def _pill(label: str, value: str, on: bool, color: bool, hue: str = "green") -> str:
    dot = paint("●", hue if on else "gray", color=color)
    val = paint(value, "bold", hue, color=color) if on else paint(value, "gray", color=color)
    return f"{dot} {paint(label, 'dim', color=color)} {val}"


def banner(cfg: Any, base_url: str, color: bool | None = None) -> str:
    """Startup box: version, proxy + dashboard URLs, copy-paste env exports, feature pills."""
    if color is None:
        color = color_enabled()
    base = base_url.rstrip("/")
    p = lambda s, *st: paint(s, *st, color=color)  # noqa: E731
    title = p("◉ llmtap", "bold", "bright_magenta") + p(f" v{__version__}", "magenta") + p(" — recording LLM traffic", "dim")
    lines = [
        "",
        f"{p('Proxy    ', 'dim')} {p(base, 'bold', 'bright_cyan')}",
        f"{p('Dashboard', 'dim')} {p(base + '/_llmtap/', 'bold', 'underline', 'bright_cyan')}",
        "",
        p("Point your app at it:", "dim"),
        f"  {p('export', 'magenta')} ANTHROPIC_BASE_URL={p(base, 'green')}",
        f"  {p('export', 'magenta')} OPENAI_BASE_URL={p(base + '/v1', 'green')}",
        "",
    ]
    guard = str(getattr(cfg, "guard", "off") or "off")
    guard_hue = {"block": "red", "redact": "yellow", "warn": "cyan"}.get(guard, "gray")
    if getattr(cfg, "guard_pii", False) and guard != "off":
        guard += "+pii"
    cache_on = bool(getattr(cfg, "cache", False))
    ttl = int(getattr(cfg, "cache_ttl_s", 0) or 0)
    cache_v = ("on" + (f" ({fmt_ms(ttl * 1000)})" if ttl else "")) if cache_on else "off"
    cas = str(getattr(cfg, "cassette", "") or "")
    cas_v = f"{_short_path(cas)} [{getattr(cfg, 'mode', 'auto')}]" if cas else "off"
    daily = float(getattr(cfg, "budget_daily_usd", 0) or 0)
    sess = float(getattr(cfg, "budget_session_usd", 0) or 0)
    budget_parts = []
    if daily > 0:
        budget_parts.append(f"{fmt_usd(daily)}/day")
    if sess > 0:
        budget_parts.append(f"{fmt_usd(sess)}/session")
    budget_v = " · ".join(budget_parts) if budget_parts else "unlimited"
    thr = int(getattr(cfg, "loop_threshold", 0) or 0)
    loop_v = f"{thr}× / {int(getattr(cfg, 'loop_window_s', 60) or 60)}s" if thr > 0 else "off"
    pills = [
        _pill("guard", guard, guard != "off", color, guard_hue),
        _pill("cache", cache_v, cache_on, color),
        _pill("cassette", cas_v, bool(cas), color, "blue"),
        _pill("budget", budget_v, bool(budget_parts), color, "yellow"),
        _pill("loop-breaker", loop_v, thr > 0, color, "magenta"),
    ]
    # flow pills onto as few lines as fit
    maxw = min(term_width(), 110) - 6
    cur = ""
    for pl in pills:
        cand = (cur + "   " + pl) if cur else pl
        if cur and vlen(cand) > maxw:
            lines.append(cur)
            cur = pl
        else:
            cur = cand
    if cur:
        lines.append(cur)
    lines += ["", p("Ctrl-C to stop · llmtap wrap -- <cmd> to auto-configure a command", "dim")]
    return box(lines, title=title, color=color, border="magenta")


def _short_path(path: str, width: int = 36) -> str:
    home = os.path.expanduser("~")
    if home and home != "/" and path.startswith(home):
        path = "~" + path[len(home):]
    if len(path) > width:
        path = "…" + path[-(width - 1):]
    return path


# --------------------------------------------------------------------------------------------
# per-request line

_SOURCE_STYLE: dict[str, tuple[str, tuple[str, ...]]] = {
    "upstream": ("LIVE", ("bold", "green")),
    "cache": ("CACHE", ("bold", "cyan")),
    "cassette": ("TAPE", ("bold", "blue")),
    "blocked": ("BLOCK", ("bold", "red")),
    "budget": ("BUDGET", ("bold", "yellow")),
    "loop": ("LOOP", ("bold", "magenta")),
    "error": ("ERROR", ("bold", "red")),
}


def source_badge(source: str, color: bool | None = None) -> str:
    """Fixed-width colored badge for an exchange source."""
    label, st = _SOURCE_STYLE.get(source, (str(source or "?").upper()[:6], ("bold", "white")))
    return paint(pad(label, 6), *st, color=color)


def status_color(status: int) -> str:
    if status >= 500 or status == 0:
        return "red"
    if status >= 400:
        return "yellow"
    if status >= 300:
        return "cyan"
    return "green"


def format_exchange(ex: Any, color: bool | None = None, width: int | None = None) -> str:
    """Build the one-line log string for an Exchange (see log_exchange)."""
    if color is None:
        color = color_enabled()
    width = width or term_width()
    p = lambda s, *st: paint(s, *st, color=color)  # noqa: E731
    ts = float(getattr(ex, "ts", 0) or time.time())
    clock = p(time.strftime("%H:%M:%S", time.localtime(ts)), "gray")
    src = str(getattr(ex, "source", "upstream") or "upstream")
    provider = str(getattr(ex, "provider", "") or "?")
    model = str(getattr(ex, "model", None) or "-")
    pm_w = max(18, min(40, width - 70))
    raw = f"{provider}/{model}"
    pm = truncate(raw, pm_w)
    if pm == raw:
        pm = p(provider, "dim") + p("/", "gray") + p(model, "bold")
    status = int(getattr(ex, "status", 0) or 0)
    st = p(pad(str(status) if status else "---", 3), "bold", status_color(status))
    u = getattr(ex, "usage", None)
    tin = int(getattr(u, "input_tokens", 0) or 0) + int(getattr(u, "cache_read_tokens", 0) or 0) \
        + int(getattr(u, "cache_write_tokens", 0) or 0)
    tout = int(getattr(u, "output_tokens", 0) or 0)
    est = "~" if getattr(u, "estimated", False) else ""
    tok = f"{est}{fmt_tokens(tin)}→{fmt_tokens(tout)} tok"
    cost = float(getattr(ex, "cost_usd", 0) or 0)
    saved = float(getattr(ex, "saved_usd", 0) or 0)
    if src in ("cache", "cassette") and saved > 0:
        cost_s = p(pad("saved " + fmt_usd(saved), 9, ">"), "cyan")
    else:
        cost_s = p(pad(fmt_usd(cost), 9, ">"), "yellow" if cost > 0 else "gray")
    lat = p(pad(fmt_ms(getattr(ex, "latency_ms", 0)), 7, ">"), "dim")
    parts = [clock, source_badge(src, color=color), pad(pm, pm_w), st, pad(tok, 16, ">"), cost_s, lat]
    ttft = getattr(ex, "ttft_ms", None)
    if getattr(ex, "streamed", False) and ttft is not None:
        parts.append(p(f"⚡ttft {fmt_ms(ttft)}", "bright_cyan"))
    findings = list(getattr(ex, "guard_findings", None) or [])
    if src == "blocked":
        n = len(findings)
        parts.append(p(f"⛔ blocked{f' ({n} secret' + ('s' if n != 1 else '') + ')' if n else ''}", "bold", "red"))
    elif findings:
        n = len(findings)
        redacted = "[REDACTED" in str(getattr(ex, "request_body", "") or "")
        verb = "redacted" if redacted else "detected"
        parts.append(p(f"⚠ {n} secret{'s' if n != 1 else ''} {verb}", "bold", "red"))
    if src in ("budget", "loop") or (getattr(ex, "error", None) and src != "blocked"):
        err = str(getattr(ex, "error", "") or "")
        if err:
            parts.append(p(truncate(err, 60), "red"))
    return " ".join(parts)


def log_exchange(ex: Any, stream: IO[Any] | None = None) -> None:
    """Print one aligned, colored line for ``ex`` to stderr. Never raises."""
    stream = sys.stderr if stream is None else stream
    try:
        line = format_exchange(ex, color=color_enabled(stream))
        print(line, file=stream, flush=True)
    except Exception as e:  # logging must never break the proxy
        try:
            print(f"llmtap: {getattr(ex, 'id', '?')} (log error: {e})", file=stream, flush=True)
        except Exception:
            pass


# --------------------------------------------------------------------------------------------
# end-of-run summary

def summary(stats: dict, session_spent: float, color: bool | None = None, title: str = "llmtap session summary") -> str:
    """Report box for ``llmtap wrap``: requests, tokens, cost, savings, latency, errors, guard, top models."""
    if color is None:
        color = color_enabled()
    p = lambda s, *st: paint(s, *st, color=color)  # noqa: E731
    s = stats or {}
    req = int(s.get("requests", 0) or 0)
    tin = int(s.get("input_tokens", 0) or 0)
    tout = int(s.get("output_tokens", 0) or 0)
    cost = float(s.get("cost_usd", 0) or 0)
    saved = float(s.get("saved_usd", 0) or 0)
    errors = int(s.get("errors", 0) or 0)
    blocked = int(s.get("blocked", 0) or 0)
    findings = int(s.get("guard_findings", 0) or 0)
    hits = int(s.get("cache_hits", 0) or 0)
    avg = s.get("avg_latency_ms")

    def kv(k: str, v: str) -> str:
        return f"{p(pad(k, 14), 'dim')} {v}"

    lines = [
        kv("Requests", p(fmt_count(req), "bold") + (p(f"  ({hits} from cache/replay)", "cyan") if hits else "")),
        kv("Tokens", f"{p(fmt_tokens(tin), 'bold')} in → {p(fmt_tokens(tout), 'bold')} out"),
        kv("Total cost", p(fmt_usd(cost), "bold", "yellow")),
    ]
    if saved > 0:
        lines.append(kv("Saved", p(fmt_usd(saved), "bold", "green") + p(" by cache/replay", "dim")))
    if abs(session_spent - cost) > 1e-9 and session_spent > 0:
        lines.append(kv("Session spend", fmt_usd(session_spent)))
    lines.append(kv("Avg latency", fmt_ms(avg) if req else "-"))
    lines.append(kv("Errors", p(str(errors), "red" if errors else "green")))
    if blocked or findings:
        lines.append(kv("Blocked", p(str(blocked), "bold", "red") if blocked else "0"))
    if findings:
        lines.append(kv("Guard",p(f"⚠ {findings} secret finding{'s' if findings != 1 else ''}", "bold", "red")))
    groups = [g for g in (s.get("groups") or []) if g.get("requests")]
    if groups:
        lines.append("")
        lines.append(p("Top models", "dim"))
        kw = min(30, max(vlen(str(g.get("key") or "-")) for g in groups[:5]))
        for g in groups[:5]:
            key = truncate(str(g.get("key") or "-"), kw)
            lines.append(
                f"  {pad(key, kw)}  {pad(str(g.get('requests', 0)), 4, '>')} req  "
                f"{pad(fmt_tokens(int(g.get('input_tokens', 0) or 0) + int(g.get('output_tokens', 0) or 0)), 6, '>')} tok  "
                f"{p(pad(fmt_usd(float(g.get('cost_usd', 0) or 0)), 9, '>'), 'yellow')}"
            )
    if req == 0:
        lines = [p("No LLM requests were recorded.", "dim"),
                 p("Does the command use ANTHROPIC_BASE_URL / OPENAI_BASE_URL?", "dim")]
    head = p("◉ " + title, "bold", "bright_magenta")
    return box(lines, title=head, color=color, border="magenta")
