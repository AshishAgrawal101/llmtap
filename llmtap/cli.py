"""The ``llmtap`` command line.

Sibling modules are imported lazily inside each command so ``llmtap --help`` keeps working
even when one of them is broken or missing.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from llmtap import __version__

# --------------------------------------------------------------------------------------------
# helpers

_UNITS = {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "hr": 3600, "d": 86400, "w": 7 * 86400}
_DUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(w|d|hr|h|min|m|sec|s)")


def parse_duration(s: str) -> float:
    """'24h' -> 86400.0; '1h30m' -> 5400.0; bare number = seconds. Raises ValueError."""
    t = str(s).strip().lower()
    if not t:
        raise ValueError("empty duration")
    try:
        return float(t)
    except ValueError:
        pass
    pos, total = 0, 0.0
    for m in _DUR_RE.finditer(t):
        if t[pos:m.start()].strip():
            raise ValueError(f"invalid duration: {s!r}")
        total += float(m.group(1)) * _UNITS[m.group(2)]
        pos = m.end()
    if pos == 0 or t[pos:].strip():
        raise ValueError(f"invalid duration: {s!r} (try 30m, 24h, 7d or all)")
    return total


def parse_since(s: str | None, now: float | None = None) -> float | None:
    """Turn '24h' / '7d' / '30m' / 'today' / '2026-01-31' / 'all' into an epoch cutoff (None = all time)."""
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in ("", "all", "ever", "forever", "none", "0"):
        return None
    now = time.time() if now is None else now
    if t == "today":
        lt = time.localtime(now)
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    if t == "yesterday":
        lt = time.localtime(now)
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday - 1, 0, 0, 0, 0, 0, -1))
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}([t ]\d{2}:\d{2}(:\d{2})?)?", t):
        from datetime import datetime
        dt = datetime.fromisoformat(t.upper().replace(" ", "T"))  # ValueError = not a real date
        try:
            cutoff = dt.timestamp()
        # outside the platform's time range: OverflowError/OSError on Windows, ValueError ("year 0 is
        # out of range") on Linux/macOS for dates near year 1
        except (OverflowError, OSError, ValueError):
            if int(t[:4]) <= 1970:
                return None
            raise ValueError(f"date out of range: {s!r}") from None
    else:
        dur = parse_duration(t)
        if dur != dur:
            raise ValueError(f"invalid duration: {s!r}")
        cutoff = now - dur
    return None if cutoff == float("-inf") else cutoff  # `inf` = all time


def _since_arg(s: str) -> float | None:
    try:
        return parse_since(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _err(msg: str) -> None:
    from llmtap import term
    print(term.paint("llmtap: ", "bold", "red", color=term.color_enabled(sys.stderr)) + msg, file=sys.stderr)


def _out(text: str = "") -> None:
    print(text, flush=True)


def _color_out() -> bool:
    from llmtap import term
    return term.color_enabled(sys.stdout)


# --------------------------------------------------------------------------------------------
# argument parsing

S = argparse.SUPPRESS

# CLI dest -> Config field
_OVERRIDE_MAP: dict[str, str] = {
    "host": "host", "port": "port", "db": "db_path", "quiet": "quiet",
    "cache": "cache", "cache_ttl": "cache_ttl_s",
    "cassette": "cassette", "mode": "mode", "allow_miss": "allow_miss",
    "guard": "guard", "pii": "guard_pii",
    "budget_daily": "budget_daily_usd", "budget_session": "budget_session_usd",
    "loop_threshold": "loop_threshold", "loop_window": "loop_window_s",
    "timeout": "upstream_timeout_s",
}


def _db_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("storage")
    g.add_argument("--db", metavar="PATH", default=S, help="SQLite database path (default ~/.llmtap/llmtap.db)")
    g.add_argument("--config", metavar="FILE", default=S, help="config file (default ./llmtap.toml)")
    return p


def _proxy_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("proxy")
    g.add_argument("--host", default=S, help="bind address (default 127.0.0.1)")
    g.add_argument("--port", type=int, default=S, help="listen port (default 4141; 0 = any free port)")
    g.add_argument("--quiet", "-q", action="store_const", const=True, default=S, help="no per-request log lines")
    g.add_argument("--timeout", type=float, default=S, metavar="SECS", help="upstream timeout (default 600)")
    g.add_argument("--upstream", action="append", default=S, metavar="NAME=URL[,STYLE]",
                   help="add/override an upstream, e.g. ollama=http://localhost:11434 (route via /<name>/...)")
    g = p.add_argument_group("cache & cassettes")
    g.add_argument("--cache", action="store_const", const=True, default=S, help="serve identical requests from a local cache")
    g.add_argument("--no-cache", dest="cache", action="store_const", const=False, default=S, help="disable the cache")
    g.add_argument("--cache-ttl", type=int, default=S, metavar="SECS", help="cache entry lifetime (0 = forever)")
    g.add_argument("--cassette", default=S, metavar="FILE", help="record/replay a .jsonl cassette (VCR for LLMs)")
    g.add_argument("--mode", choices=["record", "replay", "auto"], default=S, help="cassette mode (default auto)")
    g.add_argument("--allow-miss", action="store_const", const=True, default=S,
                   help="replay mode: forward unrecorded requests upstream instead of failing")
    g = p.add_argument_group("guard & budgets")
    g.add_argument("--guard", choices=["off", "warn", "redact", "block"], default=S, help="secret firewall mode (default warn)")
    g.add_argument("--pii", action="store_const", const=True, default=S, help="also detect emails, phones, credit cards")
    g.add_argument("--budget-daily", type=float, default=S, metavar="USD", help="refuse calls once today's spend reaches this")
    g.add_argument("--budget-session", type=float, default=S, metavar="USD", help="refuse calls once this run's spend reaches this")
    g.add_argument("--loop-threshold", type=int, default=S, metavar="N", help="block an identical request seen N times...")
    g.add_argument("--loop-window", type=int, default=S, metavar="SECS", help="...within this window (default 60)")
    return p


def build_parser() -> argparse.ArgumentParser:
    """Build the full argparse tree."""
    db = _db_parent()
    px = _proxy_parent()
    parser = argparse.ArgumentParser(
        prog="llmtap",
        description="The flight recorder & firewall for LLM apps: record, price, cache, replay, guard and budget every LLM call.",
        epilog="Quick start:  llmtap wrap -- python my_agent.py      |  llmtap serve  then  eval \"$(llmtap env)\"",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=f"llmtap {__version__}")
    parser.add_argument("--db", metavar="PATH", default=S, help=argparse.SUPPRESS)
    parser.add_argument("--config", metavar="FILE", default=S, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name: str, help: str, parents: list[argparse.ArgumentParser] | None = None, **kw: Any) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help, description=help, parents=parents or [], **kw)
        return sp

    add("serve", "run the proxy + live dashboard", [db, px])

    sp = add("wrap", "run a command with its LLM traffic routed through llmtap", [db, px],
             usage="llmtap wrap [options] -- <command> [args...]")
    sp.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run (after --)")

    sp = add("env", "print shell exports pointing SDKs at llmtap: eval \"$(llmtap env)\"")
    sp.add_argument("--host", default=S)
    sp.add_argument("--port", type=int, default=S)
    sp.add_argument("--config", default=S, help=argparse.SUPPRESS)
    sp.add_argument("--shell", choices=["sh", "fish", "powershell"], default="sh", help="output syntax (default sh)")

    sp = add("stats", "spend / token / latency report", [db])
    sp.add_argument("--since", type=_since_arg, default=None, metavar="WHEN", help="30m, 24h, 7d, today, 2026-01-31 or all (default all)")
    sp.add_argument("--by", default=None, metavar="KEY", help="group by model|provider|source|day|hour|tag:<name> (default model)")
    sp.add_argument("--json", action="store_true", help="machine-readable output")

    sp = add("ls", "list recent exchanges", [db])
    sp.add_argument("-n", "--limit", type=int, default=20)
    sp.add_argument("-q", "--query", default=None, help="substring search over bodies/model/path")
    sp.add_argument("--since", type=_since_arg, default=None, metavar="WHEN")
    sp.add_argument("--provider", default=None)
    sp.add_argument("--model", default=None)
    sp.add_argument("--source", default=None, help="upstream|cache|cassette|blocked|budget|loop|error")
    sp.add_argument("--json", action="store_true", help="JSON lines of summaries")

    sp = add("show", "show one exchange (id or unique prefix)", [db])
    sp.add_argument("id")
    sp.add_argument("--raw", action="store_true", help="dump the full exchange as JSON")

    sp = add("tail", "follow new exchanges as they are recorded", [db])
    sp.add_argument("-n", "--lines", type=int, default=10, help="show this many recent exchanges first")
    sp.add_argument("--interval", type=float, default=0.5, help=argparse.SUPPRESS)
    sp.add_argument("--once", action="store_true", help=argparse.SUPPRESS)

    sp = add("export", "export exchanges as JSONL or CSV", [db])
    sp.add_argument("--format", "-f", choices=["jsonl", "csv"], default="jsonl")
    sp.add_argument("-o", "--output", default="-", help="output file (default stdout)")
    sp.add_argument("--since", type=_since_arg, default=None, metavar="WHEN")
    sp.add_argument("-n", "--limit", type=int, default=0, help="max rows (default all)")

    sp = add("prices", "show the built-in price table (USD per 1M tokens)")
    sp.add_argument("filter", nargs="?", default=None)
    sp.add_argument("--json", action="store_true")

    sp = add("scan", "scan files/stdin for secrets (exit 1 if any found)")
    sp.add_argument("files", nargs="*", default=[], metavar="FILE", help="files to scan, or - for stdin (default stdin)")
    sp.add_argument("--pii", action="store_true", help="also detect emails, phones, credit cards")
    sp.add_argument("--json", action="store_true")

    sp = add("clear", "delete all recorded exchanges", [db])
    sp.add_argument("--yes", "-y", action="store_true", help="don't ask for confirmation")

    add("version", "print version")

    sp = add("demo", "spin up a fake LLM + traffic and open the dashboard (no API keys needed)")
    sp.add_argument("--port", type=int, default=4141)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--no-open", action="store_true", help="don't open a browser")
    sp.add_argument("--count", type=int, default=40, help="number of demo requests (default 40)")
    sp.add_argument("--delay", type=float, default=0.3, help="seconds between demo requests")
    sp.add_argument("--exit", action="store_true", help="exit (with a summary) once the demo traffic is done")
    sp.add_argument("--quiet", "-q", action="store_true", help="no per-request log lines")

    sp = add("report", "shareable cost & safety report: spend, waste finder, guard findings (HTML / md / json)", [db])
    sp.add_argument("--since", type=_since_arg, default="7d", metavar="WHEN",
                    help="30m, 24h, 7d, today, 2026-01-31 or all (default 7d)")
    sp.add_argument("-o", "--output", default=None, metavar="FILE",
                    help="output file (default llmtap-report.html; '-' = stdout)")
    sp.add_argument("--format", "-f", choices=["html", "md", "json"], default=None,
                    help="html (default; or inferred from -o extension), md or json")
    sp.add_argument("--open", action="store_true", help="open the HTML report in your browser")

    sp = add("doctor", "check your setup (port, db, env vars, SDK base URLs) and print fix hints", [db])
    sp.add_argument("--online", action="store_true", help="also check that the upstream API hosts are reachable (5s timeout)")
    sp.add_argument("--host", default=S)
    sp.add_argument("--port", type=int, default=S)
    return parser


def config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Map parsed CLI flags onto Config field overrides (unset flags are omitted)."""
    ov: dict[str, Any] = {}
    for dest, field_name in _OVERRIDE_MAP.items():
        if hasattr(args, dest):
            v = getattr(args, dest)
            if v is not None:
                ov[field_name] = v
    ups = getattr(args, "upstream", None)
    if ups:
        table: dict[str, dict[str, str]] = {}
        for spec in ups:
            name, sep, rest = spec.partition("=")
            if not sep or not name or not rest:
                raise ValueError(f"--upstream expects NAME=URL[,STYLE], got {spec!r}")
            url, _, style = rest.partition(",")
            table[name.strip()] = {"base": url.strip().rstrip("/"), "style": (style.strip() or "openai")}
        ov["upstreams"] = table
    return ov


def make_config(args: argparse.Namespace, **extra: Any) -> Any:
    """load_config(--config, overrides-from-flags + extra)."""
    from llmtap.config import load_config
    ov = config_overrides(args)
    for k, v in extra.items():
        ov.setdefault(k, v)
    return load_config(getattr(args, "config", None), ov)


def _open_store(args: argparse.Namespace) -> Any:
    from llmtap.store import Store
    cfg = make_config(args)
    return Store(cfg.resolved_db_path())


# --------------------------------------------------------------------------------------------
# request-body helpers for show / ls

def _content_text(c: Any) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        t = c.get("type")
        if "text" in c and isinstance(c["text"], str):
            return c["text"]
        if t == "tool_use" or t == "function_call":
            args = c.get("input", c.get("arguments", ""))
            return f"[tool call {c.get('name', '?')}: {json.dumps(args)[:200] if not isinstance(args, str) else args[:200]}]"
        if t == "tool_result":
            return f"[tool result: {_content_text(c.get('content'))}]"
        if t in ("image", "image_url", "input_image"):
            return "[image]"
        if "content" in c:
            return _content_text(c["content"])
        return f"[{t or 'object'}]"
    if isinstance(c, list):
        return "\n".join(x for x in (_content_text(i) for i in c) if x)
    return str(c)


def conversation(req: Any) -> list[tuple[str, str]]:
    """Extract [(role, text)] from an anthropic/openai/responses request body."""
    out: list[tuple[str, str]] = []
    if not isinstance(req, dict):
        return out
    if req.get("system"):
        out.append(("system", _content_text(req["system"])))
    if req.get("instructions"):
        out.append(("system", _content_text(req["instructions"])))
    msgs = req.get("messages")
    for m in msgs if isinstance(msgs, list) else []:
        if isinstance(m, dict):
            text = _content_text(m.get("content"))
            tcs = m.get("tool_calls")
            if tcs and isinstance(tcs, list):
                fns = [tc.get("function") for tc in tcs if isinstance(tc, dict)]
                calls = ", ".join(str(fn.get("name", "?")) if isinstance(fn, dict) else "?" for fn in fns)
                text = (text + "\n" if text else "") + f"[tool calls: {calls}]"
            out.append((str(m.get("role", "?")), text))
    inp = req.get("input")
    if isinstance(inp, str):
        out.append(("user", inp))
    elif isinstance(inp, list):
        for m in inp:
            if isinstance(m, dict) and "role" in m:
                out.append((str(m["role"]), _content_text(m.get("content"))))
            elif isinstance(m, str):
                out.append(("user", m))
            else:
                out.append(("input", _content_text(m)))
    if not out and isinstance(req.get("prompt"), str):
        out.append(("user", req["prompt"]))
    return out


def _last_user(req_body: str) -> str:
    try:
        req = json.loads(req_body) if req_body else None
    except ValueError:
        return req_body[:200]
    conv = conversation(req)
    for role, text in reversed(conv):
        if role == "user":
            return text
    return conv[-1][1] if conv else ""


def _one_line(s: str) -> str:
    from llmtap.term import clean
    return clean(" ".join(str(s or "").split()))


def _when(ts: float) -> str:
    now = time.localtime()
    lt = time.localtime(ts)
    if (lt.tm_year, lt.tm_yday) == (now.tm_year, now.tm_yday):
        return time.strftime("%H:%M:%S", lt)
    return time.strftime("%m-%d %H:%M", lt)


# --------------------------------------------------------------------------------------------
# commands

def cmd_version(args: argparse.Namespace) -> int:
    _out(f"llmtap {__version__}")
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    cfg = make_config(args)
    host = cfg.host if cfg.host not in ("0.0.0.0", "::", "") else "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # IPv6 literal
    base = f"http://{host}:{cfg.port}"
    pairs = env_vars(base)
    for k, v in pairs.items():
        if args.shell == "fish":
            _out(f"set -gx {k} {v}")
        elif args.shell == "powershell":
            _out(f'$env:{k} = "{v}"')
        else:
            _out(f"export {k}={v}")
    return 0


def env_vars(base: str) -> dict[str, str]:
    """The environment variables that point SDKs at a llmtap proxy at ``base``."""
    base = base.rstrip("/")
    return {
        "ANTHROPIC_BASE_URL": base,
        "OPENAI_BASE_URL": f"{base}/v1",
        "OPENAI_API_BASE": f"{base}/v1",
        "LLMTAP_URL": base,
    }


def _wait_forever(stop: threading.Event | None = None) -> None:
    while not (stop and stop.is_set()):
        time.sleep(0.25)


def cmd_serve(args: argparse.Namespace) -> int:
    from llmtap import term
    from llmtap.proxy import LLMTap
    cfg = make_config(args)
    try:
        tap = LLMTap(cfg)
        tap.start_background()
    except OSError as e:
        _err(f"cannot listen on {cfg.host}:{cfg.port}: {e.strerror or e}")
        return 1
    base = getattr(tap, "base_url", None) or cfg.base_url()
    print(term.banner(cfg, base, color=term.color_enabled(sys.stdout)), flush=True)
    try:
        _wait_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print(term.paint("\nshutting down…", "dim", color=term.color_enabled(sys.stderr)), file=sys.stderr, flush=True)
        _shutdown(tap)
    return 0


def _shutdown(tap: Any) -> None:
    try:
        tap.shutdown()
    except Exception:
        pass


def cmd_wrap(args: argparse.Namespace) -> int:
    from llmtap import term
    cmd = list(getattr(args, "cmd", None) or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        _err("nothing to run. usage: llmtap wrap [options] -- <command> [args...]")
        return 2
    from llmtap.proxy import LLMTap
    cfg = make_config(args, port=0)
    try:
        tap = LLMTap(cfg)
        tap.start_background()
    except OSError as e:
        _err(f"cannot start proxy: {e}")
        return 1
    base = getattr(tap, "base_url", None) or f"http://{cfg.host}:{getattr(tap, 'port', cfg.port)}"
    env = dict(os.environ)
    env.update(env_vars(base))
    # make sure HTTP(S)_PROXY settings don't hijack traffic to the local proxy. Both spellings get the
    # same merged list: most clients prefer lowercase no_proxy, which must not drop the user's NO_PROXY hosts.
    cur = ",".join(v for v in dict.fromkeys(env.get(k, "") for k in ("NO_PROXY", "no_proxy")) if v)
    env["NO_PROXY"] = env["no_proxy"] = f"{cur},127.0.0.1,localhost" if cur else "127.0.0.1,localhost"
    color = term.color_enabled(sys.stderr)
    if not cfg.quiet:
        print(term.paint("◉ llmtap", "bold", "bright_magenta", color=color)
              + term.paint(f" recording → {base}  ·  dashboard {base}/_llmtap/", "dim", color=color),
              file=sys.stderr, flush=True)
    start = time.time() - 0.001
    try:
        proc = subprocess.Popen(cmd, env=env)
    except FileNotFoundError:
        _shutdown(tap)
        _err(f"command not found: {cmd[0]}")
        return 127
    except PermissionError:
        _shutdown(tap)
        _err(f"permission denied: {cmd[0]}")
        return 126
    except OSError as e:  # e.g. ENOEXEC / WinError 193: not a program this OS can run
        _shutdown(tap)
        _err(f"cannot run {cmd[0]}: {e.strerror or e}")
        return 126

    def forward(signum: int, _frame: Any) -> None:
        # A terminal Ctrl-C already reaches the child (same process group). Forward only if
        # it is still running shortly after, which covers `kill -INT <llmtap pid>`.
        def later() -> None:
            time.sleep(0.3)
            if proc.poll() is None:
                try:
                    proc.send_signal(signum)
                except (OSError, ValueError):  # ValueError: Windows only supports SIGTERM/CTRL_*_EVENT
                    pass
        threading.Thread(target=later, daemon=True).start()

    old: dict[int, Any] = {}
    in_main = threading.current_thread() is threading.main_thread()
    if in_main:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                old[sig] = signal.signal(sig, forward)
            except (ValueError, OSError):
                pass
    try:
        while True:
            try:
                rc = proc.wait()
                break
            except KeyboardInterrupt:  # pragma: no cover - only if handler couldn't be installed
                continue
    finally:
        for sig, h in old.items():
            try:
                signal.signal(sig, h)
            except (ValueError, OSError):
                pass
    time.sleep(0.15)  # let in-flight records land
    try:
        stats = tap.store.stats(since=start, group_by="model")
        try:
            spent = float(tap.budget.session_spent())
        except Exception:
            spent = float(stats.get("cost_usd", 0) or 0)
        print(term.summary(stats, spent, color=color), file=sys.stderr, flush=True)
    except Exception as e:
        _err(f"could not compute summary: {e}")
    finally:
        _shutdown(tap)
    if rc < 0:
        return 128 + (-rc)
    return rc


def cmd_stats(args: argparse.Namespace) -> int:
    from llmtap import term
    store = _open_store(args)
    group = args.by or "model"
    try:
        st = store.stats(since=args.since, group_by=group)
        waste_usd, waste_n = _potential_savings(store, args.since)
    except ValueError as e:
        _err(str(e))
        return 2
    finally:
        store.close()
    if args.json:
        st = dict(st, potential_savings_usd=waste_usd, duplicate_requests=waste_n)
        _out(json.dumps(st, indent=2, default=str))
        return 0
    color = _color_out()
    try:
        label = "all time" if args.since is None else "since " + time.strftime("%Y-%m-%d %H:%M", time.localtime(args.since))
    except (OverflowError, OSError, ValueError):  # e.g. --since 99999d: before the platform's epoch
        label = "all time"
    totals = {k: v for k, v in st.items() if k != "groups"}
    hint = ("Record traffic with `llmtap serve` or `llmtap wrap -- <command>`." if args.since is None
            else "Nothing in this window; try --since all.")
    _out(term.summary(totals, float(st.get("cost_usd", 0) or 0), color=color, title=f"llmtap stats · {label}",
                      empty_hint=hint))
    if int(st.get("requests", 0) or 0) > 0:
        _out(_savings_line(waste_usd, waste_n, color))
    groups = st.get("groups") or []
    if groups:
        total_cost = float(st.get("cost_usd", 0) or 0)
        rows = []
        for g in groups:
            c = float(g.get("cost_usd", 0) or 0)
            share = f"{100 * c / total_cost:.0f}%" if total_cost > 0 else "-"
            rows.append([
                str(g.get("key") if g.get("key") not in (None, "") else "-"),
                fmt_int(g.get("requests")),
                term.fmt_tokens(g.get("input_tokens", 0)),
                term.fmt_tokens(g.get("output_tokens", 0)),
                term.fmt_usd(c),
                share,
            ])
        _out(term.table(rows, [group.upper(), "REQS", "IN", "OUT", "COST", "SHARE"], color=color))
    return 0


def _potential_savings(store: Any, since: float | None) -> tuple[float, int]:
    """Waste-finder numbers for ``stats``; (0, 0) if the report module is unavailable."""
    try:
        from llmtap.report import potential_savings
        return potential_savings(store, since)
    except ImportError:  # pragma: no cover
        return 0.0, 0


def _savings_line(usd: float, n: int, color: bool) -> str:
    from llmtap import term
    amount = term.paint(term.fmt_usd(usd), "bold", "green" if usd > 0 else "dim", color=color)
    return f"Potential savings from caching: {amount} ({n} duplicate request{'s' if n != 1 else ''})"


def fmt_int(n: Any) -> str:
    return f"{int(n or 0):,}"


def cmd_ls(args: argparse.Namespace) -> int:
    from llmtap import term
    store = _open_store(args)
    try:
        items = store.list(limit=args.limit, q=args.query, since=args.since, provider=args.provider,
                           model=args.model, source=args.source)
    finally:
        store.close()
    if args.json:
        for ex in items:
            d = ex.to_dict()
            d.pop("request_body", None)
            d.pop("response_body", None)
            _out(json.dumps(d, default=str))
        return 0
    if not items:
        _out("no exchanges recorded yet — try: llmtap wrap -- <your command>   or   llmtap demo")
        return 0
    color = _color_out()
    rows = []
    for ex in items:
        u = ex.usage
        tin = u.input_tokens + u.cache_read_tokens + u.cache_write_tokens
        st = str(ex.status or "---")
        rows.append([
            term.paint(ex.id[:8], "gray", color=color),
            _when(ex.ts),
            term.source_badge(ex.source, color=color).strip(),
            term.clean(f"{ex.provider}/{ex.model or '-'}"),
            term.paint(st, term.status_color(ex.status), color=color),
            f"{term.fmt_tokens(tin)}→{term.fmt_tokens(u.output_tokens)}",
            term.fmt_usd(ex.cost_usd),
            term.fmt_ms(ex.latency_ms),
            term.truncate(_one_line(_last_user(ex.request_body)), 40),
        ])
    _out(term.table(rows, ["ID", "TIME", "SRC", "MODEL", "ST", "TOKENS", "COST", "LATENCY", "PROMPT"],
                    align=["<", "<", "<", "<", "<", ">", ">", ">", "<"], color=color))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    from llmtap import term
    store = _open_store(args)
    try:
        ex = store.get(args.id)
        matches = _count_id_prefix(store, args.id) if ex is None and len(args.id) >= 4 else 0
    finally:
        store.close()
    if ex is None:
        if matches > 1:
            _err(f"{args.id!r} is ambiguous: {matches} exchanges match; use a longer id prefix")
        else:
            _err(f"no exchange matching {args.id!r}")
        return 1
    if args.raw:
        _out(json.dumps(ex.to_dict(), indent=2, default=str))
        return 0
    color = _color_out()
    p = lambda s, *st: term.paint(term.clean(s), *st, color=color)  # noqa: E731
    c = term.clean
    u = ex.usage
    meta = [
        ("id", c(ex.id)),
        ("time", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ex.ts))),
        ("route", c(f"{ex.method} {ex.path}  ({ex.provider}, {ex.style} style)")),
        ("model", c(ex.model or "-")),
        ("status", p(str(ex.status), "bold", term.status_color(ex.status))),
        ("source", term.source_badge(ex.source, color=color).strip()),
        ("tokens", f"{fmt_int(u.input_tokens)} in · {fmt_int(u.output_tokens)} out"
                   + (f" · {fmt_int(u.cache_read_tokens)} cache-read" if u.cache_read_tokens else "")
                   + (f" · {fmt_int(u.cache_write_tokens)} cache-write" if u.cache_write_tokens else "")
                   + (" (estimated)" if u.estimated else "")),
        ("cost", term.fmt_usd(ex.cost_usd) + (f"  (saved {term.fmt_usd(ex.saved_usd)})" if ex.saved_usd else "")),
        ("latency", term.fmt_ms(ex.latency_ms) + (f"  ·  ttft {term.fmt_ms(ex.ttft_ms)}" if ex.ttft_ms is not None else "")
                    + ("  ·  streamed" if ex.streamed else "")),
    ]
    if ex.tags:
        meta.append(("tags", c(", ".join(f"{k}={v}" for k, v in ex.tags.items()))))
    if ex.error:
        meta.append(("error", p(ex.error, "red")))
    lines = [f"{p(k.ljust(8), 'dim')} {v}" for k, v in meta]
    _out(term.box(lines, title=p(f"exchange {ex.id[:8]}", "bold"), color=color))
    if ex.guard_findings:
        _out(p(f"\n⚠ {len(ex.guard_findings)} guard finding(s)", "bold", "red"))
        for f in ex.guard_findings:
            _out(f"  {p(str(f.get('kind', '?')), 'red')}  {c(f.get('path', ''))}  {c(f.get('preview', ''))}")
    try:
        req = json.loads(ex.request_body) if ex.request_body else None
    except ValueError:
        req = None
    conv = conversation(req)
    width = term.term_width()
    role_style = {"system": ("magenta",), "user": ("cyan",), "assistant": ("green",), "tool": ("yellow",)}
    if conv or ex.output_text:
        _out("")
    for role, text in conv:
        _out(p(f"▍{role}", "bold", *role_style.get(role, ("white",))))
        _out(_indent(_clip(text), width))
    if ex.output_text:
        _out(p("▍assistant (response)", "bold", "green"))
        _out(_indent(_clip(ex.output_text), width))
    elif not conv and ex.request_body:
        _out(p("▍request body", "bold"))
        _out(_indent(_clip(ex.request_body), width))
    if ex.response_body and not ex.output_text:
        _out(p("▍response body", "bold"))
        _out(_indent(_clip(ex.response_body), width))
    return 0


def _clip(text: str, limit: int = 4000) -> str:
    from llmtap.term import clean
    text = str(text or "").replace("\r\n", "\n")
    return clean(text if len(text) <= limit else text[:limit] + f"\n… ({len(text) - limit} more chars; use --raw)")


def _indent(text: str, width: int) -> str:
    return "\n".join("  " + line for line in text.splitlines() or [""])


def _ro_connect(store: Any) -> sqlite3.Connection | None:
    """A read-only connection to the store's database file (None for :memory: or on error)."""
    path = str(getattr(store, "path", "") or "")
    if not path or path == ":memory:" or path.startswith("file:"):
        return None
    try:
        return sqlite3.connect(Path(path).expanduser().resolve().as_uri() + "?mode=ro", uri=True, timeout=30.0)
    except sqlite3.Error:
        return None


def _count_id_prefix(store: Any, prefix: str) -> int:
    """How many exchange ids start with `prefix` (0 if unknown)."""
    conn = _ro_connect(store)
    if conn is None:
        return 0
    try:
        return int(conn.execute("SELECT COUNT(*) FROM exchanges WHERE substr(id, 1, ?) = ?",
                                (len(prefix), prefix)).fetchone()[0])
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def cmd_tail(args: argparse.Namespace) -> int:
    from llmtap import term
    store = _open_store(args)
    # Follow by insertion order (rowid): `ts` is the request START time, so a slow call recorded
    # now can carry a ts far older than rows already printed.
    conn = _ro_connect(store)
    try:
        last_row = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM exchanges").fetchone()[0] if conn else 0
        recent = store.list(limit=max(args.lines, 0)) if args.lines > 0 else []
        seen = {ex.id for ex in recent}
        for ex in reversed(recent):
            term.log_exchange(ex, stream=sys.stdout)
        if args.once or conn is None:
            return 0
        print(term.paint("… waiting for new exchanges (Ctrl-C to stop)", "dim", color=_color_out()), flush=True)
        while True:
            time.sleep(args.interval)
            for rowid, ex_id in conn.execute("SELECT rowid, id FROM exchanges WHERE rowid > ? ORDER BY rowid LIMIT 200",
                                             (last_row,)).fetchall():
                last_row = rowid
                ex = store.get(ex_id)
                if ex is None or ex_id in seen:
                    continue
                seen.add(ex_id)
                term.log_exchange(ex, stream=sys.stdout)
    except KeyboardInterrupt:
        return 0
    finally:
        if conn is not None:
            conn.close()
        store.close()


CSV_COLUMNS = [
    "id", "ts", "time", "provider", "style", "method", "path", "model", "status", "source", "streamed",
    "latency_ms", "ttft_ms", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "usage_estimated", "cost_usd", "saved_usd", "tags", "guard_findings", "error", "output_text", "request_body",
]


def flatten(ex: Any) -> dict[str, Any]:
    """One flat CSV-friendly row for an Exchange."""
    u = ex.usage
    return {
        "id": ex.id, "ts": f"{ex.ts:.3f}",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ex.ts)),
        "provider": ex.provider, "style": ex.style, "method": ex.method, "path": ex.path,
        "model": ex.model or "", "status": ex.status, "source": ex.source, "streamed": int(bool(ex.streamed)),
        "latency_ms": round(ex.latency_ms or 0, 1), "ttft_ms": "" if ex.ttft_ms is None else round(ex.ttft_ms, 1),
        "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
        "cache_read_tokens": u.cache_read_tokens, "cache_write_tokens": u.cache_write_tokens,
        "usage_estimated": int(bool(u.estimated)),
        "cost_usd": f"{ex.cost_usd:.6f}", "saved_usd": f"{ex.saved_usd:.6f}",
        "tags": json.dumps(ex.tags) if ex.tags else "",
        "guard_findings": len(ex.guard_findings or []),
        "error": ex.error or "", "output_text": ex.output_text, "request_body": ex.request_body,
    }


def _iter_all(store: Any, since: float | None, limit: int) -> Iterable[Any]:
    page, offset, n = 500, 0, 0
    seen: set[str] = set()  # rows recorded meanwhile shift OFFSET pages: don't export a row twice
    while True:
        want = page if not limit else min(page, limit - n)
        if want <= 0:
            return
        batch = store.list(limit=want, offset=offset, since=since)
        for ex in batch:
            if ex.id not in seen:
                seen.add(ex.id)
                n += 1
                yield ex
        offset += len(batch)
        if len(batch) < want:
            return


def cmd_export(args: argparse.Namespace) -> int:
    store = _open_store(args)
    try:
        items = list(_iter_all(store, args.since, args.limit))
    finally:
        store.close()
    items.reverse()  # chronological
    wrapped = False
    if args.output in ("-", ""):
        fh: Any = sys.stdout
        close = False
        if args.format == "csv" and hasattr(sys.stdout, "buffer"):
            # csv writes its own \r\n; a text-mode stdout on Windows would turn that into \r\r\n
            sys.stdout.flush()
            fh = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding or "utf-8",
                                  errors="replace", newline="", write_through=True)
            wrapped = True
    else:
        try:
            fh = open(args.output, "w", newline="", encoding="utf-8")
        except OSError as e:
            _err(f"cannot write {args.output}: {e.strerror or e}")
            return 1
        close = True
    try:
        if args.format == "csv":
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for ex in items:
                w.writerow(flatten(ex))
        else:
            for ex in items:
                fh.write(json.dumps(ex.to_dict(), default=str) + "\n")
        fh.flush()
    finally:
        if close:
            fh.close()
        elif wrapped:
            fh.detach()  # leave sys.stdout's buffer open
    if close:
        print(f"exported {len(items)} exchange(s) to {args.output}", file=sys.stderr)
    return 0


def cmd_prices(args: argparse.Namespace) -> int:
    from llmtap import term
    from llmtap.pricing import LONG_CONTEXT, PRICES
    flt = (args.filter or "").lower()
    items = sorted((k, v) for k, v in PRICES.items() if flt in k.lower())
    if args.json:
        data: dict[str, Any] = {}
        for k, v in items:
            data[k] = dict(v)
            if k in LONG_CONTEXT:
                above, lp = LONG_CONTEXT[k]
                data[k]["long_context"] = {"above_prompt_tokens": above, **lp}
        _out(json.dumps(data, indent=2))
        return 0
    if not items:
        _out(f"no models match {args.filter!r}")
        return 1

    def money(v: Any) -> str:
        if v is None:
            return "-"
        v = float(v)
        if v == 0:
            return "free"
        txt = f"{v:,.4f}".rstrip("0")             # $0.125 stays $0.125, not $0.12
        if len(txt.split(".")[1]) < 2:
            txt = f"{v:,.2f}"
        return "$" + txt

    rows = [[k, money(v.get("input")), money(v.get("output")), money(v.get("cache_read")), money(v.get("cache_write"))]
            for k, v in items]
    color = _color_out()
    _out(term.table(rows, ["MODEL", "INPUT", "OUTPUT", "CACHE READ", "CACHE WRITE"],
                    align=["<", ">", ">", ">", ">"], color=color))
    _out(term.paint(f"{len(rows)} model(s) · USD per 1M tokens · override in llmtap.toml [pricing]", "dim", color=color))
    longs = [(k, LONG_CONTEXT[k]) for k, _ in items if k in LONG_CONTEXT]
    if longs:
        _out(term.paint("Long prompts are billed at a higher rate for the whole request:", "dim", color=color))
        for k, (above, lp) in longs:
            _out(term.paint(f"  {k}: prompts over {above // 1000}K tokens cost "
                            f"{money(lp['input'])} in / {money(lp['output'])} out", "dim", color=color))
    return 0


def _line_col(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    col = offset - (text.rfind("\n", 0, offset) + 1) + 1
    return line, col


def cmd_scan(args: argparse.Namespace) -> int:
    from llmtap import guard, term
    kinds: dict[str, str] = dict(getattr(guard, "KINDS", {}) or {})
    files = args.files or ["-"]
    color = _color_out()
    p = lambda s, *st: term.paint(s, *st, color=color)  # noqa: E731
    total = 0
    report: list[dict[str, Any]] = []
    for name in files:
        try:
            if name == "-":
                buf = getattr(sys.stdin, "buffer", None)  # not the locale codec: scan any bytes
                text = buf.read().decode("utf-8", errors="replace") if buf is not None else sys.stdin.read()
                label = "<stdin>"
            else:
                with open(name, "rb") as fh:
                    raw = fh.read()
                # UTF-16 with a BOM (Windows PowerShell's `>`) would otherwise hide every secret behind NULs
                utf16 = raw.startswith((b"\xff\xfe", b"\xfe\xff"))
                text = raw.decode("utf-16" if utf16 else "utf-8-sig", errors="replace")
                text = text.replace("\r\n", "\n").replace("\r", "\n")  # universal newlines, as before
                label = name
        except OSError as e:
            _err(f"{name}: {e.strerror or e}")
            return 2
        findings = guard.scan_text(text, pii=args.pii)
        total += len(findings)
        for f in findings:
            ln, col = _line_col(text, f.start)
            d = f.to_dict() if hasattr(f, "to_dict") else dict(vars(f))
            d.update({"file": label, "line": ln, "col": col})
            report.append(d)
            if not args.json:
                desc = kinds.get(f.kind, "")
                _out(f"{p(f'{label}:{ln}:{col}', 'bold')}  {p(f.kind, 'bold', 'red')}  {term.clean(f.preview)}"
                     + (p(f"  — {desc}", "dim") if desc else ""))
    if args.json:
        _out(json.dumps(report, indent=2))
    elif total:
        _out(p(f"\n⚠ {total} potential secret{'s' if total != 1 else ''} found", "bold", "red"))
    else:
        _out(p("✓ no secrets found", "green"))
    return 1 if total else 0


def cmd_clear(args: argparse.Namespace) -> int:
    store = _open_store(args)
    try:
        n = store.count()
        if not args.yes:
            if not sys.stdin.isatty():
                _err(f"refusing to delete {n} exchange(s) without --yes")
                return 1
            try:
                ans = input(f"Delete all {n} recorded exchange(s)? [y/N] ").strip().lower()
            except EOFError:
                ans = ""
            if ans not in ("y", "yes"):
                _out("aborted")
                return 1
        store.clear()
    finally:
        store.close()
    _out(f"cleared {n} exchange(s)")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from llmtap import term
    from llmtap.config import load_config
    from llmtap.demo import generate_traffic
    from llmtap.mock import MockUpstream
    from llmtap.proxy import LLMTap

    mock = MockUpstream(token_delay_s=0.01).start()
    tmp = tempfile.mkdtemp(prefix="llmtap-demo-")
    cfg = load_config(getattr(args, "config", None), {
        "host": args.host, "port": args.port, "quiet": args.quiet,
        "db_path": os.path.join(tmp, "demo.db"),
        "cache": True, "guard": "redact", "loop_threshold": 4, "loop_window_s": 60,
        # a project's llmtap.toml / LLMTAP_* must not record fake traffic into its cassette or cap the demo
        "cassette": "", "budget_daily_usd": 0.0, "budget_session_usd": 0.0,
        "upstreams": {"anthropic": {"base": mock.base_url, "style": "anthropic"},
                      "openai": {"base": mock.base_url, "style": "openai"}},
    })
    try:
        tap = LLMTap(cfg)
        tap.start_background()
    except OSError as e:
        mock.stop()
        _err(f"cannot listen on {cfg.host}:{cfg.port}: {e.strerror or e} (try --port 0)")
        return 1
    base = getattr(tap, "base_url", None) or cfg.base_url()
    print(term.banner(cfg, base, color=term.color_enabled(sys.stdout)), flush=True)
    print(term.paint(f"demo: fake LLM upstream at {mock.base_url} · no API keys, no network, no cost", "dim",
                     color=term.color_enabled(sys.stdout)), flush=True)
    stop = threading.Event()
    done = threading.Event()

    def run() -> None:
        try:
            generate_traffic(base, count=args.count, delay_s=args.delay, stop=stop)
        except Exception as e:  # pragma: no cover
            _err(f"demo traffic failed: {e}")
        finally:
            done.set()

    start = time.time() - 0.001
    threading.Thread(target=run, daemon=True, name="llmtap-demo").start()
    if not args.no_open:
        try:
            import webbrowser
            webbrowser.open(base + "/_llmtap/")
        except Exception:
            pass
    try:
        if args.exit:
            done.wait()
        else:
            _wait_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        time.sleep(0.2)  # let in-flight records land
        try:
            st = tap.store.stats(since=start, group_by="model")
            print(term.summary(st, float(tap.budget.session_spent()), color=term.color_enabled(sys.stderr),
                               title="llmtap demo summary"), file=sys.stderr, flush=True)
        except Exception:
            pass
        _shutdown(tap)
        mock.stop()
    return 0


_REPORT_EXT = {".md": "md", ".markdown": "md", ".json": "json", ".html": "html", ".htm": "html"}


def cmd_report(args: argparse.Namespace) -> int:
    """`llmtap report`: build the cost & safety report and write/print it."""
    from llmtap import report, term
    out = args.output
    fmt = args.format
    if fmt is None:
        ext = os.path.splitext(out)[1].lower() if out and out != "-" else ""
        fmt = _REPORT_EXT.get(ext, "html")
    store = _open_store(args)
    try:
        data = report.build_report(store, since=args.since)
    finally:
        store.close()
    text = {"html": report.render_html, "md": report.render_markdown, "json": report.render_json}[fmt](data)
    if out is None:
        out = "llmtap-report.html" if fmt == "html" else "-"
    if out == "-":
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        sys.stdout.flush()
        return 0
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        _err(f"cannot write {out}: {e.strerror or e}")
        return 1
    t, w = data["totals"], data["waste"]
    color = _color_out()
    path = os.path.abspath(out)
    _out(f"{term.paint('✓', 'bold', 'green', color=color)} wrote {out} "
         + term.paint(f"({fmt}, {t['requests']:,} requests, {term.fmt_usd(t['cost_usd'])} spend", "dim", color=color)
         + (term.paint(f", {term.fmt_usd(w['savings_usd'])} avoidable waste", "yellow", color=color) if w["savings_usd"] > 0 else "")
         + term.paint(")", "dim", color=color))
    if fmt == "html":
        # as_uri() gives file:///C:/... on Windows and percent-encodes spaces; "file://" + path did neither.
        uri = Path(path).resolve().as_uri()
        _out(term.paint("  " + uri, "dim", color=color))
        if args.open:
            try:
                import webbrowser
                webbrowser.open(uri)
            except Exception:
                pass
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """`llmtap doctor`: print checks with fix hints; exit 1 on a hard failure."""
    from llmtap import doctor
    from llmtap.config import Config
    try:
        cfg, problem = make_config(args), ""
    except ValueError as e:  # a broken config file is something to report, not a crash
        cfg, problem = Config(), str(e)
    checks = doctor.run_checks(cfg, getattr(args, "config", None), online=args.online)
    if problem and not any(c.level == doctor.FAIL and c.title.startswith("Config file") for c in checks):
        checks.append(doctor.Check(doctor.FAIL, "Invalid settings", problem, "fix the value named above"))
    _out(doctor.render(checks, color=_color_out()))
    return doctor.exit_code(checks)


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "serve": cmd_serve, "wrap": cmd_wrap, "env": cmd_env, "stats": cmd_stats, "ls": cmd_ls,
    "show": cmd_show, "tail": cmd_tail, "export": cmd_export, "prices": cmd_prices, "scan": cmd_scan,
    "clear": cmd_clear, "version": cmd_version, "demo": cmd_demo,
}

# Added later; kept apart so the core table (and tests that pin it) stay stable.
EXTRA_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "report": cmd_report, "doctor": cmd_doctor,
}


def _split_wrap(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """For ``wrap``, everything after the first ``--`` is the command (kept verbatim)."""
    if "wrap" in argv and "--" in argv and argv.index("wrap") < argv.index("--"):
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse argv (default sys.argv[1:]) into a Namespace; ``wrap`` gets ``.cmd``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    head, tail = _split_wrap(argv)
    args = build_parser().parse_args(head)
    if tail is not None:
        args.cmd = tail
    return args


def _ensure_unicode_output() -> None:
    """Switch stdout/stderr to UTF-8 when their encoding can't represent llmtap's output.

    On Windows, output redirected to a file or pipe (CI logs, ``> out.txt``) uses the ANSI code page
    (e.g. cp1252), which has no box-drawing or check-mark characters, so printing a table crashed.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            "┌─✓".encode(getattr(stream, "encoding", None) or "ascii")
        except (LookupError, UnicodeEncodeError):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``llmtap`` console script."""
    _ensure_unicode_output()
    try:
        args = parse_args(argv)
    except SystemExit as e:  # argparse --help / errors
        return int(e.code or 0) if isinstance(e.code, int) or e.code is None else 2
    if not getattr(args, "command", None):
        build_parser().print_help()
        return 0
    fn = COMMANDS.get(args.command) or EXTRA_COMMANDS[args.command]
    try:
        return int(fn(args) or 0)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        try:
            sys.stdout = open(os.devnull, "w")
        except Exception:
            pass
        return 0
    except (ImportError, ModuleNotFoundError) as e:
        _err(f"`{args.command}` is unavailable: {e}")
        return 1
    except ValueError as e:
        _err(str(e))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
