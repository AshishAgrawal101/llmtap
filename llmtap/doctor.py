"""`llmtap doctor`: environment checks with fix hints. Never prints secret values."""
from __future__ import annotations

import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from llmtap import __version__

OK, FAIL, WARN, INFO = "ok", "fail", "warn", "info"
_GLYPH = {OK: "✓", FAIL: "✗", WARN: "!", INFO: "·"}
_COLOR = {OK: "green", FAIL: "red", WARN: "yellow", INFO: "gray"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


@dataclass
class Check:
    level: str            # ok | fail | warn | info
    title: str
    detail: str = ""
    hint: str = ""


# --------------------------------------------------------------------------------------------
# individual checks (small, monkeypatch-friendly)

def check_python() -> Check:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 11):
        return Check(OK, f"Python {ver}")
    return Check(FAIL, f"Python {ver}", "llmtap needs Python 3.11 or newer",
                 "install Python 3.11+ (e.g. `pyenv install 3.12`) and reinstall llmtap")


def _listener_info(port: int) -> str:
    """Best-effort 'who is listening' via ss/lsof; empty string when unknown."""
    for cmd in (["ss", "-ltnpH", f"sport = :{port}"], ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"]):
        if not shutil.which(cmd[0]):
            continue
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout.strip()
        except Exception:
            continue
        if out:
            line = out.splitlines()[-1]
            if "users:" in line:
                return line[line.index("users:"):][:80]
            return " ".join(line.split())[:80]
    return ""


def _port_in_use(host: str, port: int) -> bool:
    """True when something is already bound to host:port."""
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    s = socket.socket(fam, socket.SOCK_STREAM)
    try:
        s.settimeout(1.0)
        s.bind((host, port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def _http_base(host: str, port: int) -> str:
    """``http://host:port`` for a bind address (wildcards -> loopback, IPv6 literals bracketed)."""
    h = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}.get(host, host)
    if ":" in h and not h.startswith("["):
        h = f"[{h}]"
    return f"http://{h}:{port}"


def _is_llmtap(host: str, port: int) -> str | None:
    """Return the llmtap version if /_llmtap/health answers on host:port."""
    import json
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # local: never via HTTP_PROXY
        with opener.open(f"{_http_base(host, port)}/_llmtap/health", timeout=1.5) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        if isinstance(data, dict) and data.get("ok"):
            return str(data.get("version") or "?")
    except Exception:
        pass
    return None


def check_port(host: str, port: int) -> Check:
    if port == 0:
        return Check(OK, "Port 0 (any free port)")
    if not _port_in_use(host, port):
        return Check(OK, f"Port {port} is free")
    ver = _is_llmtap(host, port)
    if ver is not None:
        return Check(OK, f"Port {port}: llmtap v{ver} is already running", f"dashboard: {_http_base(host, port)}/_llmtap/")
    who = _listener_info(port)
    return Check(WARN, f"Port {port} is in use" + (f" ({who})" if who else ""), "",
                 f"stop that process, or run `llmtap serve --port {port + 1}`")


def _db_path(cfg: Any) -> Path:
    if cfg.db_path:
        return Path(cfg.db_path).expanduser()
    from llmtap.config import llmtap_home
    return llmtap_home() / "llmtap.db"


def check_db(cfg: Any) -> Check:
    if cfg.db_path == ":memory:":
        return Check(OK, "Database: in-memory (nothing persisted)")
    p = _db_path(cfg)
    probe = p if p.exists() else p.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not os.access(probe, os.W_OK):
        return Check(FAIL, f"Database path not writable: {p}", "", f"fix permissions (`chmod u+w {probe}`) or use `--db PATH`")
    detail = ""
    if p.is_file():
        try:
            # as_uri() percent-encodes '#', '?' and '%', which a raw "file:{p}" URI would misparse
            c = sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
            try:
                n = c.execute("SELECT COUNT(*) FROM exchanges").fetchone()[0]
                detail = f"{n:,} exchanges recorded"
            finally:
                c.close()
        except sqlite3.Error as e:
            return Check(FAIL, f"Database unreadable: {p}", str(e), "move the file aside or point `--db` elsewhere")
    return Check(OK, f"Database writable: {p}", detail)


def check_llmtap_env() -> Check:
    names = sorted(k for k in os.environ if k.startswith("LLMTAP_"))
    if not names:
        return Check(INFO, "No LLMTAP_* env vars set", "using defaults / config file")
    return Check(OK, "LLMTAP_* env vars set", ", ".join(names))


def _base_url_check(var: str, cfg: Any) -> Check:
    val = os.environ.get(var)
    if not val:
        return Check(WARN, f"{var} not set", "this SDK talks to the provider directly",
                     'run `eval "$(llmtap env)"` or `llmtap wrap -- <your command>`')
    try:
        u = urlsplit(val if "://" in val else "http://" + val)
        host, port = (u.hostname or ""), (u.port or (443 if u.scheme == "https" else 80))
    except ValueError:
        return Check(WARN, f"{var} is not a valid URL")
    shown = f"{u.scheme}://{host}" + (f":{u.port}" if u.port else "")
    if host in LOCAL_HOSTS and port in (cfg.port, 4141):
        return Check(OK, f"{var} points at llmtap", shown)
    if host in LOCAL_HOSTS:
        return Check(WARN, f"{var} points at a local server on port {port}, not llmtap's {cfg.port}", shown,
                     'run `eval "$(llmtap env)"` to point it at llmtap')
    return Check(WARN, f"{var} is set to {host}: traffic bypasses llmtap", shown,
                 'run `eval "$(llmtap env)"` (or `llmtap wrap -- <cmd>`)')


def check_key(var: str) -> Check:
    if os.environ.get(var):
        return Check(OK, f"{var} set")
    return Check(INFO, f"{var} not set", "", "export it if you call that provider (or use `llmtap demo` with no keys)")


def check_sqlite() -> Check:
    v = sqlite3.sqlite_version
    nums = tuple(int(x) for x in v.split(".")[:3] if x.isdigit())
    try:
        c = sqlite3.connect(":memory:")
        c.execute("SELECT json_valid('{}'), json_array_length('[1]')").fetchone()
        c.close()
    except sqlite3.Error:
        return Check(FAIL, f"SQLite {v} lacks JSON1 functions", "llmtap stats/report need json_* support",
                     "use a Python build linked against SQLite >= 3.38")
    if nums < (3, 45):
        return Check(OK, f"SQLite {v}", "works, but 3.45+ is recommended (newer JSON handling)")
    return Check(OK, f"SQLite {v}")


def check_config(path: str | None) -> Check:
    cands = [Path(path)] if path else [Path("llmtap.toml"), Path.home() / ".config" / "llmtap" / "config.toml"]
    for p in cands:
        if p.is_file():
            from llmtap.config import Config, apply_file
            try:
                apply_file(Config(), p)
            except (ValueError, OSError) as e:  # bad TOML / UTF-8 / value; message names file + key
                return Check(FAIL, f"Config file is invalid: {p}", str(e).removeprefix(f"{p}: "),
                             "fix the TOML syntax or the value named above")
            return Check(OK, f"Config file: {p}")
    if path:
        return Check(FAIL, f"Config file not found: {path}", "", "check the --config path")
    return Check(INFO, "No config file found", "using defaults", "copy llmtap.example.toml to llmtap.toml to customize")


def probe_host(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """HEAD (fallback GET) the host. ANY HTTP response, even 401/404/405, means reachable."""
    last = ""
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers={"User-Agent": f"llmtap-doctor/{__version__}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return True, f"HTTP {r.status}"
        except urllib.error.HTTPError as e:
            return True, f"HTTP {e.code}"
        except Exception as e:  # URLError, timeout, ssl ...
            last = getattr(e, "reason", None) or e
    return False, str(last)[:80]


def check_online(cfg: Any, timeout: float = 5.0) -> list[Check]:
    ups = {k: v for k, v in cfg.upstreams.items()}
    items = sorted(ups.items(), key=lambda kv: (urlsplit(kv[1]["base"]).hostname in LOCAL_HOSTS, kv[0]))

    def one(kv: tuple[str, dict[str, str]]) -> Check:
        name, spec = kv
        host = urlsplit(spec["base"]).hostname or spec["base"]
        ok, detail = probe_host(spec["base"], timeout)
        if ok:
            return Check(OK, f"{name} reachable ({host})", detail)
        local = host in LOCAL_HOSTS
        return Check(INFO if local else WARN, f"{name} unreachable ({host})", detail,
                     "start it locally if you use it" if local else "check network/proxy/firewall settings")

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(items)))) as ex:
        return list(ex.map(one, items))


# --------------------------------------------------------------------------------------------

def run_checks(cfg: Any, config_path: str | None = None, online: bool = False) -> list[Check]:
    checks = [
        check_python(),
        check_port(cfg.host, cfg.port),
        check_db(cfg),
        check_sqlite(),
        check_config(config_path),
        check_llmtap_env(),
        _base_url_check("ANTHROPIC_BASE_URL", cfg),
        _base_url_check("OPENAI_BASE_URL", cfg),
        check_key("ANTHROPIC_API_KEY"),
        check_key("OPENAI_API_KEY"),
    ]
    if online:
        checks.extend(check_online(cfg))
    return checks


def render(checks: list[Check], color: bool = False) -> str:
    from llmtap import term
    lines = [term.paint(f"llmtap doctor · v{__version__}", "bold", color=color), ""]
    for c in checks:
        g = term.paint(_GLYPH[c.level], "bold", _COLOR[c.level], color=color)
        line = f" {g} {c.title}"
        if c.detail:
            line += term.paint(f"  {c.detail}", "dim", color=color)
        lines.append(line)
        if c.hint and c.level != OK:
            lines.append("     " + term.paint("→ " + c.hint, "cyan", color=color))
    fails = sum(c.level == FAIL for c in checks)
    warns = sum(c.level == WARN for c in checks)
    lines.append("")
    if fails:
        lines.append(term.paint(f"{fails} problem{'s' if fails != 1 else ''} to fix", "bold", "red", color=color)
                     + (f", {warns} warning{'s' if warns != 1 else ''}" if warns else ""))
    elif warns:
        lines.append(term.paint(f"All good, with {warns} warning{'s' if warns != 1 else ''}", "bold", "yellow", color=color))
    else:
        lines.append(term.paint("All good — you're ready to tap.", "bold", "green", color=color))
    return "\n".join(lines)


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.level == FAIL for c in checks) else 0
