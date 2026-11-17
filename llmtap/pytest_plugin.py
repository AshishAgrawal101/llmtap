"""pytest plugin: record real LLM calls once, replay them forever (free, deterministic, keyless).

Registered through the ``pytest11`` entry point, so ``pip install llmtap-proxy`` is all it takes. The plugin is
inert until a test requests the ``llmtap`` fixture or carries ``@pytest.mark.llmtap``: no servers, no env
changes. The proxy modules are imported lazily, only when the fixture actually runs.

Usage::

    def test_summary(llmtap):                 # cassettes/test_mod__test_summary.jsonl next to the file
        client = anthropic.Anthropic()        # ANTHROPIC_BASE_URL already points at the in-process proxy
        ...

    @pytest.mark.llmtap(cassette="shared", mode="replay", guard="redact", allow_miss=False)
    def test_other(): ...                     # the marker alone requests the fixture

Command line: ``--llmtap-mode {auto,record,replay}``, ``--llmtap-record``, ``--llmtap-cassette-dir DIR``;
ini: ``llmtap_cassette_dir``. Default mode is ``replay`` when ``$CI`` is truthy, else ``auto``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pytest

__all__ = ["LLMTapSession"]

MODES = ("auto", "record", "replay")
GUARDS = ("off", "warn", "redact", "block")
_MARKER_KEYS = ("cassette", "mode", "guard", "allow_miss")
_FAKE_KEY = "llmtap-fake-key-for-replay"
_ENV_FALSY = ("", "0", "false", "no", "off")
_MAX_NAME = 120


# ------------------------------------------------------------------------------------------------
# session-wide bookkeeping (feeds the terminal summary)
# ------------------------------------------------------------------------------------------------
@dataclass
class _Totals:
    used: bool = False
    calls: int = 0
    cost: float = 0.0
    saved: float = 0.0
    cassettes: set[str] = field(default_factory=set)
    misses: list[str] = field(default_factory=list)

    def merge(self, d: dict[str, Any]) -> None:
        self.used = self.used or bool(d.get("used"))
        self.calls += int(d.get("calls", 0))
        self.cost += float(d.get("cost", 0.0))
        self.saved += float(d.get("saved", 0.0))
        self.cassettes.update(d.get("cassettes", []))
        self.misses.extend(d.get("misses", []))

    def dump(self) -> dict[str, Any]:
        return {"used": self.used, "calls": self.calls, "cost": self.cost, "saved": self.saved,
                "cassettes": sorted(self.cassettes), "misses": list(self.misses)}


_TOTALS_KEY = pytest.StashKey[_Totals]()
_REPORTS_KEY = pytest.StashKey[dict]()


def _totals(config: pytest.Config) -> _Totals:
    t = config.stash.get(_TOTALS_KEY, None)
    if t is None:
        t = config.stash[_TOTALS_KEY] = _Totals()
    return t


# ------------------------------------------------------------------------------------------------
# options, marker
# ------------------------------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("llmtap", "llmtap: record/replay LLM calls")
    g.addoption("--llmtap-mode", choices=MODES, default=None, dest="llmtap_mode",
                help="cassette mode: auto (replay hits, record misses), record (re-record), replay "
                     "(misses fail the test). Default: replay if $CI is set, else auto.")
    g.addoption("--llmtap-record", action="store_true", default=False, dest="llmtap_record",
                help="shortcut for --llmtap-mode record: re-record every cassette that is used")
    g.addoption("--llmtap-cassette-dir", default=None, dest="llmtap_cassette_dir",
                help="store all cassettes in this directory instead of <test dir>/cassettes")
    parser.addini("llmtap_cassette_dir", "directory for llmtap cassettes (relative to the rootdir)", default="")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "llmtap(cassette=None, mode=None, guard=None, allow_miss=False): record/replay this test's LLM calls "
        "with llmtap (requests the `llmtap` fixture automatically)",
    )


def _ci_mode_default() -> str:
    return "auto" if os.environ.get("CI", "").strip().lower() in _ENV_FALSY else "replay"


def _marker_kwargs(item: pytest.Item) -> dict[str, Any]:
    """Validated kwargs of the closest ``llmtap`` marker ({} if none)."""
    m = item.get_closest_marker("llmtap")
    if m is None:
        return {}
    if m.args:
        if len(m.args) == 1 and isinstance(m.args[0], str):
            kw: dict[str, Any] = {"cassette": m.args[0]}
        else:
            raise pytest.UsageError("@pytest.mark.llmtap takes keyword arguments: "
                                    + ", ".join(_MARKER_KEYS))
    else:
        kw = {}
    kw.update(m.kwargs)
    unknown = sorted(set(kw) - set(_MARKER_KEYS))
    if unknown:
        raise pytest.UsageError(f"@pytest.mark.llmtap: unknown argument(s) {', '.join(unknown)} "
                                f"(expected {', '.join(_MARKER_KEYS)})")
    if kw.get("mode") is not None and kw["mode"] not in MODES:
        raise pytest.UsageError(f"@pytest.mark.llmtap: mode must be one of {', '.join(MODES)}, got {kw['mode']!r}")
    if kw.get("guard") is not None and kw["guard"] not in GUARDS:
        raise pytest.UsageError(f"@pytest.mark.llmtap: guard must be one of {', '.join(GUARDS)}, got {kw['guard']!r}")
    return {k: v for k, v in kw.items() if v is not None}


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """The marker alone is enough: pull the fixture into the item's closure before fixtures run."""
    if item.get_closest_marker("llmtap") is not None:
        names = getattr(item, "fixturenames", None)
        if names is not None and "llmtap" not in names:
            names.append("llmtap")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[None]:
    report = yield
    if "llmtap" in getattr(item, "fixturenames", ()):
        item.stash.setdefault(_REPORTS_KEY, {})[report.when] = report
    return report


# ------------------------------------------------------------------------------------------------
# naming
# ------------------------------------------------------------------------------------------------
_SAFE = re.compile(r"[^A-Za-z0-9_.\-]+")


def _sanitize(text: str) -> tuple[str, bool]:
    """(filename-safe text, whether anything was lost)."""
    out = _SAFE.sub("_", text).strip("._") or "_"
    return out, out != text


def _short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:6]


def _test_label(item: pytest.Item) -> str:
    """``Class.test_name__param-ids``: stable, filename-safe, and distinct per parametrization."""
    name = getattr(item, "originalname", None) or item.name
    cls = getattr(item, "cls", None)
    base = f"{cls.__name__}.{name}" if cls is not None else name
    callspec = getattr(item, "callspec", None)
    label, lossy = _sanitize(base)
    if callspec is not None:
        params, plossy = _sanitize(callspec.id)
        label = f"{label}__{params}"
        lossy = lossy or plossy
    if lossy or len(label) > _MAX_NAME:  # disambiguate names that collapsed or were cut short
        label = f"{label[:_MAX_NAME]}-{_short(item.name)}"
    return label


def _cassette_dir(item: pytest.Item) -> tuple[Path, bool]:
    """(directory, is_custom). Custom dirs come from --llmtap-cassette-dir or the ini file."""
    cfg = item.config
    opt = cfg.getoption("llmtap_cassette_dir", None)
    if opt:
        p = Path(opt)
        return (p if p.is_absolute() else Path(cfg.invocation_params.dir) / p), True
    ini = cfg.getini("llmtap_cassette_dir")
    if ini:
        p = Path(str(ini))
        return (p if p.is_absolute() else Path(cfg.rootpath) / p), True
    return Path(str(item.path)).parent / "cassettes", False


def _cassette_path(item: pytest.Item, override: str | None) -> Path:
    directory, custom = _cassette_dir(item)
    if override:
        name = override if override.endswith(".jsonl") else override + ".jsonl"
        p = Path(name)
        return p if p.is_absolute() else directory / p
    path = Path(str(item.path))
    module = path.stem
    if custom:  # many test dirs share one cassette dir: keep the module's location in the name
        try:
            rel = path.resolve().relative_to(Path(item.config.rootpath).resolve()).with_suffix("")
            module = ".".join(rel.parts)
        except ValueError:
            pass
    module, _ = _sanitize(module)
    return directory / f"{module}__{_test_label(item)}.jsonl"


def _resolve_mode(item: pytest.Item, marker_mode: str | None) -> str:
    if marker_mode:
        return marker_mode
    cfg = item.config
    if cfg.getoption("llmtap_record", False):
        return "record"
    return cfg.getoption("llmtap_mode", None) or _ci_mode_default()


# ------------------------------------------------------------------------------------------------
# the object tests receive
# ------------------------------------------------------------------------------------------------
def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and isinstance(b.get("text"), str):
                parts.append(b["text"])
        return "\n".join(parts)
    return ""


def _last_user_text(req: Any) -> str:
    if not isinstance(req, dict):
        return ""
    for key in ("messages", "input"):
        val = req.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            for m in reversed(val):
                if isinstance(m, dict) and m.get("role", "user") == "user" and "content" in m:
                    t = _content_text(m["content"])
                    if t:
                        return t
                elif isinstance(m, str):
                    return m
    p = req.get("prompt")
    return p if isinstance(p, str) else ""


class LLMTapSession:
    """What the ``llmtap`` fixture yields: the running proxy, its cassette and the call ledger."""

    def __init__(self, tap: Any, cassette_path: Path, mode: str, allow_miss: bool) -> None:
        self.tap = tap
        self.cassette_path = cassette_path
        self.mode = mode
        self.allow_miss = allow_miss
        # The proxy answers the client *before* it stores the exchange, so a test reading the ledger
        # right after an SDK call could miss the last one. Track in-flight requests to wait for them.
        self._cond = threading.Condition()
        self._inflight = 0
        orig = tap.handle

        def handle(*args: Any, **kwargs: Any) -> Any:
            with self._cond:
                self._inflight += 1
            try:
                return orig(*args, **kwargs)
            finally:
                with self._cond:
                    self._inflight -= 1
                    self._cond.notify_all()

        tap.handle = handle

    def _settle(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._cond.wait_for(lambda: self._inflight == 0, timeout)

    @property
    def base_url(self) -> str:
        return str(self.tap.base_url)

    def exchanges(self) -> list[Any]:
        """Every call that went through the proxy this test, oldest first."""
        self._settle()
        return list(reversed(self.tap.store.list(limit=1_000_000)))

    @property
    def calls(self) -> int:
        return len(self.exchanges())

    @property
    def cost_usd(self) -> float:
        """Money actually spent upstream by this test (0 when everything was replayed)."""
        return sum(float(e.cost_usd or 0.0) for e in self.exchanges())

    @property
    def saved_usd(self) -> float:
        """Money avoided because responses came from the cassette."""
        return sum(float(e.saved_usd or 0.0) for e in self.exchanges())

    def misses(self) -> list[Any]:
        """Replay-mode requests that had no recording (the SDK saw HTTP 409)."""
        return [e for e in self.exchanges()
                if e.status == 409 and "cassette miss" in (e.error or e.response_body or "")]

    def describe_misses(self) -> str:
        lines = []
        for ex in self.misses():
            try:
                req = json.loads(ex.request_body)
            except ValueError:
                req = None
            last = " ".join(_last_user_text(req).split())
            last = last[:80] + ("..." if len(last) > 80 else "")
            lines.append(f"  - model={ex.model or '?'} {ex.path}: last user message: {last!r}")
        exists = "" if self.cassette_path.is_file() else " (the cassette file does not exist yet)"
        return (f"llmtap: {len(lines)} LLM request(s) had no recording in {self.cassette_path}{exists}:\n"
                + "\n".join(lines)
                + "\nre-run with --llmtap-record to re-record this cassette (needs real API keys).")


# ------------------------------------------------------------------------------------------------
# the fixture
# ------------------------------------------------------------------------------------------------
@pytest.fixture()
def llmtap(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[LLMTapSession]:
    """Run an in-process llmtap proxy backed by this test's cassette and point the LLM SDKs at it."""
    # lazy imports: nothing of llmtap beyond this module is loaded unless a test uses the fixture
    from .config import load_config
    from .proxy import LLMTap

    item = request.node
    kw = _marker_kwargs(item)
    mode = _resolve_mode(item, kw.get("mode"))
    path = _cassette_path(item, kw.get("cassette"))

    cfg = load_config()  # picks up upstreams/pricing from llmtap.toml; everything else is set below
    cfg.host, cfg.port, cfg.db_path, cfg.quiet, cfg.cache = "127.0.0.1", 0, ":memory:", True, False
    cfg.cassette, cfg.mode = str(path), mode
    cfg.allow_miss = bool(kw.get("allow_miss", False))
    if kw.get("guard"):
        cfg.guard = kw["guard"]

    tap = LLMTap(cfg).start_background()
    session = LLMTapSession(tap, path, mode, cfg.allow_miss)
    try:
        base = session.base_url
        monkeypatch.setenv("ANTHROPIC_BASE_URL", base)
        monkeypatch.setenv("OPENAI_BASE_URL", base + "/v1")
        monkeypatch.setenv("OPENAI_API_BASE", base + "/v1")
        monkeypatch.setenv("LLMTAP_URL", base)
        recorded = tap.cassette is not None and len(tap.cassette) > 0
        if mode == "replay" or (mode == "auto" and recorded):  # SDK clients need *a* key to construct
            for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
                if not os.environ.get(var):
                    monkeypatch.setenv(var, _FAKE_KEY)
        yield session
        _finish(request, session)
    finally:
        # runs even if _finish() raised the miss failure
        _stop(tap)


def _stop(tap: Any) -> None:
    """tap.shutdown(), without paying the server's 0.25s poll interval for every single test."""
    t = threading.Thread(target=tap.shutdown, name="llmtap-shutdown", daemon=True)
    t.start()
    addr = (tap.cfg.host if tap.cfg.host not in ("", "0.0.0.0", "::") else "127.0.0.1", tap.port)
    for _ in range(100):  # a throwaway connection wakes serve_forever's select() so it sees the flag
        t.join(0.01)
        if not t.is_alive():
            return
        try:
            socket.create_connection(addr, timeout=0.2).close()
        except OSError:
            pass
    t.join(10)


def _finish(request: pytest.FixtureRequest, session: LLMTapSession) -> None:
    """Teardown: update session totals, then fail the test on replay misses."""
    item = request.node
    exchanges = session.exchanges()
    totals = _totals(item.config)
    totals.used = True
    totals.calls += len(exchanges)
    totals.cost += sum(float(e.cost_usd or 0.0) for e in exchanges)
    totals.saved += sum(float(e.saved_usd or 0.0) for e in exchanges)
    if exchanges or session.cassette_path.is_file():
        totals.cassettes.add(str(session.cassette_path))
    if session.mode != "replay" or session.allow_miss:
        return
    misses = session.misses()
    if not misses:
        return
    msg = session.describe_misses()
    totals.misses.append(f"{item.nodeid}\n{msg}")
    call = item.stash.get(_REPORTS_KEY, {}).get("call")
    if call is not None and call.failed:
        return  # the test already failed: don't mask its own error (the miss is in the terminal summary)
    pytest.fail(msg, pytrace=False)


# ------------------------------------------------------------------------------------------------
# terminal summary (+ pytest-xdist aggregation)
# ------------------------------------------------------------------------------------------------
def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    t = config.stash.get(_TOTALS_KEY, None)
    if t is None or not t.used:
        return
    from .pricing import format_usd

    n = len(t.cassettes)
    terminalreporter.section("llmtap")
    terminalreporter.write_line(
        f"llmtap: {t.calls} call{'s' if t.calls != 1 else ''}, {format_usd(t.cost)} spent, "
        f"{format_usd(t.saved)} saved by replay, {n} cassette{'s' if n != 1 else ''}")
    for m in t.misses:
        terminalreporter.write_line("")
        terminalreporter.write_line(m, red=True)


def pytest_sessionfinish(session: pytest.Session) -> None:
    out = getattr(session.config, "workeroutput", None)  # xdist worker: ship totals to the controller
    t = session.config.stash.get(_TOTALS_KEY, None)
    if out is not None and t is not None and t.used:
        out["llmtap_totals"] = t.dump()


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: Any) -> None:
    """xdist controller: fold a finished worker's totals into the session's."""
    data = getattr(node, "workeroutput", {}).get("llmtap_totals")
    if data:
        _totals(node.config).merge(data)
