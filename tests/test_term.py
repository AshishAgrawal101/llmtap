"""Tests for llmtap.term: ANSI handling, width math, formatters, banner/summary/table, log lines."""
from __future__ import annotations

import io

import pytest

from llmtap import term
from llmtap.config import Config
from llmtap.models import Exchange, Usage


@pytest.fixture(autouse=True)
def _no_force(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("COLUMNS", "120")


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_strip_ansi_and_vlen() -> None:
    s = "\x1b[1;31mhello\x1b[0m world"
    assert term.strip_ansi(s) == "hello world"
    assert term.vlen(s) == 11
    assert term.vlen("⚡ttft") == 6  # wide glyph counts as 2
    assert term.vlen("") == 0


def test_pad_and_truncate() -> None:
    red = term.paint("ab", "red", color=True)
    assert term.vlen(term.pad(red, 5)) == 5
    assert term.pad("ab", 5, ">") == "   ab"
    assert term.pad("ab", 6, "^") == "  ab  "
    assert term.pad("abcdef", 3) == "abcdef"
    assert term.truncate("abcdefghij", 5) == "abcd…"
    assert term.truncate("abc", 5) == "abc"
    assert term.vlen(term.truncate(term.paint("x" * 50, "red", color=True), 10)) == 10


def test_paint_respects_color_flag() -> None:
    assert term.paint("x", "red", color=False) == "x"
    assert term.paint("x", "red", color=True) == "\x1b[31mx\x1b[0m"
    assert term.paint("x", color=True) == "x"


def test_color_enabled_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    assert term.color_enabled(io.StringIO()) is False
    assert term.color_enabled(_TTY()) is True
    monkeypatch.setenv("NO_COLOR", "1")
    assert term.color_enabled(_TTY()) is False
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert term.color_enabled(io.StringIO()) is True
    monkeypatch.setenv("FORCE_COLOR", "0")
    assert term.color_enabled(io.StringIO()) is False


@pytest.mark.parametrize("n,expected", [
    (0, "0"), (999, "999"), (1000, "1k"), (1234, "1.2k"), (12_345, "12.3k"), (340, "340"),
    (2_500_000, "2.5M"), (150_000, "150k"), (None, "0"), (3_000_000_000, "3B"),
])
def test_fmt_tokens(n: int | None, expected: str) -> None:
    assert term.fmt_tokens(n) == expected


@pytest.mark.parametrize("ms,expected", [
    (210, "210ms"), (1240, "1.24s"), (0, "0ms"), (None, "-"), (12_500, "12.5s"), (75_000, "1m15s"), (0.5, "0.5ms"),
])
def test_fmt_ms(ms: float | None, expected: str) -> None:
    assert term.fmt_ms(ms) == expected


def test_fmt_usd() -> None:
    assert term.fmt_usd(0) == "$0.00"
    assert term.fmt_usd(0.0042) == "$0.0042"
    assert term.fmt_usd(1.234) == "$1.23"


def test_table_alignment_ignores_ansi() -> None:
    out = term.table([[term.paint("red", "red", color=True), "1"], ["longer-cell", "22"]], ["NAME", "N"], color=False)
    lines = out.splitlines()
    widths = {term.vlen(l) for l in lines}
    assert len(widths) == 1, out
    assert lines[0].startswith("┌") and lines[-1].startswith("└")
    assert "│  1 │" in term.strip_ansi(out)  # numeric column right-aligned


def test_table_empty_and_ragged() -> None:
    out = term.table([], ["A", "B"], color=False)
    assert "A" in out and "B" in out
    out = term.table([["x"], ["y", "z", "w"]], ["A"], color=False)
    assert len({term.vlen(l) for l in out.splitlines()}) == 1


def test_table_shrinks_to_width() -> None:
    out = term.table([["x" * 300, "y"]], ["A", "B"], color=False, max_width=60)
    assert all(term.vlen(l) <= 60 for l in out.splitlines())


def test_box_is_rectangular() -> None:
    out = term.box(["hello", term.paint("colored ⚡", "red", color=True), ""], title="T", color=True)
    assert len({term.vlen(l) for l in out.splitlines()}) == 1


def test_banner_contents() -> None:
    cfg = Config(cache=True, cassette="tapes/ci.jsonl", mode="replay", guard="redact",
                 budget_daily_usd=5, loop_threshold=4)
    out = term.strip_ansi(term.banner(cfg, "http://127.0.0.1:4141", color=False))
    assert "llmtap v0.1.0" in out and "recording LLM traffic" in out
    assert "http://127.0.0.1:4141/_llmtap/" in out
    assert "export ANTHROPIC_BASE_URL=http://127.0.0.1:4141" in out
    assert "export OPENAI_BASE_URL=http://127.0.0.1:4141/v1" in out
    assert "redact" in out and "ci.jsonl" in out and "replay" in out
    assert "$5.00/day" in out and "4×" in out
    lines = out.splitlines()
    assert len({term.vlen(l) for l in lines}) == 1


def test_banner_defaults_off() -> None:
    out = term.banner(Config(), "http://127.0.0.1:4141/", color=False)
    assert "cache off" in out and "unlimited" in out and "loop-breaker off" in out


def test_log_exchange_minimal_does_not_crash() -> None:
    buf = io.StringIO()
    term.log_exchange(Exchange(), stream=buf)
    line = buf.getvalue()
    assert line.endswith("\n") and "\x1b[" not in line
    assert "unknown" in line


def test_log_exchange_full_line() -> None:
    ex = Exchange(provider="anthropic", model="claude-sonnet-4-5", status=200,
                  usage=Usage(input_tokens=1234, output_tokens=340), cost_usd=0.0123,
                  latency_ms=1240, streamed=True, ttft_ms=210,
                  guard_findings=[{"kind": "aws_access_key"}, {"kind": "openai_key"}],
                  request_body='{"x": "[REDACTED:aws_access_key]"}')
    line = term.format_exchange(ex, color=False, width=120)
    assert "anthropic/claude-sonnet-4-5" in line
    assert "1.2k→340 tok" in line
    assert "$0.0123" in line and "1.24s" in line
    assert "⚡ttft 210ms" in line
    assert "⚠ 2 secrets redacted" in line
    assert "LIVE" in line and "200" in line


def test_log_exchange_blocked_and_cache() -> None:
    ex = Exchange(source="blocked", status=400, guard_findings=[{"kind": "x"}])
    assert "⛔ blocked" in term.format_exchange(ex, color=False)
    ex = Exchange(source="cache", status=200, saved_usd=0.5, model="gpt-4o")
    line = term.format_exchange(ex, color=False)
    assert "CACHE" in line and "saved $0.5" in line
    ex = Exchange(source="loop", status=429, error="loop detected: identical request sent 4 times in 60s")
    assert "loop detected" in term.format_exchange(ex, color=False)


def test_log_exchange_truncates_long_model() -> None:
    ex = Exchange(provider="openrouter", model="m" * 200, status=200)
    line = term.format_exchange(ex, color=False, width=100)
    assert "…" in line and term.vlen(line) < 160


def test_log_exchange_with_colors_aligned() -> None:
    a = Exchange(provider="openai", model="gpt-4o", status=200, latency_ms=12)
    b = Exchange(provider="anthropic", model="claude-haiku-4-5", status=500, latency_ms=3000)
    la = term.format_exchange(a, color=True, width=120)
    lb = term.format_exchange(b, color=True, width=120)
    assert "\x1b[" in la
    assert term.vlen(la) == term.vlen(lb)


def test_status_color() -> None:
    assert term.status_color(200) == "green"
    assert term.status_color(429) == "yellow"
    assert term.status_color(502) == "red"
    assert term.status_color(0) == "red"


def test_summary_box() -> None:
    stats = {"requests": 12, "input_tokens": 12_000, "output_tokens": 3400, "cost_usd": 0.42, "saved_usd": 0.1,
             "avg_latency_ms": 850.0, "errors": 1, "cache_hits": 3, "blocked": 1, "guard_findings": 2,
             "groups": [{"key": "claude-sonnet-4-5", "requests": 8, "cost_usd": 0.4, "input_tokens": 10_000,
                         "output_tokens": 3000},
                        {"key": "gpt-4o-mini", "requests": 4, "cost_usd": 0.02, "input_tokens": 2000,
                         "output_tokens": 400}]}
    out = term.summary(stats, 0.42, color=False)
    for needle in ("12", "12k in", "3.4k out", "$0.42", "$0.10", "850ms", "Errors", "Blocked",
                   "2 secret findings", "claude-sonnet-4-5", "gpt-4o-mini", "Top models"):
        assert needle in out, needle
    assert len({term.vlen(l) for l in out.splitlines()}) == 1


def test_summary_empty() -> None:
    out = term.summary({}, 0.0, color=False)
    assert "No LLM requests" in out
    out = term.summary({"requests": 0}, 0.0, color=False)
    assert "No LLM requests" in out
