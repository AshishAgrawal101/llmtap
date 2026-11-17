"""Tests for llmtap.config: TOML loading, type checking and error messages."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from llmtap.config import Config, load_config


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for k in list(os.environ):
        if k.startswith("LLMTAP_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("LLMTAP_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)


def _cfg(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.toml"
    p.write_bytes(text.encode("utf-8"))
    return p


def test_valid_file_still_loads(tmp_path: Path) -> None:
    p = _cfg(tmp_path, '[llmtap]\nport = 5000\ncache = true\nguard = "block"\nbudget_daily_usd = 5\n'
                       '[llmtap.upstreams.local]\nbase = "http://localhost:9/"\n')
    cfg = load_config(str(p))
    assert cfg.port == 5000 and cfg.cache is True and cfg.guard == "block"
    assert cfg.budget_daily_usd == 5.0 and isinstance(cfg.budget_daily_usd, float)
    assert cfg.upstreams["local"] == {"base": "http://localhost:9", "style": "openai"}


def test_string_booleans_and_numbers_are_coerced(tmp_path: Path) -> None:
    cfg = load_config(str(_cfg(tmp_path, 'cache = "false"\nquiet = "yes"\nport = "4242"\n')))
    assert cfg.cache is False and cfg.quiet is True and cfg.port == 4242


@pytest.mark.parametrize("text,key", [
    ('port = "abc"', "port"),
    ("port = 70000", "port"),
    ('cache = "maybe"', "cache"),
    ("db_path = 5", "db_path"),
    ('guard = "loud"', "guard"),
    ('mode = "rewind"', "mode"),
    ("budget_daily_usd = true", "budget_daily_usd"),
    ('upstreams = "x"', "upstreams"),
    ('[upstreams.x]\nstyle = "openai"', "upstreams.x"),
    ('[upstreams.x]\nbase = "http://h"\nstyle = "soap"', "upstreams.x"),
    ("llmtap = 5", "llmtap"),
    ('pricing = "cheap"', "pricing"),
])
def test_bad_values_raise_clear_value_error(tmp_path: Path, text: str, key: str) -> None:
    p = _cfg(tmp_path, text + "\n")
    with pytest.raises(ValueError) as ei:
        load_config(str(p))
    msg = str(ei.value)
    assert str(p) in msg and key in msg


def test_invalid_toml_names_the_file(tmp_path: Path) -> None:
    p = _cfg(tmp_path, "this is = = not toml\n")
    with pytest.raises(ValueError, match="c.toml"):
        load_config(str(p))


def test_toml_is_read_as_utf8(tmp_path: Path) -> None:
    p = _cfg(tmp_path, '# café ✓ ā\ncassette = "tāpe.jsonl"\n')
    assert load_config(str(p)).cassette == "tāpe.jsonl"
    # also outside UTF-8 mode, where the locale encoding (cp1252 on Windows) would be used
    env = dict(os.environ, PYTHONUTF8="0", PYTHONIOENCODING="utf-8")
    code = "import sys; from llmtap.config import load_config; print(ascii(load_config(sys.argv[1]).cassette))"
    root = str(Path(__file__).resolve().parents[1])
    r = subprocess.run([sys.executable, "-c", code, str(p)], env=dict(env, PYTHONPATH=root),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ascii("tāpe.jsonl")


def test_bad_env_var_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMTAP_PORT", "abc")
    with pytest.raises(ValueError, match="LLMTAP_PORT"):
        load_config()


def test_overrides_and_defaults_untouched() -> None:
    cfg = load_config(None, {"port": 0, "cache": True, "upstreams": {"u": {"base": "http://x/", "style": "anthropic"}}})
    assert cfg.port == 0 and cfg.cache is True and cfg.upstreams["u"]["style"] == "anthropic"
    assert Config().upstreams["openai"]["style"] == "openai"
