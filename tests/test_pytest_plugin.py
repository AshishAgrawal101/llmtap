"""Tests for llmtap.pytest_plugin, run in-process through ``pytester`` against a MockUpstream.

The plugin is loaded explicitly (``-p llmtap.pytest_plugin``) so the tests pass whether or not the
package's ``pytest11`` entry point is installed; ``-p no:llmtap`` blocks the entry-point copy.
"""
from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from llmtap.mock import MockUpstream

SECRET = "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OPENAI_SECRET = "sk-proj-ZYXWVUTSRQPONMLKJIHGFEDCBA9876543210"

HELPERS = '''
import json, os, urllib.error, urllib.request

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SECRET = "%(secret)s"
OPENAI_SECRET = "%(osecret)s"


def post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with _opener.open(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def ask(text, model="claude-sonnet-4-5"):
    """Anthropic-style call through ANTHROPIC_BASE_URL -> (status, reply text or error message)."""
    st, j = post(os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages",
                 {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": text}]},
                 {"x-api-key": os.environ.get("ANTHROPIC_API_KEY") or SECRET, "anthropic-version": "2023-06-01",
                  "content-type": "application/json"})
    return st, (j["content"][0]["text"] if st == 200 else j["error"]["message"])


def ask_openai(text, model="gpt-4o-mini"):
    """OpenAI-style call through OPENAI_BASE_URL."""
    st, j = post(os.environ["OPENAI_BASE_URL"] + "/chat/completions",
                 {"model": model, "messages": [{"role": "user", "content": text}]},
                 {"authorization": "Bearer " + OPENAI_SECRET, "content-type": "application/json"})
    return st, (j["choices"][0]["message"]["content"] if st == 200 else j["error"]["message"])
''' % {"secret": SECRET, "osecret": OPENAI_SECRET}

ENV_VARS = ("CI", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL",
            "OPENAI_API_BASE", "LLMTAP_URL")


class Env:
    """A pytester wired to a mock upstream; ``run`` loads the plugin."""

    def __init__(self, pytester: pytest.Pytester, mock: MockUpstream) -> None:
        self.pt = pytester
        self.mock = mock
        self._point_at(mock.base_url)
        pytester.makepyfile(llmhelpers=HELPERS)
        pytester.makeconftest("import os, sys\nsys.path.insert(0, os.path.dirname(__file__))\n")

    def _point_at(self, base: str) -> None:
        (self.pt.path / "llmtap.toml").write_text(
            f'[upstreams.anthropic]\nbase = "{base}"\nstyle = "anthropic"\n'
            f'[upstreams.openai]\nbase = "{base}"\nstyle = "openai"\n')

    def upstream_down(self) -> None:
        """Repoint the upstreams at a closed port: anything that reaches the network now fails."""
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self._point_at(f"http://127.0.0.1:{port}")

    def run(self, *args: str, **files: str) -> pytest.RunResult:
        for name, src in files.items():
            self.pt.makepyfile(**{name: src})
        return self.pt.runpytest_inprocess("-p", "no:llmtap", "-p", "llmtap.pytest_plugin",
                                           "-p", "no:cacheprovider", *args)

    def cassettes(self, sub: str = "cassettes") -> list[Path]:
        return sorted((self.pt.path / sub).glob("*.jsonl"))


@pytest.fixture(scope="module")
def _shared_mock() -> Any:
    m = MockUpstream().start()
    yield m
    m.stop()


@pytest.fixture()
def env(pytester: pytest.Pytester, _shared_mock: MockUpstream, monkeypatch: pytest.MonkeyPatch) -> Env:
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    _shared_mock.reset()
    return Env(pytester, _shared_mock)


BASIC = '''
from llmhelpers import ask

def test_hello(llmtap):
    assert ask("hi there") == (200, "Echo: hi there")
'''


# ------------------------------------------------------------------------------------ record + replay
def test_record_then_replay_with_upstream_stopped(env: Env) -> None:
    r = env.run(test_mod=BASIC)  # CI unset -> auto: records
    r.assert_outcomes(passed=1)
    cass = env.cassettes()
    assert [c.name for c in cass] == ["test_mod__test_hello.jsonl"]
    assert len(env.mock.requests) == 1

    env.upstream_down()  # nothing to talk to: replay must come entirely from the cassette
    r = env.run("--llmtap-mode", "replay")
    r.assert_outcomes(passed=1)
    assert env.cassettes() == cass


def test_openai_style_roundtrip(env: Env) -> None:
    src = '''
from llmhelpers import ask_openai

def test_chat(llmtap):
    assert ask_openai("ping") == (200, "Echo: ping")
'''
    env.run(test_mod=src).assert_outcomes(passed=1)
    env.upstream_down()
    env.run("--llmtap-mode", "replay").assert_outcomes(passed=1)


def test_ledger_api(env: Env) -> None:
    src = '''
import os
from llmhelpers import ask

def test_ledger(llmtap):
    assert llmtap.cassette_path.name == "test_mod__test_ledger.jsonl"
    assert llmtap.base_url == os.environ["ANTHROPIC_BASE_URL"]
    assert llmtap.calls == 0 and llmtap.cost_usd == 0 and llmtap.exchanges() == []
    ask("one"); ask("two")
    exs = llmtap.exchanges()
    assert [e.output_text for e in exs] == ["Echo: one", "Echo: two"]   # oldest first
    if llmtap.mode == "replay":
        assert llmtap.calls == 2 and llmtap.cost_usd == 0 and llmtap.saved_usd > 0
        assert {e.source for e in exs} == {"cassette"}
    else:
        assert llmtap.calls == 2 and llmtap.cost_usd > 0 and llmtap.saved_usd == 0
        assert {e.source for e in exs} == {"upstream"}
'''
    env.run(test_mod=src).assert_outcomes(passed=1)
    env.upstream_down()
    env.run("--llmtap-mode", "replay").assert_outcomes(passed=1)


def test_record_flag_rerecords_everything(env: Env) -> None:
    env.run(test_mod=BASIC).assert_outcomes(passed=1)
    cassette = env.cassettes()[0]
    cassette.write_text(cassette.read_text() + cassette.read_text())  # make it stale: 2 lines
    n = len(env.mock.requests)
    env.run("--llmtap-record").assert_outcomes(passed=1)
    assert len(env.mock.requests) == n + 1            # record mode never reads the cassette
    assert len(cassette.read_text().splitlines()) == 1  # and starts the file afresh


# ------------------------------------------------------------------------------------ replay misses
MISS = '''
from llmhelpers import ask

def test_miss(llmtap):
    st, msg = ask("%s")
    assert st == 409 and "cassette miss" in msg     # the SDK sees a 409
'''


def test_replay_miss_fails_test_at_teardown_with_clear_message(env: Env) -> None:
    long_prompt = "unrecorded question " + "x" * 200
    r = env.run("--llmtap-mode", "replay", test_mod=MISS % long_prompt)
    r.assert_outcomes(passed=1, errors=1)  # the test body passed; teardown failed it
    out = r.stdout.str()
    assert "had no recording" in out
    assert "model=claude-sonnet-4-5" in out
    assert "unrecorded question " + "x" * 60 in out
    assert "x" * 100 not in out                    # only the first 80 chars
    assert "re-run with --llmtap-record" in out
    assert "test_mod__test_miss.jsonl" in out
    assert not env.mock.requests                   # a miss never reaches the upstream


def test_replay_miss_does_not_mask_the_tests_own_failure(env: Env) -> None:
    src = '''
from llmhelpers import ask

def test_boom(llmtap):
    ask("never recorded")
    assert False, "the original assertion"
'''
    r = env.run("--llmtap-mode", "replay", test_mod=src)
    r.assert_outcomes(failed=1)  # no extra teardown error
    out = r.stdout.str()
    assert "the original assertion" in out
    assert "had no recording" in out  # still surfaced, in the llmtap summary section
    assert "re-run with --llmtap-record" in out


def test_allow_miss_marker_forwards_instead_of_failing(env: Env) -> None:
    src = '''
import pytest
from llmhelpers import ask

@pytest.mark.llmtap(allow_miss=True)
def test_forward():
    assert ask("fresh") == (200, "Echo: fresh")
'''
    env.run("--llmtap-mode", "replay", test_mod=src).assert_outcomes(passed=1)
    assert len(env.mock.requests) == 1


# ------------------------------------------------------------------------------------ marker
def test_marker_alone_requests_fixture_and_sets_env(env: Env) -> None:
    src = '''
import os, pytest
from llmhelpers import ask

@pytest.mark.llmtap
def test_only_marker():
    assert os.environ["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")
    assert ask("m") == (200, "Echo: m")

@pytest.mark.llmtap
class TestGroup:
    def test_method(self):
        assert ask("k") == (200, "Echo: k")
'''
    env.run(test_mod=src).assert_outcomes(passed=2)
    assert [c.name for c in env.cassettes()] == ["test_mod__TestGroup.test_method.jsonl",
                                                  "test_mod__test_only_marker.jsonl"]


def test_marker_cassette_name_is_shared_between_tests(env: Env) -> None:
    src = '''
import pytest
from llmhelpers import ask

@pytest.mark.llmtap(cassette="shared")
def test_a(llmtap):
    assert llmtap.cassette_path.name == "shared.jsonl"
    ask("a")

@pytest.mark.llmtap(cassette="shared")
def test_b(llmtap):
    assert llmtap.cassette_path.name == "shared.jsonl"
    ask("b")
'''
    env.run(test_mod=src).assert_outcomes(passed=2)
    assert [c.name for c in env.cassettes()] == ["shared.jsonl"]
    assert len(env.cassettes()[0].read_text().splitlines()) == 2
    env.upstream_down()
    env.run("--llmtap-mode", "replay").assert_outcomes(passed=2)


def test_marker_mode_overrides_command_line(env: Env) -> None:
    src = '''
import pytest
from llmhelpers import ask

@pytest.mark.llmtap(mode="record")
def test_rec(llmtap):
    assert llmtap.mode == "record"
    assert ask("r") == (200, "Echo: r")

def test_plain(llmtap):
    assert llmtap.mode == "replay"
'''
    r = env.run("--llmtap-mode", "replay", test_mod=src)
    r.assert_outcomes(passed=2)
    assert len(env.mock.requests) == 1


def test_marker_guard_redacts_secrets_before_upstream(env: Env) -> None:
    src = '''
import pytest
from llmhelpers import ask, SECRET

@pytest.mark.llmtap(guard="redact")
def test_guard():
    st, text = ask("my key is " + SECRET)
    assert st == 200
'''
    env.run(test_mod=src).assert_outcomes(passed=1)
    assert SECRET not in json.dumps(env.mock.requests[0]["body"])
    assert "[REDACTED" in json.dumps(env.mock.requests[0]["body"])


@pytest.mark.parametrize("marker", ['mode="sometimes"', 'guard="yes"', "bogus=1"])
def test_bad_marker_arguments_are_rejected(env: Env, marker: str) -> None:
    src = f'''
import pytest

@pytest.mark.llmtap({marker})
def test_x():
    pass
'''
    r = env.run(test_mod=src)
    assert r.ret != 0
    assert "llmtap" in r.stdout.str() + r.stderr.str()


# ------------------------------------------------------------------------------------ env handling
def test_env_vars_set_during_test_and_restored(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://original.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    src = '''
import os

def test_a_inside(llmtap):
    base = llmtap.base_url
    assert os.environ["ANTHROPIC_BASE_URL"] == base
    assert os.environ["OPENAI_BASE_URL"] == base + "/v1"
    assert os.environ["OPENAI_API_BASE"] == base + "/v1"
    assert os.environ["LLMTAP_URL"] == base
    assert llmtap.mode == "replay"
    assert os.environ["ANTHROPIC_API_KEY"]                 # fake key so SDKs can construct
    assert os.environ["OPENAI_API_KEY"] == "real-openai-key"  # real keys are never overwritten

def test_b_outside():
    assert os.environ["OPENAI_BASE_URL"] == "http://original.invalid/v1"
    for v in ("ANTHROPIC_BASE_URL", "OPENAI_API_BASE", "LLMTAP_URL", "ANTHROPIC_API_KEY"):
        assert v not in os.environ
'''
    env.run("--llmtap-mode", "replay", test_mod=src).assert_outcomes(passed=2)
    import os
    assert os.environ["OPENAI_BASE_URL"] == "http://original.invalid/v1"
    assert "LLMTAP_URL" not in os.environ


def test_no_fake_keys_in_record_mode(env: Env) -> None:
    src = '''
import os

def test_rec(llmtap):
    assert "ANTHROPIC_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ
'''
    env.run("--llmtap-mode", "record", test_mod=src).assert_outcomes(passed=1)


def test_record_mode_uses_default_upstreams_not_the_mock(env: Env) -> None:
    """Without llmtap.toml overrides the real hosts are the defaults (checked, never contacted)."""
    (env.pt.path / "llmtap.toml").unlink()
    src = '''
def test_defaults(llmtap):
    ups = llmtap.tap.cfg.upstreams
    assert ups["anthropic"]["base"] == "https://api.anthropic.com"
    assert ups["openai"]["base"] == "https://api.openai.com"
    assert llmtap.tap.cfg.cache is False and llmtap.tap.cfg.db_path == ":memory:"
'''
    env.run("--llmtap-record", test_mod=src).assert_outcomes(passed=1)


# ------------------------------------------------------------------------------------ summary
def test_terminal_summary_line(env: Env) -> None:
    env.run(test_mod=BASIC).assert_outcomes(passed=1)
    r = env.run()  # second run: auto mode, now a cassette hit
    r.assert_outcomes(passed=1)
    r.stdout.re_match_lines([r"llmtap: 1 call, \$\d+\.\d+ spent, \$\d+\.\d+ saved by replay, 1 cassette$"])
    out = r.stdout.str()
    assert "$0.00 spent" in out or "$0.0000 spent" in out
    assert not re.search(r"\$0\.00 saved|\$0\.0000 saved", out)  # replay saved something


def test_terminal_summary_first_run_spends(env: Env) -> None:
    r = env.run(test_mod=BASIC)
    assert re.search(r"llmtap: 1 call, \$0\.0\d+ spent, \$0\.00(00)? saved by replay, 1 cassette", r.stdout.str())


# ------------------------------------------------------------------------------------ inert
def test_inert_when_unused(env: Env) -> None:
    src = '''
import os, threading

def test_nothing():
    before = threading.active_count()
    for v in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "LLMTAP_URL"):
        assert v not in os.environ
    assert not [t for t in threading.enumerate() if t.name == "llmtap-proxy"]
'''
    r = env.run(test_mod=src)
    r.assert_outcomes(passed=1)
    assert "llmtap:" not in r.stdout.str()
    assert not (env.pt.path / "cassettes").exists()


def test_plugin_import_is_cheap_and_does_not_load_the_proxy() -> None:
    code = ("import sys, llmtap.pytest_plugin;"
            "assert 'llmtap.proxy' not in sys.modules, 'proxy imported';"
            "assert 'llmtap.store' not in sys.modules, 'store imported'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                       cwd=str(Path(__file__).resolve().parent.parent))
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------------------------ mode defaults
def test_default_mode_depends_on_ci_env(env: Env) -> None:
    src = '''
import pytest

@pytest.mark.parametrize("ci, expected", [(None, "auto"), ("0", "auto"), ("false", "auto"), ("", "auto"),
                                          ("true", "replay"), ("1", "replay"), ("yes", "replay")])
def test_mode(request, monkeypatch, ci, expected):
    monkeypatch.delenv("CI", raising=False)
    if ci is not None:
        monkeypatch.setenv("CI", ci)
    assert request.getfixturevalue("llmtap").mode == expected
'''
    env.run(test_mod=src).assert_outcomes(passed=7)


@pytest.mark.parametrize("args, expected", [
    (("--llmtap-mode", "auto"), "auto"),
    (("--llmtap-record",), "record"),
    (("--llmtap-mode", "replay", "--llmtap-record"), "record"),  # the shortcut wins
])
def test_command_line_beats_ci_default(env: Env, monkeypatch: pytest.MonkeyPatch, args: tuple[str, ...],
                                       expected: str) -> None:
    monkeypatch.setenv("CI", "1")
    src = f'''
def test_mode(llmtap):
    assert llmtap.mode == "{expected}", llmtap.mode
'''
    env.run(*args, test_mod=src).assert_outcomes(passed=1)


def test_ci_replay_fails_on_missing_cassette(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "true")
    r = env.run(test_mod=MISS % "brand new prompt")
    r.assert_outcomes(passed=1, errors=1)
    assert "cassette file does not exist" in r.stdout.str()
    assert not env.mock.requests


# ------------------------------------------------------------------------------------ naming + dirs
def test_parametrized_cassette_names_are_distinct_and_sanitized(env: Env) -> None:
    src = '''
import pytest
from llmhelpers import ask

@pytest.mark.parametrize("p", ["a/b", "a b", "a_b", "plain", "../../evil", "x" * 300, "ünï"])
def test_p(llmtap, p):
    ask("p")

class TestK:
    def test_m(self, llmtap):
        ask("m")
'''
    env.run(test_mod=src).assert_outcomes(passed=8)
    names = [c.name for c in env.cassettes()]
    assert len(names) == len(set(names)) == 8
    for n in names:
        assert re.fullmatch(r"[A-Za-z0-9_.\\-]+\.jsonl", n), n
        assert len(n) < 200 and not n.startswith(".")
    assert any(n.startswith("test_mod__test_p__plain") for n in names)
    assert "test_mod__TestK.test_m.jsonl" in names
    assert not list(env.pt.path.glob("*evil*")) and not (env.pt.path.parent / "evil").exists()


def test_cassette_dir_option_and_ini(env: Env) -> None:
    env.run("--llmtap-cassette-dir", "recorded", test_mod=BASIC).assert_outcomes(passed=1)
    assert [c.name for c in env.cassettes("recorded")] == ["test_mod__test_hello.jsonl"]
    assert not (env.pt.path / "cassettes").exists()

    env.pt.makeini("[pytest]\nllmtap_cassette_dir = from_ini\n")
    env.run("--llmtap-mode", "replay").assert_outcomes(failed=1)  # nothing recorded at from_ini yet
    env.run().assert_outcomes(passed=1)
    assert [c.name for c in env.cassettes("from_ini")] == ["test_mod__test_hello.jsonl"]


def test_cassette_next_to_test_file_in_subdir(env: Env) -> None:
    env.pt.mkdir("sub")
    (env.pt.path / "sub" / "test_in_sub.py").write_text(BASIC)
    env.run().assert_outcomes(passed=1)
    assert (env.pt.path / "sub" / "cassettes" / "test_in_sub__test_hello.jsonl").is_file()


# ------------------------------------------------------------------------------------ secrets
def test_no_api_key_ends_up_in_cassette(env: Env) -> None:
    src = '''
import os
from llmhelpers import ask, ask_openai, SECRET, OPENAI_SECRET

def test_keys(llmtap):
    ask("hello")           # sends x-api-key: <SECRET> upstream
    ask_openai("hello")    # sends Authorization: Bearer <OPENAI_SECRET>
'''
    env.run(test_mod=src).assert_outcomes(passed=1)
    # the mock really did receive the keys...
    sent = json.dumps([r["headers"] for r in env.mock.requests])
    assert SECRET in sent and OPENAI_SECRET in sent
    # ...but the cassette contains neither the keys nor any request header.
    text = env.cassettes()[0].read_text()
    assert SECRET not in text and OPENAI_SECRET not in text
    for line in text.splitlines():
        d: dict[str, Any] = json.loads(line)
        assert "headers" not in d["request"] if isinstance(d["request"], dict) else True
        assert not any(k.lower() in ("x-api-key", "authorization") for k in d["response"]["headers"])
    assert "x-api-key" not in text.lower() and "bearer" not in text.lower()
