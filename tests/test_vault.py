"""Key vault: storage, the placeholder swap through the proxy, per-key limits, the CLI and the dashboard API."""
from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from llmtap import dashboard
from llmtap.vault import (FileBackend, KeyInfo, MemoryBackend, Vault, VaultError, check_limits, find_placeholder,
                          mask, substitute)

# Fake keys (built by concatenation so secret scanners don't flag the test file)
OPENAI_KEY = "sk-" + "test" + "0123456789abcdefOPENAI"
ANTHROPIC_KEY = "sk-ant-" + "test" + "0123456789abcdefANTH"
GEMINI_KEY = "AIza" + "Test" + "0123456789abcdefGEMINI00"


@pytest.fixture()
def vault(tmp_path: Path) -> Vault:
    return Vault(home=tmp_path / "home", backend=MemoryBackend())


@pytest.fixture()
def vtap(tap: Any, vault: Vault, mock_upstream: Any) -> Any:
    """A proxy (anthropic, openai and gemini upstreams on the mock) using a throwaway vault."""
    def make(**overrides: Any) -> Any:
        ups = {"gemini": {"base": mock_upstream.base_url, "style": "gemini"}}
        t = tap(upstreams=ups, **overrides)
        t._vault = vault
        return t
    return make


def _call(t: Any, path: str, body: dict | None, headers: dict[str, str], method: str = "POST") -> tuple[int, dict]:
    c = http.client.HTTPConnection("127.0.0.1", t.port, timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    c.request(method, path, body=data, headers={"Content-Type": "application/json", **headers})
    r = c.getresponse()
    raw = r.read()
    c.close()
    try:
        return r.status, json.loads(raw or b"{}")
    except ValueError:
        return r.status, {"raw": raw.decode("utf-8", "replace")}


def _openai(t: Any, key: str = "llmtap-key-openai", model: str = "gpt-4o-mini") -> tuple[int, dict]:
    return _call(t, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}]},
                 {"Authorization": f"Bearer {key}"})


def _anthropic(t: Any, key: str = "llmtap-key-anthropic", model: str = "claude-sonnet-4-5") -> tuple[int, dict]:
    return _call(t, "/v1/messages", {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
                 {"x-api-key": key, "anthropic-version": "2023-06-01"})


def _wait_rows(t: Any, n: int) -> list[Any]:
    deadline = time.time() + 5
    rows = t.store.list(limit=100)
    while len(rows) < n and time.time() < deadline:
        time.sleep(0.02)
        rows = t.store.list(limit=100)
    return rows


def _everything_stored(t: Any) -> str:
    return json.dumps([r.to_dict() for r in t.store.list(limit=1000)])


# ------------------------------------------------------------------------------------------------ storage

def test_add_list_update_remove_and_no_secret_in_metadata(vault: Vault) -> None:
    info = vault.add("openai", "openai", OPENAI_KEY)
    assert info.placeholder == "llmtap-key-openai" and info.preview == mask(OPENAI_KEY)
    assert OPENAI_KEY not in vault.meta_path.read_text(encoding="utf-8")
    assert [k.name for k in vault.list()] == ["openai"]
    assert vault.secret("openai") == OPENAI_KEY
    vault.update("openai", daily_usd=2, models="gpt-4o*, gpt-5-mini", paused=True, expires=0)
    k = vault.get("openai")
    assert k.daily_usd == 2 and k.models == ["gpt-4o*", "gpt-5-mini"] and k.paused
    vault.add("openai", "openai", OPENAI_KEY + "X")          # replacing the secret keeps the limits
    k = vault.get("openai")
    assert k.rev == 2 and k.daily_usd == 2 and vault.secret("openai") == OPENAI_KEY + "X"
    assert vault.remove("openai") is True and vault.list() == [] and vault.remove("openai") is False


def test_other_processes_see_changes(tmp_path: Path) -> None:
    be = MemoryBackend()
    a, b = Vault(home=tmp_path, backend=be), Vault(home=tmp_path, backend=be)
    a.add("openai", "openai", OPENAI_KEY)
    assert b.secret("openai") == OPENAI_KEY
    time.sleep(0.02)
    a.add("openai", "openai", OPENAI_KEY + "2")               # new rev: b drops its cached secret
    os.utime(a.meta_path, (time.time() + 5, time.time() + 5))
    assert b.secret("openai") == OPENAI_KEY + "2"


@pytest.mark.parametrize("name", ["", "Has Space", "-dash", "x" * 41, "a/b"])
def test_bad_names_rejected(vault: Vault, name: str) -> None:
    with pytest.raises(VaultError):
        vault.add(name, "openai", OPENAI_KEY)


@pytest.mark.parametrize("secret", ["", "has space", 'quo"te', "llmtap-key-openai", "x" * 2001])
def test_bad_secrets_rejected(vault: Vault, secret: str) -> None:
    with pytest.raises(VaultError) as e:
        vault.add("openai", "openai", secret)
    assert OPENAI_KEY not in str(e.value)


@pytest.mark.parametrize("limits", [{"daily_usd": -1}, {"total_usd": "lots"}, {"paused": "yes"}, {"bogus": 1},
                                    {"expires": -5}, {"models": 7}])
def test_bad_limits_rejected(vault: Vault, limits: dict) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    with pytest.raises(VaultError):
        vault.update("openai", **limits)


def test_file_backend_is_private(tmp_path: Path) -> None:
    be = FileBackend(tmp_path / "s.json")
    be.set("openai", OPENAI_KEY)
    assert be.get("openai") == OPENAI_KEY
    if os.name == "posix":
        assert (tmp_path / "s.json").stat().st_mode & 0o077 == 0
    be.delete("openai")
    assert be.get("openai") is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Credential Manager")
def test_windows_credential_manager_roundtrip() -> None:
    from llmtap.vault import WindowsBackend
    be = WindowsBackend()
    name = f"pytest-{os.getpid()}"
    try:
        be.set(name, OPENAI_KEY)
        assert be.get(name) == OPENAI_KEY
    finally:
        be.delete(name)
    assert be.get(name) is None


def test_mask_never_reveals_much() -> None:
    assert mask(OPENAI_KEY) == OPENAI_KEY[:4] + "…" + OPENAI_KEY[-4:]
    assert mask("short123") == "…23"
    assert mask("abc") == "…"


def test_child_env_prefers_key_named_like_provider(vault: Vault) -> None:
    vault.add("work", "openai", OPENAI_KEY)
    vault.add("openai", "openai", OPENAI_KEY + "b")
    vault.add("gem", "gemini", GEMINI_KEY)
    vault.add("mine", "my-gateway", OPENAI_KEY + "c")          # custom upstreams get no env var
    env = vault.child_env({"ANTHROPIC_API_KEY": "real", "OPENAI_API_KEY": OPENAI_KEY})
    assert env == {"OPENAI_API_KEY": "llmtap-key-openai", "GEMINI_API_KEY": "llmtap-key-gem",
                   "GOOGLE_API_KEY": "llmtap-key-gem"}
    assert vault.child_env({"OPENAI_API_KEY": "llmtap-key-work"}).get("OPENAI_API_KEY") is None


def test_find_and_substitute() -> None:
    f = find_placeholder({"authorization": "Bearer llmtap-key-openai"}, "/v1/chat/completions")
    assert f and f.name == "openai" and f.where == "authorization"
    hdrs, path = substitute({"authorization": "Bearer llmtap-key-openai"}, "/p", f, OPENAI_KEY)
    assert hdrs["authorization"] == "Bearer " + OPENAI_KEY and path == "/p"
    f = find_placeholder({}, "/v1beta/models/g:generateContent?alt=sse&key=llmtap-key-gem")
    assert f and f.where == "?key"
    _, path = substitute({}, "/v1beta/models/g:generateContent?alt=sse&key=llmtap-key-gem", f, GEMINI_KEY)
    assert path == f"/v1beta/models/g:generateContent?alt=sse&key={GEMINI_KEY}"
    assert find_placeholder({"authorization": "Bearer llmtap-key-openaiX!"}, "/") is not None
    assert find_placeholder({"authorization": "Bearer " + OPENAI_KEY}, "/?key=abc") is None
    assert find_placeholder({"x-other": "llmtap-key-openai"}, "/") is None


def test_check_limits_rules() -> None:
    k = KeyInfo(name="k", provider="openai", daily_usd=1.0, total_usd=5.0, models=["gpt-4o*"])
    assert check_limits(k, "gpt-4o-mini", 0.5, 1.0, 0.0, priced=True) is None
    assert check_limits(k, "gpt-5", 0, 0, 0, priced=True).err_type == "llmtap_key_model_not_allowed"
    assert check_limits(k, None, 0, 0, 0, priced=False) is None            # GET /v1/models etc.
    assert check_limits(k, "gpt-4o", 0.6, 1.0, 0.4, priced=True).status == 429   # spent + in flight
    assert check_limits(k, "gpt-4o", 0.1, 5.0, 0.0, priced=True).status == 429
    k.paused = True
    assert check_limits(k, "gpt-4o", 0, 0, 0, priced=False).err_type == "llmtap_key_paused"
    k.paused, k.expires = False, time.time() - 1
    assert check_limits(k, "gpt-4o", 0, 0, 0, priced=False).err_type == "llmtap_key_expired"


# ------------------------------------------------------------------------------------------- through the proxy

def test_openai_placeholder_is_swapped_and_never_stored(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    t = vtap()
    status, _ = _openai(t)
    assert status == 200
    assert mock_upstream.requests[-1]["headers"]["authorization"] == "Bearer " + OPENAI_KEY
    rows = _wait_rows(t, 1)
    assert rows[0].tags.get("vault_key") == "openai"
    assert OPENAI_KEY not in _everything_stored(t)


def test_anthropic_x_api_key_swapped(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("anthropic", "anthropic", ANTHROPIC_KEY)
    t = vtap()
    assert _anthropic(t)[0] == 200
    assert mock_upstream.requests[-1]["headers"]["x-api-key"] == ANTHROPIC_KEY
    assert ANTHROPIC_KEY not in _everything_stored(t)


def test_gemini_query_key_swapped_and_masked(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("gem", "gemini", GEMINI_KEY)
    t = vtap()
    _call(t, "/v1beta/models/gemini-2.5-flash:generateContent?key=llmtap-key-gem",
          {"contents": [{"parts": [{"text": "hi"}]}]}, {})
    assert mock_upstream.requests[-1]["path"].endswith("key=" + GEMINI_KEY)
    _wait_rows(t, 1)
    assert GEMINI_KEY not in _everything_stored(t)


def test_unknown_key_is_refused_with_provider_shaped_error(vtap: Any, mock_upstream: Any) -> None:
    t = vtap()
    status, body = _openai(t, key="llmtap-key-nope")
    assert status == 401 and "no stored API key named 'nope'" in json.dumps(body)
    assert not mock_upstream.requests


def test_key_only_goes_to_its_own_provider(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    t = vtap()
    status, body = _anthropic(t, key="llmtap-key-openai")
    assert status == 403 and "only ever sent to their own provider" in json.dumps(body)
    assert not mock_upstream.requests


def test_real_keys_still_work_unchanged(vtap: Any, mock_upstream: Any) -> None:
    t = vtap()
    assert _openai(t, key=OPENAI_KEY)[0] == 200
    assert mock_upstream.requests[-1]["headers"]["authorization"] == "Bearer " + OPENAI_KEY
    assert t._vault.list() == []


def test_client_cannot_fake_the_attribution_tag(vtap: Any, vault: Vault) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    t = vtap()
    _call(t, "/v1/chat/completions", {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]},
          {"Authorization": "Bearer " + OPENAI_KEY, "X-LLMTap-Tag-vault_key": "openai"})
    assert "vault_key" not in _wait_rows(t, 1)[0].tags


def test_paused_expired_and_model_limits(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("openai", "openai", OPENAI_KEY, paused=True)
    t = vtap()
    assert _openai(t)[0] == 403
    vault.update("openai", paused=False, expires=time.time() - 60)
    assert _openai(t)[0] == 403
    vault.update("openai", expires=0, models=["gpt-4o*"])
    assert _openai(t, model="gpt-5")[0] == 403
    assert _openai(t, model="gpt-4o-mini")[0] == 200
    assert len(mock_upstream.requests) == 1


def test_daily_cap_counts_spend_per_key(vtap: Any, vault: Vault) -> None:
    vault.add("anthropic", "anthropic", ANTHROPIC_KEY, daily_usd=0.000001)
    t = vtap()
    assert _anthropic(t)[0] == 200                   # nothing spent yet: allowed
    status, body = _anthropic(t)                     # the first call cost more than the cap
    assert status == 429 and "daily limit" in json.dumps(body)
    rows = _wait_rows(t, 2)
    assert {r.source for r in rows} == {"upstream", "budget"}
    # another key / real keys are not affected by this key's cap
    assert _anthropic(t, key=ANTHROPIC_KEY)[0] == 200


def test_cache_hits_do_not_need_the_key(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    t = vtap(cache=True)
    assert _openai(t)[0] == 200
    vault.update("openai", paused=True)
    assert _openai(t)[0] == 200                      # served from the cache: free and never reaches the provider
    assert len(mock_upstream.requests) == 1


# ------------------------------------------------------------------------------------------------- CLI + wrap

def _cli(args: list[str], home: Path, stdin: str = "") -> subprocess.CompletedProcess:
    env = dict(os.environ, LLMTAP_HOME=str(home), LLMTAP_VAULT_BACKEND="file")
    return subprocess.run([sys.executable, "-m", "llmtap", *args], input=stdin, capture_output=True, text=True,
                          env=env, timeout=60)


def test_cli_add_list_limit_remove(tmp_path: Path) -> None:
    r = _cli(["keys", "add", "openai"], tmp_path, stdin=OPENAI_KEY + "\n")
    assert r.returncode == 0, r.stderr
    assert OPENAI_KEY not in r.stdout + r.stderr
    r = _cli(["keys", "limit", "openai", "--daily", "3", "--models", "gpt-4o*", "--expires", "2030-01-02", "--pause"],
             tmp_path)
    assert r.returncode == 0, r.stderr
    r = _cli(["keys"], tmp_path)
    assert "llmtap-key-openai" in r.stdout and "$3/day" in r.stdout and "PAUSED" in r.stdout
    assert OPENAI_KEY not in r.stdout and OPENAI_KEY not in (tmp_path / "keys.json").read_text()
    assert _cli(["keys", "add", "work"], tmp_path, stdin=OPENAI_KEY).returncode == 2      # provider unknown
    assert _cli(["keys", "remove", "openai"], tmp_path).returncode == 0
    assert "no stored keys" in _cli(["keys", "list"], tmp_path).stdout


def test_wrap_gives_the_child_only_placeholders(tmp_path: Path) -> None:
    assert _cli(["keys", "add", "openai"], tmp_path, stdin=OPENAI_KEY).returncode == 0
    code = "import os; print('KEY=' + os.environ.get('OPENAI_API_KEY', ''))"
    r = _cli(["wrap", "--quiet", "--db", str(tmp_path / "db.sqlite"), "--", sys.executable, "-c", code], tmp_path)
    assert "KEY=llmtap-key-openai" in r.stdout
    assert OPENAI_KEY not in r.stdout + r.stderr


# ------------------------------------------------------------------------------------------------ dashboard API

class _R:
    def __init__(self) -> None:
        self.status, self.headers, self.body = 0, {}, b""

    def start(self, status: int, headers: dict) -> None:
        self.status, self.headers = status, headers

    def write(self, chunk: bytes) -> None:
        self.body += chunk

    def end(self) -> None:
        pass

    def json(self) -> Any:
        return json.loads(self.body or b"{}")


def _api(t: Any, method: str, path: str, body: Any = None, headers: dict | None = None) -> _R:
    r = _R()
    h = {"host": f"127.0.0.1:{t.port}", **(headers or {})}
    dashboard.handle_internal(t, method, path, h, json.dumps(body).encode() if body is not None else b"", r)
    return r


GOOD = {"x-llmtap-request": "dashboard", "content-type": "application/json"}


def test_dashboard_add_list_update_delete(vtap: Any) -> None:
    t = vtap()
    r = _api(t, "POST", "/_llmtap/api/keys", {"name": "openai", "provider": "openai", "secret": OPENAI_KEY,
                                               "daily_usd": 1}, GOOD)
    assert r.status == 200 and OPENAI_KEY not in r.body.decode()
    lst = _api(t, "GET", "/_llmtap/api/keys")
    assert lst.status == 200 and OPENAI_KEY not in lst.body.decode()
    k = lst.json()["keys"][0]
    assert k["name"] == "openai" and k["daily_usd"] == 1 and k["env"] == ["OPENAI_API_KEY"]
    assert k["spent_today_usd"] == 0 and "openai" in lst.json()["providers"]
    assert _api(t, "POST", "/_llmtap/api/keys/openai", {"paused": True, "models": "gpt-4o"}, GOOD).status == 200
    assert t.vault.get("openai").paused and t.vault.get("openai").models == ["gpt-4o"]
    assert _api(t, "POST", "/_llmtap/api/keys/openai", {"daily_usd": -3}, GOOD).status == 400
    assert _api(t, "POST", "/_llmtap/api/keys/openai/delete", {}, GOOD).status == 200
    assert _api(t, "POST", "/_llmtap/api/keys/openai/delete", {}, GOOD).status == 404


@pytest.mark.parametrize("headers", [
    {"content-type": "application/json"},                                         # no dashboard header
    {**GOOD, "origin": "https://evil.example"},                                   # another site
    {**GOOD, "sec-fetch-site": "cross-site"},
])
def test_dashboard_key_changes_rejected_from_elsewhere(vtap: Any, headers: dict) -> None:
    t = vtap()
    r = _api(t, "POST", "/_llmtap/api/keys", {"name": "openai", "provider": "openai", "secret": OPENAI_KEY}, headers)
    assert r.status == 403 and t.vault.list() == []


def test_dashboard_rejects_rebinding_host(vtap: Any) -> None:
    t = vtap()
    r = _api(t, "GET", "/_llmtap/api/keys", headers={"host": "attacker.example"})
    assert r.status == 403


def test_dashboard_post_body_with_secret_is_not_recorded(vtap: Any) -> None:
    t = vtap()
    status, _ = _call(t, "/_llmtap/api/keys", {"name": "openai", "provider": "openai", "secret": OPENAI_KEY},
                      {"x-llmtap-request": "dashboard"})
    assert status == 200 and t.vault.get("openai") is not None
    assert t.store.list(limit=10) == []
