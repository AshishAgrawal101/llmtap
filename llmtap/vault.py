"""Key vault: store provider API keys once and let apps (or AI coding agents) use them without ever seeing them.

You save a key once (dashboard "Keys" panel or ``llmtap keys add NAME``). The secret goes into the operating
system's credential store; llmtap keeps only non-secret metadata (provider, a masked preview, limits) in
``~/.llmtap/keys.json``.

Code then uses a *placeholder* instead of the key, e.g. ``OPENAI_API_KEY=llmtap-key-openai`` (``llmtap wrap`` sets
these for you). When a request carrying a placeholder passes through llmtap, the real key is swapped in on the way
out, only towards that key's own provider, and only if the key's limits (paused, expiry, allowed models, daily /
total spend) allow it. The real key is never returned by any API, never written to the history database, and
scrubbed from anything an upstream echoes back.

What this protects against: keys pasted into prompts, printed in logs, committed in ``.env`` files, or read by an
agent that dumps its environment. What it does not: a deliberately malicious program running as your own user can
read your OS credential store, or change the limits through the local API, just as you can.

Secret backends (``LLMTAP_VAULT_BACKEND``): ``auto`` (default) uses Windows Credential Manager, the macOS Keychain
(``security``) or the Linux Secret Service (``secret-tool``), and falls back to ``file``: ``~/.llmtap/keys.secret.json``
readable only by you (like ``~/.aws/credentials``; not encrypted). ``memory`` keeps secrets in this process only.
"""
from __future__ import annotations

import fnmatch
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import parse_qsl, quote, urlsplit

from .config import DEFAULT_UPSTREAMS, llmtap_home

SERVICE = "llmtap"
PLACEHOLDER_PREFIX = "llmtap-key-"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
PLACEHOLDER_RE = re.compile(re.escape(PLACEHOLDER_PREFIX) + r"([a-z0-9][a-z0-9_-]{0,39})(?![a-z0-9_-])")
MAX_SECRET_LEN = 2000
# Credential headers a placeholder may travel in (lowercase), plus the ``?key=`` query parameter (Gemini).
PLACEHOLDER_HEADERS = ("authorization", "x-api-key", "x-goog-api-key", "api-key", "openai-api-key")
PLACEHOLDER_QUERY = ("key", "api_key", "apikey")
TAG = "vault_key"          # Exchange tag naming the stored key a request used (clients cannot set it)

# Environment variables ``llmtap wrap`` fills with a placeholder for each provider that has a stored key.
PROVIDER_ENV: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
}


class VaultError(ValueError):
    """Invalid vault input or a failing secret backend; the message is safe to show (never contains a secret)."""


def _atomic_write(path: Path, text: str, private: bool = False) -> None:
    """Write via a fresh, exclusively created temp file in the same directory, then rename over ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".")   # mode 0600, O_EXCL
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        if not private:
            try:
                os.chmod(tmp, 0o644)
            except OSError:
                pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ----------------------------------------------------------------------------------------------------- backends

class SecretBackend(Protocol):
    name: str

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, secret: str) -> None: ...
    def delete(self, key: str) -> None: ...


class MemoryBackend:
    """Secrets in this process only (tests, throwaway sessions)."""
    name = "memory"

    def __init__(self) -> None:
        self._d: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._d.get(key)

    def set(self, key: str, secret: str) -> None:
        with self._lock:
            self._d[key] = secret

    def delete(self, key: str) -> None:
        with self._lock:
            self._d.pop(key, None)


class FileBackend:
    """A JSON file readable only by the current user (mode 0600 on POSIX). Not encrypted."""
    name = "file"

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            raise VaultError(f"cannot read {self.path}: {e}") from None
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _write(self, data: dict[str, str]) -> None:
        _atomic_write(self.path, json.dumps(data), private=True)

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._read().get(key)

    def set(self, key: str, secret: str) -> None:
        with self._lock:
            data = self._read()
            data[key] = secret
            self._write(data)

    def delete(self, key: str) -> None:
        with self._lock:
            data = self._read()
            if data.pop(key, None) is not None:
                self._write(data)


class WindowsBackend:
    """Windows Credential Manager (generic credentials named ``llmtap:<name>``), through advapi32."""
    name = "windows-credential-manager"
    _CRED_TYPE_GENERIC = 1
    _CRED_PERSIST_LOCAL_MACHINE = 2
    _ERROR_NOT_FOUND = 1168

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

        class CREDENTIALW(ctypes.Structure):
            _fields_ = [("Flags", wintypes.DWORD), ("Type", wintypes.DWORD), ("TargetName", wintypes.LPWSTR),
                        ("Comment", wintypes.LPWSTR), ("LastWritten", FILETIME),
                        ("CredentialBlobSize", wintypes.DWORD), ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
                        ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD),
                        ("Attributes", ctypes.c_void_p), ("TargetAlias", wintypes.LPWSTR),
                        ("UserName", wintypes.LPWSTR)]

        self._ct, self._CRED = ctypes, CREDENTIALW
        adv = ctypes.WinDLL("advapi32", use_last_error=True)
        self._write = adv.CredWriteW
        self._write.argtypes = [ctypes.POINTER(CREDENTIALW), wintypes.DWORD]
        self._write.restype = wintypes.BOOL
        self._read = adv.CredReadW
        self._read.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                               ctypes.POINTER(ctypes.POINTER(CREDENTIALW))]
        self._read.restype = wintypes.BOOL
        self._delete = adv.CredDeleteW
        self._delete.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        self._delete.restype = wintypes.BOOL
        self._free = adv.CredFree
        self._free.argtypes = [ctypes.c_void_p]
        self._free.restype = None

    @staticmethod
    def _target(key: str) -> str:
        return f"{SERVICE}:{key}"

    def get(self, key: str) -> str | None:
        ct = self._ct
        p = ct.POINTER(self._CRED)()
        if not self._read(self._target(key), self._CRED_TYPE_GENERIC, 0, ct.byref(p)):
            err = ct.get_last_error()
            if err == self._ERROR_NOT_FOUND:
                return None
            raise VaultError(f"Windows Credential Manager read failed (error {err})")
        try:
            cred = p.contents
            raw = ct.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
            return raw.decode("utf-8")
        finally:
            self._free(p)

    def set(self, key: str, secret: str) -> None:
        ct = self._ct
        blob = secret.encode("utf-8")
        buf = (ct.c_ubyte * len(blob)).from_buffer_copy(blob)
        cred = self._CRED()
        cred.Type = self._CRED_TYPE_GENERIC
        cred.TargetName = self._target(key)
        cred.Comment = "llmtap API key vault"
        cred.CredentialBlobSize = len(blob)
        cred.CredentialBlob = ct.cast(buf, ct.POINTER(ct.c_ubyte))
        cred.Persist = self._CRED_PERSIST_LOCAL_MACHINE
        cred.UserName = SERVICE
        if not self._write(ct.byref(cred), 0):
            raise VaultError(f"Windows Credential Manager write failed (error {ct.get_last_error()})")

    def delete(self, key: str) -> None:
        if not self._delete(self._target(key), self._CRED_TYPE_GENERIC, 0):
            err = self._ct.get_last_error()
            if err != self._ERROR_NOT_FOUND:
                raise VaultError(f"Windows Credential Manager delete failed (error {err})")


def _run(cmd: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=20)


class MacKeychainBackend:
    """The macOS login Keychain through ``security``. The secret goes over stdin, never on a command line."""
    name = "macos-keychain"

    def __init__(self, exe: str) -> None:
        self.exe = exe

    def get(self, key: str) -> str | None:
        r = _run([self.exe, "find-generic-password", "-s", SERVICE, "-a", key, "-w"])
        if r.returncode == 44:                      # errSecItemNotFound
            return None
        if r.returncode != 0:
            raise VaultError(f"macOS Keychain read failed (security exit {r.returncode})")
        return r.stdout.rstrip("\n")

    def set(self, key: str, secret: str) -> None:
        # `security -i` reads commands from stdin; secrets are validated to contain no quotes or whitespace.
        line = f'add-generic-password -U -s "{SERVICE}" -a "{key}" -l "llmtap: {key}" -w "{secret}"\n'
        r = _run([self.exe, "-i"], stdin=line)
        if r.returncode != 0 or "error" in (r.stderr or "").lower():
            raise VaultError(f"macOS Keychain write failed (security exit {r.returncode})")

    def delete(self, key: str) -> None:
        r = _run([self.exe, "delete-generic-password", "-s", SERVICE, "-a", key])
        if r.returncode not in (0, 44):
            raise VaultError(f"macOS Keychain delete failed (security exit {r.returncode})")


class SecretToolBackend:
    """The Linux Secret Service (GNOME Keyring, KWallet...) through ``secret-tool``; secrets go over stdin."""
    name = "secret-service"

    def __init__(self, exe: str) -> None:
        self.exe = exe

    def get(self, key: str) -> str | None:
        r = _run([self.exe, "lookup", "service", SERVICE, "account", key])
        if r.returncode != 0:
            return None if not r.stderr.strip() else self._fail("read", r)
        return r.stdout.rstrip("\n") or None

    def set(self, key: str, secret: str) -> None:
        r = _run([self.exe, "store", f"--label=llmtap: {key}", "service", SERVICE, "account", key], stdin=secret)
        if r.returncode != 0:
            self._fail("write", r)

    def delete(self, key: str) -> None:
        r = _run([self.exe, "clear", "service", SERVICE, "account", key])
        if r.returncode != 0 and r.stderr.strip():
            self._fail("delete", r)

    @staticmethod
    def _fail(what: str, r: subprocess.CompletedProcess) -> Any:
        raise VaultError(f"Secret Service {what} failed (secret-tool exit {r.returncode})")


_MEMORY = MemoryBackend()     # shared by every Vault in this process when LLMTAP_VAULT_BACKEND=memory


def _secret_tool_works(exe: str) -> bool:
    """secret-tool exists on many headless boxes without a running Secret Service; probe with a harmless lookup."""
    try:
        r = _run([exe, "lookup", "service", SERVICE, "account", "__llmtap_probe__"])
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode in (0, 1) and "error" not in (r.stderr or "").lower()


def make_backend(kind: str | None = None, home: Path | None = None) -> SecretBackend:
    kind = (kind or os.environ.get("LLMTAP_VAULT_BACKEND") or "auto").strip().lower()
    home = home or llmtap_home()
    if kind == "memory":
        return _MEMORY
    if kind == "file":
        return FileBackend(home / "keys.secret.json")
    if kind not in ("auto", "os"):
        raise VaultError(f"LLMTAP_VAULT_BACKEND: expected auto, os, file or memory, got {kind!r}")
    if sys.platform == "win32":
        return WindowsBackend()
    if sys.platform == "darwin" and shutil.which("security"):
        return MacKeychainBackend(shutil.which("security") or "security")
    st = shutil.which("secret-tool")
    if st and _secret_tool_works(st):
        return SecretToolBackend(st)
    if kind == "os":
        raise VaultError("no OS credential store found (install secret-tool, or set LLMTAP_VAULT_BACKEND=file)")
    return FileBackend(home / "keys.secret.json")


# ----------------------------------------------------------------------------------------------------- metadata

@dataclass
class KeyInfo:
    """Everything llmtap knows about a stored key, except the secret itself."""
    name: str
    provider: str
    preview: str = ""
    base: str = ""                     # the only upstream (scheme://host) this key may be sent to
    created: float = field(default_factory=time.time)
    rev: int = 1                       # bumped whenever the secret changes
    uid: str = field(default_factory=lambda: uuid.uuid4().hex)   # new on every add: other processes drop caches
    daily_usd: float = 0.0             # 0 = no daily cap
    total_usd: float = 0.0             # 0 = no lifetime cap (counted from `created`)
    models: list[str] = field(default_factory=list)   # allowed model patterns (fnmatch, case-insensitive); [] = any
    expires: float = 0.0               # epoch seconds; 0 = never
    paused: bool = False

    @property
    def placeholder(self) -> str:
        return PLACEHOLDER_PREFIX + self.name

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["placeholder"] = self.placeholder
        return d


def mask(secret: str) -> str:
    """A preview that identifies a key without revealing it: ``sk-p…a1b2``."""
    s = secret.strip()
    if len(s) >= 16:
        return f"{s[:4]}…{s[-4:]}"
    return f"…{s[-2:]}" if len(s) >= 6 else "…"


def _validate_name(name: str) -> str:
    n = (name or "").strip().lower()
    if not NAME_RE.match(n):
        raise VaultError("key name: use 1-40 lowercase letters, digits, '-' or '_', starting with a letter or digit")
    return n


def _validate_secret(secret: str) -> str:
    s = (secret or "").strip()
    if not s:
        raise VaultError("the key is empty")
    if len(s) > MAX_SECRET_LEN:
        raise VaultError(f"the key is longer than {MAX_SECRET_LEN} characters")
    if any(not (33 <= ord(c) <= 126) or c in "\"'\\" for c in s):
        raise VaultError("the key contains spaces, quotes, non-ASCII or control characters; paste only the key itself")
    if s.startswith(PLACEHOLDER_PREFIX):
        raise VaultError("that is a llmtap placeholder, not a real API key")
    return s


def _validate_limits(limits: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in limits.items():
        if v is None:
            continue
        if k in ("daily_usd", "total_usd"):
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise VaultError(f"{k}: expected a number of dollars") from None
            if not (f >= 0) or f == float("inf"):
                raise VaultError(f"{k}: expected 0 (no cap) or a positive number")
            out[k] = round(f, 6)
        elif k == "expires":
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise VaultError("expires: expected a timestamp (0 = never)") from None
            if f < 0:
                raise VaultError("expires: expected a timestamp (0 = never)")
            out[k] = f
        elif k == "models":
            items = v.split(",") if isinstance(v, str) else v
            if not isinstance(items, (list, tuple)):
                raise VaultError("models: expected a list of model names or patterns")
            pats = [str(m).strip() for m in items if str(m).strip()]
            if any(len(p) > 200 for p in pats) or len(pats) > 100:
                raise VaultError("models: too many or too long patterns")
            out[k] = pats
        elif k == "paused":
            if not isinstance(v, bool):
                raise VaultError("paused: expected true or false")
            out[k] = v
        else:
            raise VaultError(f"unknown limit {k!r}")
    return out


# --------------------------------------------------------------------------------------------------------- vault

class Vault:
    """Key metadata (``keys.json``) + secrets (a backend). Safe to share between threads; other processes'
    changes (CLI vs. running proxy) are picked up on the next access."""

    def __init__(self, home: Path | None = None, backend: SecretBackend | None = None) -> None:
        self.home = Path(home) if home else llmtap_home()
        self.meta_path = self.home / "keys.json"
        self._backend = backend               # opened on first secret access (listing keys never needs it)
        self._lock = threading.RLock()
        self._keys: dict[str, KeyInfo] = {}
        self._mtime: float | None = None
        self._secrets: dict[str, tuple[int, str]] = {}     # name -> (rev, secret) cache
        self._inflight: dict[str, float] = {}               # name -> reserved USD of in-flight calls

    @property
    def backend(self) -> SecretBackend:
        with self._lock:
            if self._backend is None:
                self._backend = make_backend(home=self.home)
            return self._backend

    # -- metadata file --------------------------------------------------------------------------------
    def _load(self) -> None:
        try:
            mtime = self.meta_path.stat().st_mtime
        except FileNotFoundError:
            self._keys, self._mtime = {}, None
            return
        if mtime == self._mtime:
            return
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise VaultError(f"cannot read {self.meta_path}: {e}") from None
        keys: dict[str, KeyInfo] = {}
        known = set(KeyInfo.__dataclass_fields__)
        for item in (data.get("keys") if isinstance(data, dict) else None) or []:
            if isinstance(item, dict) and NAME_RE.match(str(item.get("name", ""))):
                keys[item["name"]] = KeyInfo(**{k: v for k, v in item.items() if k in known})
        self._keys, self._mtime = keys, mtime

    def _save(self) -> None:
        body = {"version": 1, "keys": [asdict(k) for k in sorted(self._keys.values(), key=lambda k: k.name)]}
        _atomic_write(self.meta_path, json.dumps(body, indent=2) + "\n")
        try:
            self._mtime = self.meta_path.stat().st_mtime
        except OSError:
            self._mtime = None

    # -- reading ------------------------------------------------------------------------------------------
    def list(self) -> list[KeyInfo]:
        with self._lock:
            self._load()
            return sorted(self._keys.values(), key=lambda k: k.name)

    def get(self, name: str) -> KeyInfo | None:
        with self._lock:
            self._load()
            return self._keys.get((name or "").lower())

    def secret(self, name: str) -> str | None:
        """The real key, for the proxy's outgoing request only. Cached per add (``uid``), so a key removed and
        re-added by another process (CLI vs. running proxy) is never served from a stale cache."""
        with self._lock:
            self._load()
            info = self._keys.get(name)
            if info is None:
                return None
            hit = self._secrets.get(name)
            if hit and hit[0] == info.uid:
                return hit[1]
        value = self.backend.get(name)
        if value:
            with self._lock:
                self._secrets[name] = (info.uid, value)
        return value

    # -- writing ------------------------------------------------------------------------------------------
    def add(self, name: str, provider: str, secret: str, base: str | None = None, **limits: Any) -> KeyInfo:
        """Store or replace a key. ``base`` is the upstream it may be sent to; for built-in providers it defaults
        to the official API (a project's llmtap.toml can't redirect a stored key), for others it is required."""
        n = _validate_name(name)
        p = (provider or "").strip().lower()
        if not NAME_RE.match(p):
            raise VaultError("provider: expected an upstream name such as openai, anthropic or gemini")
        s = _validate_secret(secret)
        b = normalize_base(base if base else DEFAULT_UPSTREAMS.get(p, {}).get("base", ""))
        if not b:
            raise VaultError(f"{p!r} is not a built-in provider: give the upstream URL this key may be sent to")
        lim = _validate_limits(limits)
        with self._lock:
            self._load()
            old = self._keys.get(n)
            self.backend.set(n, s)
            info = KeyInfo(name=n, provider=p, preview=mask(s), base=b, rev=(old.rev + 1) if old else 1,
                           created=old.created if old and old.provider == p else time.time())
            if old:                    # replacing the secret keeps the limits you set, unless you change them
                for k in ("daily_usd", "total_usd", "models", "expires", "paused"):
                    setattr(info, k, getattr(old, k))
            for k, v in lim.items():
                setattr(info, k, v)
            self._keys[n] = info
            self._secrets.pop(n, None)
            self._save()
            return info

    def update(self, name: str, **limits: Any) -> KeyInfo:
        lim = _validate_limits(limits)
        with self._lock:
            self._load()
            info = self._keys.get((name or "").lower())
            if info is None:
                raise VaultError(f"no stored key named {name!r}")
            for k, v in lim.items():
                setattr(info, k, v)
            self._save()
            return info

    def remove(self, name: str) -> bool:
        n = (name or "").lower()
        with self._lock:
            self._load()
            if n not in self._keys:
                return False
            self.backend.delete(n)
            del self._keys[n]
            self._secrets.pop(n, None)
            self._save()
            return True

    # -- wiring -------------------------------------------------------------------------------------------
    def child_env(self, current: Mapping[str, str]) -> dict[str, str]:
        """Placeholders for ``llmtap wrap``: one per provider env var that has a stored key.

        A key named exactly like its provider wins, else the alphabetically first one. A variable that already
        holds a placeholder is left alone (you picked a key explicitly)."""
        chosen: dict[str, KeyInfo] = {}
        for k in self.list():
            if k.provider not in PROVIDER_ENV:
                continue
            cur = chosen.get(k.provider)
            if cur is None or (k.name == k.provider and cur.name != cur.provider):
                chosen[k.provider] = k
        out: dict[str, str] = {}
        for prov, info in chosen.items():
            for var in PROVIDER_ENV[prov]:
                if not str(current.get(var, "")).startswith(PLACEHOLDER_PREFIX):
                    out[var] = info.placeholder
        return out

    # -- spend reservations (per-key caps) ---------------------------------------------------------------
    def admit(self, info: KeyInfo, models: list[str | None], estimate: float, priced: bool,
              spent: Callable[[KeyInfo], tuple[float, float]]) -> Refusal | None:
        """Check ``info``'s limits and, if allowed and priced, reserve ``estimate`` atomically.

        ``spent(info)`` returns (spent today, spent in total); it runs under the lock, so a call finishing in
        between (row written, then reservation released) is counted at least once. Call :meth:`release` with the
        same estimate once the finished call is stored."""
        with self._lock:
            today, total = spent(info) if priced and (info.daily_usd or info.total_usd) else (0.0, 0.0)
            refusal = check_limits(info, models, today, total, self._inflight.get(info.name, 0.0), priced,
                                   estimate=estimate)
            if refusal is None and priced:
                self._inflight[info.name] = self._inflight.get(info.name, 0.0) + max(0.0, estimate)
            return refusal

    def release(self, name: str, usd: float) -> None:
        with self._lock:
            left = self._inflight.get(name, 0.0) - max(0.0, usd)
            if left > 1e-12:
                self._inflight[name] = left
            else:
                self._inflight.pop(name, None)

    def inflight(self, name: str) -> float:
        with self._lock:
            return self._inflight.get(name, 0.0)


# ------------------------------------------------------------------------------------------- placeholder swap

@dataclass
class Found:
    name: str
    where: str          # header name (lowercase) or "?<param>"


def find_placeholder(headers: Mapping[str, str], path: str) -> Found | None:
    """The first ``llmtap-key-<name>`` placeholder in a credential header or the ``?key=`` parameter."""
    for h in PLACEHOLDER_HEADERS:
        v = headers.get(h)
        if v and PLACEHOLDER_PREFIX in v:
            m = PLACEHOLDER_RE.search(v)
            if m:
                return Found(m.group(1), h)
    if "?" in path and PLACEHOLDER_PREFIX in path:
        for k, v in parse_qsl(urlsplit(path).query, keep_blank_values=True):
            if k.lower() in PLACEHOLDER_QUERY:
                m = PLACEHOLDER_RE.fullmatch(v.strip())
                if m:
                    return Found(m.group(1), "?" + k)
    return None


def substitute(headers: dict[str, str], path: str, found: Found, secret: str) -> tuple[dict[str, str], str]:
    """Copies of ``headers`` / ``path`` with the placeholder replaced by the real key (in that one place)."""
    ph = PLACEHOLDER_PREFIX + found.name
    hdrs = dict(headers)
    if not found.where.startswith("?"):
        hdrs[found.where] = PLACEHOLDER_RE.sub(lambda m: secret if m.group(1) == found.name else m.group(0),
                                               hdrs[found.where])
        return hdrs, path
    param = found.where[1:]
    base, _, query = path.partition("?")
    parts = []
    for piece in query.split("&"):
        k, eq, v = piece.partition("=")
        if eq and k == param and v.strip() == ph:
            piece = f"{k}={quote(secret, safe='')}"
        parts.append(piece)
    return hdrs, base + "?" + "&".join(parts)


@dataclass
class Refusal:
    status: int
    message: str
    err_type: str


def _local_midnight(now: float | None = None) -> float:
    t = time.localtime(now if now is not None else time.time())
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


_COUNT_TOKENS_RE = re.compile(r"(?:/v1/messages/count_tokens|:countTokens)$")


def has_limits(info: KeyInfo) -> bool:
    return bool(info.models or info.daily_usd or info.total_usd)


def endpoint_refusal(info: KeyInfo, method: str, upstream_path: str, generation: bool) -> Refusal | None:
    """A key with model or spend limits may only reach endpoints llmtap can check: generation calls (priced and
    model-checked), reads, and token counting. Anything else (batches, images, audio, fine-tuning, long-running
    jobs...) could spend money or use other models without being counted, so it is refused."""
    if generation or not has_limits(info) or method.upper() in ("GET", "HEAD", "OPTIONS"):
        return None
    if _COUNT_TOKENS_RE.search(upstream_path.split("?", 1)[0]):
        return None
    return Refusal(403, f"llmtap: the stored key {info.name!r} has model or spend limits, and llmtap can't check "
                        f"them for {method.upper()} {upstream_path.split('?', 1)[0]}. Use a key without limits for "
                        "this endpoint.", "llmtap_key_endpoint_not_allowed")


def normalize_base(base: str) -> str:
    """``scheme://host[:port]`` of an upstream base URL, lowercase ("" if it isn't an http(s) URL)."""
    try:
        u = urlsplit((base or "").strip())
    except ValueError:
        return ""
    if u.scheme.lower() not in ("http", "https") or not u.netloc:
        return ""
    return f"{u.scheme.lower()}://{u.netloc.lower()}"


def base_allows(info: KeyInfo, upstream_base: str) -> bool:
    """Is ``upstream_base`` the upstream ``info`` was stored for? (``{region}`` matches any AWS region.)"""
    want = info.base or normalize_base(DEFAULT_UPSTREAMS.get(info.provider, {}).get("base", ""))
    got = normalize_base(upstream_base)
    if not want or not got:
        return False
    if "{region}" in want:
        pattern = re.escape(want).replace(re.escape("{region}"), r"[a-z]{2}(?:-[a-z]+){1,2}-\d{1,2}")
        return re.fullmatch(pattern, got) is not None
    return want == got


def caller_refusal(headers: Mapping[str, str], client: str | None, host_ok: bool) -> Refusal | None:
    """Stored keys are for programs on this machine. Refuse browsers on other sites (they could otherwise spend
    the key, since ``llmtap-key-openai`` is easy to guess), DNS-rebinding hosts, and remote clients."""
    origin = (headers.get("origin") or "").strip()
    site = (headers.get("sec-fetch-site") or "").strip().lower()
    host = (headers.get("host") or "").strip().lower()
    cross = (site not in ("", "none", "same-origin")) or (
        origin and (origin == "null" or urlsplit(origin).netloc.lower() != host))
    if cross:
        return Refusal(403, "llmtap: stored keys can't be used from a web page on another site.",
                       "llmtap_key_cross_origin")
    if not host_ok:
        return Refusal(403, "llmtap: stored keys are only used for requests to localhost / 127.0.0.1 "
                            "(possible DNS-rebinding attack).", "llmtap_key_bad_host")
    if client:
        try:
            local = ipaddress.ip_address(client.split("%", 1)[0]).is_loopback
        except ValueError:
            local = False
        if not local:
            return Refusal(403, "llmtap: stored keys are only used for programs on this computer, not for "
                                f"requests from {client}.", "llmtap_key_remote_client")
    return None


def model_allowed(info: KeyInfo, model: str | None) -> bool:
    if not info.models:
        return True
    m = (model or "").lower()
    return bool(m) and any(fnmatch.fnmatchcase(m, p.lower()) for p in info.models)


def check_limits(info: KeyInfo, models: list[str | None] | str | None, spent_today: float, spent_total: float,
                 pending: float, priced: bool, now: float | None = None,
                 estimate: float | None = None) -> Refusal | None:
    """Why a call through stored key ``info`` must be refused, or None.

    ``models``: every model the request names (path and body); all must be allowed. Spend caps apply to priced
    (generation) calls and work like the global budgets: refused once what was spent plus what is in flight
    (``pending``) reaches the cap. A capped key refuses models llmtap can't price (``estimate`` 0), since a cap
    that can't count is no cap."""
    now = time.time() if now is None else now
    names = [models] if models is None or isinstance(models, str) else list(models)
    named = [m for m in names if m] or [None]
    if info.paused:
        return Refusal(403, f"llmtap: the stored key {info.name!r} is paused. Resume it in the llmtap dashboard "
                            f"(Keys) or with `llmtap keys limit {info.name} --resume`.", "llmtap_key_paused")
    if info.expires and now >= info.expires:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(info.expires))
        return Refusal(403, f"llmtap: the stored key {info.name!r} expired on {when}.", "llmtap_key_expired")
    bad = [m for m in named if not model_allowed(info, m)] if priced else []
    if bad:
        allowed = ", ".join(info.models)
        return Refusal(403, f"llmtap: the stored key {info.name!r} may only be used with: {allowed} "
                            f"(this request asked for {bad[0] or 'no model'}).", "llmtap_key_model_not_allowed")
    if priced and (info.daily_usd or info.total_usd) and estimate is not None and estimate <= 0:
        return Refusal(403, f"llmtap: the stored key {info.name!r} has a spend limit, but llmtap has no price for "
                            f"{named[0] or 'this request'}, so it can't count what the call costs. Add a price "
                            "under [pricing_overrides] or remove the key's spend limit.", "llmtap_key_unpriced_model")
    if priced:
        if info.daily_usd and spent_today + pending >= info.daily_usd:
            return Refusal(429, f"llmtap: daily limit of ${info.daily_usd:.2f} for the stored key {info.name!r} "
                                f"reached (${spent_today:.4f} spent today). Raise it in the llmtap dashboard (Keys) "
                                "or wait until midnight.", "llmtap_key_limit")
        if info.total_usd and spent_total + pending >= info.total_usd:
            return Refusal(429, f"llmtap: total limit of ${info.total_usd:.2f} for the stored key {info.name!r} "
                                f"reached (${spent_total:.4f} spent). Raise it in the llmtap dashboard (Keys).",
                           "llmtap_key_limit")
    return None
